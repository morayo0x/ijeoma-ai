"""Loads Gemma 3 1B and the Ijeoma AI LoRA adapter, and generates text.

Two model instances are kept: the fine-tuned one, loaded at start-up, and the
untouched base model, loaded the first time the Compare page asks for it.
At 1B parameters in float32 that is about 4 GB each, so both fit on a 16 GB GPU.
Swapping adapter weights on one instance would save memory but forces a
recompile of the generate function on every switch.
"""
import os
import threading
import time

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

SOT, EOT = "<start_of_turn>", "<end_of_turn>"
# Generation length buckets. keras-hub pads to max_length and compiles once per
# distinct value, so rounding up to a few fixed sizes avoids recompiling on
# every request.
BUCKETS = (768, 1024, 1536, 2048)


def chat_template(prompt):
    """Exactly the string the training notebook built at inference time."""
    return f"{SOT}user\n{prompt}{EOT}\n{SOT}model\n"


def _strip(raw, prompt):
    out = raw[len(prompt):] if raw.startswith(prompt) else raw.replace(prompt, "")
    out = out.strip()
    for tok in (EOT, "<eos>", "<pad>"):
        i = out.find(tok)
        if i >= 0:
            out = out[:i].strip()
    for tok in (SOT + "user", SOT + "model", SOT):
        i = out.find(tok)
        if i > 0:
            out = out[:i].strip()
    return out


class Ijeoma:
    def __init__(self, preset, adapter_path, lora_rank=8):
        self.preset, self.adapter_path, self.rank = preset, adapter_path, lora_rank
        self.lock = threading.Lock()
        self.models = {}
        self.models["ijeoma"] = self._load(with_adapter=True)

    def _load(self, with_adapter):
        import keras_hub as kh
        import numpy as np

        t0 = time.time()
        m = kh.models.Gemma3CausalLM.from_preset(self.preset)
        if with_adapter:
            m.backbone.enable_lora(rank=self.rank)
            z = np.load(self.adapter_path)
            saved = {str(n): z[f"w{i}"] for i, n in enumerate(z["names"])}
            hit = 0
            for w in m.weights:
                key = getattr(w, "path", None) or w.name
                if key in saved:
                    w.assign(saved[key])
                    hit += 1
            if hit != len(saved):
                raise RuntimeError(f"adapter loaded {hit} of {len(saved)} tensors - "
                                   f"wrong rank or wrong base model?")
        m.compile(sampler="greedy")
        m.generate_function = None
        tok = m.preprocessor.tokenizer
        stops = [int(tok.token_to_id(t)) for t in (EOT, "<eos>")
                 if tok.token_to_id(t) is not None]
        m._ijeoma_stops = stops or "auto"
        print(f"loaded {'Ijeoma AI' if with_adapter else 'base'} model "
              f"in {time.time() - t0:.0f}s")
        return m

    def has(self, which):
        return which in self.models

    def ensure(self, which):
        if which not in self.models:
            with self.lock:
                if which not in self.models:
                    self.models[which] = self._load(with_adapter=(which == "ijeoma"))

    def generate(self, prompt, which="ijeoma", max_new_tokens=320):
        self.ensure(which)
        m = self.models[which]
        text = chat_template(prompt)
        n = len(m.preprocessor.tokenizer.tokenize(text))
        need = n + max_new_tokens
        max_length = next((b for b in BUCKETS if b >= need), BUCKETS[-1])
        if n >= max_length:
            return "(passage too long for the model - shorten it and try again)", 0.0
        with self.lock:
            t0 = time.time()
            raw = m.generate(text, max_length=max_length, stop_token_ids=m._ijeoma_stops)
            dt = time.time() - t0
        return _strip(raw, text), dt


class FakeIjeoma:
    """Stand-in for testing the pages without a GPU or a Gemma download."""

    def __init__(self, *a, **k):
        self.models = {"ijeoma": None}

    def has(self, which):
        return True

    def ensure(self, which):
        pass

    def generate(self, prompt, which="ijeoma", max_new_tokens=320):
        tag = prompt.split("]")[0] + "]"
        if tag == "[CLASSIFY]":
            out = ('{\n  "country": "Ghana",\n  "theme": "Fort & Castle",\n  "terrain": '
                   '"Coastal",\n  "traveler_profile": "History and heritage",\n  '
                   '"world_heritage": true\n}')
        else:
            q = prompt.split("QUESTION:")[-1].split("\n")[0].strip() if "QUESTION:" in prompt else ""
            out = f"[{which} - fake output] {tag} {q}"[:300]
        return out, 0.0
