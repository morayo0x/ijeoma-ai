"""Model loading and text generation for the Ijeoma AI demo.

Two model instances are kept: the fine-tuned model, loaded at start-up, and the
unmodified base model, loaded on first use by the comparison page. In float32 a
1B-parameter Gemma takes about 4 GB, so both fit on a 16 GB GPU. Swapping adapter
weights on a single instance would save memory but forces the generate function
to recompile on every switch.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Literal

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

Variant = Literal["ijeoma", "base"]

START_OF_TURN, END_OF_TURN = "<start_of_turn>", "<end_of_turn>"

# keras-hub pads every generation to max_length and compiles once per distinct
# value, so rounding requests up to a few fixed lengths bounds recompilation.
LENGTH_BUCKETS = (768, 1024, 1536, 2048)


def chat_template(prompt: str) -> str:
    """Wrap a prompt in Gemma's chat format, exactly as during training."""
    return f"{START_OF_TURN}user\n{prompt}{END_OF_TURN}\n{START_OF_TURN}model\n"


def extract_reply(generated: str, prompt: str) -> str:
    """Return only the model's turn from the full generated sequence."""
    if generated.startswith(prompt):
        reply = generated[len(prompt) :]
    else:
        reply = generated.replace(prompt, "")
    reply = reply.strip()
    for marker in (END_OF_TURN, "<eos>", "<pad>"):
        cut = reply.find(marker)
        if cut >= 0:
            reply = reply[:cut].strip()
    for marker in (START_OF_TURN + "user", START_OF_TURN + "model", START_OF_TURN):
        cut = reply.find(marker)
        if cut > 0:
            reply = reply[:cut].strip()
    return reply


class IjeomaModel:
    """Gemma 3 1B with and without the Ijeoma AI LoRA adapter."""

    def __init__(
        self, preset: str, adapter_path: str, lora_rank: int = 8, dtype: str = "float32"
    ) -> None:
        self.preset = preset
        self.dtype = dtype
        self.adapter_path = adapter_path
        self.lora_rank = lora_rank
        self._lock = threading.Lock()
        self._models: dict[str, object] = {"ijeoma": self._load(with_adapter=True)}

    def _load(self, with_adapter: bool):
        import keras_hub
        import numpy as np

        started = time.time()
        model = keras_hub.models.Gemma3CausalLM.from_preset(self.preset, dtype=self.dtype)
        if with_adapter:
            model.backbone.enable_lora(rank=self.lora_rank)
            archive = np.load(self.adapter_path)
            saved = {str(name): archive[f"w{i}"] for i, name in enumerate(archive["names"])}
            loaded = 0
            for weight in model.weights:
                key = getattr(weight, "path", None) or weight.name
                if key in saved:
                    weight.assign(saved[key].astype(weight.dtype))
                    loaded += 1
            if loaded != len(saved):
                raise RuntimeError(
                    f"adapter: loaded {loaded} of {len(saved)} tensors; "
                    "check the LoRA rank and the base model preset"
                )
        model.compile(sampler="greedy")
        model.generate_function = None

        tokenizer = model.preprocessor.tokenizer
        stop_ids = [
            int(tokenizer.token_to_id(t))
            for t in (END_OF_TURN, "<eos>")
            if tokenizer.token_to_id(t) is not None
        ]
        model.stop_token_ids_ = stop_ids or "auto"
        name = "Ijeoma AI" if with_adapter else "base"
        print(f"loaded {name} model in {time.time() - started:.0f}s")
        return model

    def ensure_loaded(self, variant: Variant) -> None:
        if variant not in self._models:
            with self._lock:
                if variant not in self._models:
                    self._models[variant] = self._load(with_adapter=variant == "ijeoma")

    def generate(
        self, prompt: str, variant: Variant = "ijeoma", max_new_tokens: int = 320
    ) -> tuple[str, float]:
        """Generate a reply. Returns (text, seconds)."""
        self.ensure_loaded(variant)
        model = self._models[variant]
        text = chat_template(prompt)
        n_prompt = len(model.preprocessor.tokenizer.tokenize(text))
        needed = n_prompt + max_new_tokens
        max_length = next((b for b in LENGTH_BUCKETS if b >= needed), LENGTH_BUCKETS[-1])
        if n_prompt >= max_length:
            return "The passage is too long for the model. Shorten it and try again.", 0.0

        with self._lock:
            started = time.time()
            generated = model.generate(
                text, max_length=max_length, stop_token_ids=model.stop_token_ids_
            )
            elapsed = time.time() - started
        return extract_reply(generated, text), elapsed


class StubModel:
    """Drop-in replacement that returns canned text; for testing the UI without a GPU."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def ensure_loaded(self, variant: Variant) -> None:
        pass

    def generate(
        self, prompt: str, variant: Variant = "ijeoma", max_new_tokens: int = 320
    ) -> tuple[str, float]:
        tag = prompt.split("]", 1)[0] + "]"
        if tag == "[CLASSIFY]":
            return (
                '{\n  "country": "Ghana",\n  "theme": "Fort & Castle",\n'
                '  "terrain": "Coastal",\n  "traveler_profile": "History and heritage",\n'
                '  "world_heritage": true\n}',
                0.0,
            )
        question = ""
        if "QUESTION:" in prompt:
            question = prompt.split("QUESTION:")[-1].split("\n")[0].strip()
        return f"[stub {variant}] {tag} {question}".strip(), 0.0
