# Code walkthrough

An annotated guide to the main components of the codebase: what each part does,
how the parts connect, and the reasoning behind the less obvious lines. Code
excerpts are taken from the repository; long excerpts are abridged with `...`.
A glossary of terms is at the end.

## Contents

1. [System overview](#1-system-overview)
2. [Dataset construction](#2-dataset-construction)
3. [Training notebook](#3-training-notebook)
4. [Judge evaluation](#4-judge-evaluation)
5. [Demo application](#5-demo-application)
6. [Glossary](#6-glossary)

---

## 1. System overview

The system has four stages, each consuming the previous stage's output.

```
 Dataset construction      Training                Evaluation               Demo
 pipeline/*.py             notebooks/…v0_3.ipynb   notebook, evaluation/    app/*.py

 passages ──► examples ──► LoRA adapter (.npz) ──► metrics, judge scores    adapter ──► web UI
```

1. **Dataset construction** turns 1,586 destination passages into about 7,800
   examples. Deterministic templates produce some; a teacher model writes the
   rest. Every example is validated before it is kept.
2. **Training** fine-tunes Gemma 3 1B with LoRA, producing a 5 MB adapter.
3. **Evaluation** scores the baseline and fine-tuned models on the same held-out
   items, then an LLM judge checks each factual claim against its passage.
4. **The demo** loads Gemma with the adapter and serves it through a web interface.

One principle runs through every stage: **each prompt contains its source
passage**, and the model is trained to answer only from it.

---

## 2. Dataset construction

| Module | Responsibility |
|---|---|
| `ijeoma_clean.py` | Clean raw records into passages; assign labels |
| `ijeoma_pairs.py` | Template examples: classification, extractive answers, refusals |
| `ijeoma_teacher.py` | Teacher-generated digests, question-answer pairs and dialogues |
| `ijeoma_build.py` | Validation, passage-level train/test split, output files |
| `ijeoma_summarize.py`, `ijeoma_build_summarization.py` | Extractive summarisation set |

### 2.1 Permitted labels in the classification prompt (`ijeoma_pairs.py`)

```python
THEMES = ["Town or Area Guide", "Nature & Wildlife", "Museum & Gallery", ...]
TERRAINS = ["Coastal", "Riverine", "Island", ...]
PROFILES = ["General, base for exploring", "Nature and wildlife, outdoor", ...]
```

The closed label sets, as they appear in the dataset. The model must reproduce
one of these strings exactly.

```python
LABEL_MENU = (
    "Allowed values (copy one exactly):\n"
    f"theme: {' | '.join(THEMES)}\n"
    f"terrain: {' | '.join(TERRAINS)} | null\n"
    f"traveler_profile: {' | '.join(PROFILES)}\n"
    "world_heritage: true | false"
)
```

`' | '.join(THEMES)` renders the list as `Town or Area Guide | Nature & Wildlife | …`.
Terrain also allows `null`, since 645 passages have no terrain label.

```python
"input": (
    f"{TASK_TAGS['classify']} {instr}\n{LABEL_MENU}\n\n"
    f"PASSAGE:\n{row['passage']}"
),
```

The classification prompt: task tag, instruction, label menu, then the passage.
Without the menu, the model invented its own label wording and scored 0 on theme
accuracy (decision D-09).

### 2.2 The OpenAI teacher (`ijeoma_teacher.py`)

The teacher is a larger model that writes training examples from a passage. The
module originally targeted Gemini; `OpenAITeacher` adds an OpenAI backend with
the same interface, so the rest of the module is backend-agnostic.

```python
class OpenAITeacher:
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str):
        self.key = get_api_key(("OPENAI_API_KEY",))
        self.model = model
        self.lock = threading.Lock()
        self.calls = 0
```

The constructor reads the key (from Colab Secrets or the environment) and keeps
a call counter guarded by a lock, since generation runs on several threads.

```python
body = json.dumps(
    {
        "model": self.model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
).encode()
```

`response_format: json_object` constrains the reply to valid JSON. Every teacher
prompt requests JSON, so this eliminates unparseable responses.

```python
for attempt in range(retries):
    try:
        ...
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
        ...
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:
        if attempt == retries - 1:
            ...
            return None
        time.sleep(delay)
        delay *= 2
```

Up to four attempts with exponential backoff (2, 4, 8 s) absorb rate limits and
transient server errors. The request uses the standard library (`urllib`), so no
additional dependency is needed.

### 2.3 Multi-turn dialogues (`ijeoma_teacher.py`)

`DIALOGUE_PROMPT` asks the teacher for a three-exchange traveller–guide
conversation in which every guide reply is drawn from the passage, and the
third question asks for something the passage does not cover (cost, hours,
transport, safety), which the guide declines. `dialogue_pairs` converts one
conversation into training examples:

```python
def dialogue_pairs(row, raw) -> list[dict]:
    obj = parse_json(raw)
    if not obj or obj.get("skip"):
        return []
```

Teacher replies that fail to parse, or in which the teacher declined because the
passage was too thin, produce no examples.

```python
    out, history = [], []
    for i, t in enumerate(turns):
        q, a = str(t["traveller"]).strip(), str(t["guide"]).strip()
        answerable = bool(t.get("answerable", True))
        convo = "\n".join(history) if history else "(start of conversation)"
```

`history` accumulates the dialogue so far. The first turn has none and is marked
`(start of conversation)`.

```python
        out.append(
            {
                "input": (
                    f"{TASK_TAGS['answer']} {DIALOGUE_INSTRUCTION}\n\n"
                    f"CONVERSATION SO FAR:\n{convo}\n\n"
                    f"QUESTION: {q}\n\nPASSAGE:\n{row['passage']}"
                ),
                "output": a,
                "task": "answer" if answerable else "refuse",
                ...
            }
        )
        history += [f"Traveller: {q}", f"Guide: {a}"]
```

Each guide turn becomes one example whose input contains the conversation so
far, the new question and the passage. This preserves the single input/output
format used for training while conditioning each reply on the dialogue history.
The final, unanswerable turn is labelled as a refusal.

The run loop builds a job list of (task, passage) pairs, skips any already in
the cache (`teacher_cache.jsonl`), and executes the remainder in a thread pool.
Because every response is cached, re-running the pipeline makes no API calls.

### 2.4 Validation of teacher examples (`ijeoma_build.py`)

`validate(pair)` returns a list of failure codes; an empty list means the example
is kept. Earlier rules check structure, length, contact details, and that every
number and proper noun appears in the passage. Rule 6 applies to teacher
examples only:

```python
if str(pair.get("subtype", "")).startswith(("teacher", "dialogue")):
    refuses = bool(RE_REFUSAL.search(out))
    if task == "refuse" and not refuses:
        fails.append("refusal_label_mismatch")
    if task in GROUNDED_TASKS and refuses:
        fails.append("refusal_label_mismatch")
```

`RE_REFUSAL` is a regular expression matching refusal phrasing ("does not
cover", "passage does not", "I cannot", …). An example labelled as a refusal must
refuse, and an example labelled as an answer must not; a contradiction would
train the opposite behaviour. The same pattern is used to score refusals during
evaluation.

```python
    words = re.findall(r"[a-z]+", out.lower())
    if words and sum(w in ENGLISH_MARKERS for w in words) / len(words) < 0.08:
        fails.append("not_english")
```

A lightweight language check: in English prose, common function words (`the`,
`and`, `is`, …) make up roughly a quarter to a third of tokens. Below 8%, the text
is almost certainly in another language. This rejected four Italian outputs.

---

## 3. Training notebook

`notebooks/ijeoma_train_and_evaluate_v0_3.ipynb` loads the data, scores the
baseline, trains, scores again and logs the run. The notable components are below.

### 3.1 Locating the working folder (section 2)

```python
_hits = [
    d
    for d in ["/content/drive/MyDrive/Ijeoma"]
    + glob.glob("/content/drive/MyDrive/*/Ijeoma")
    + glob.glob("/content/drive/MyDrive/*/*/Ijeoma")
    if os.path.exists(f"{d}/conversational_training_data.jsonl")
]
```

The notebook searches for a Drive folder named `Ijeoma` at the top level or up to
two levels down, and accepts it only if it contains the training data. If none
is found it raises an error explaining what is missing. This avoids hard-coding
any one user's Drive layout.

### 3.2 Run configuration (section 2)

```python
SMOKE_TEST = False
TRAIN_SUBSET = 200 if SMOKE_TEST else None  # None: use every training example
EVAL_N = 20 if SMOKE_TEST else 100
CHECKPOINT_EVERY = 500
```

`SMOKE_TEST` switches between a 15-minute end-to-end check and the full run.

### 3.3 Periodic checkpoints (section 9)

```python
class SaveLoRA(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.every and self.step % self.every == 0:
            ws = [
                w
                for w in self.model.weights
                if "lora" in (getattr(w, "path", None) or w.name).lower()
            ]
            np.savez(
                path,
                names=...,
                step=self.step,
                **{f"w{j}": np.asarray(w) for j, w in enumerate(ws)},
            )
```

A Keras callback that runs after every training step. Every 500 steps it
collects the LoRA weights (the only weights that change during training) and
writes them, with their names, to a `.npz` file on Drive. A disconnect therefore
loses at most 500 steps.

### 3.4 Exporting prompts with predictions (section 7)

```python
row = {"task": task, ..., "pred": pred, "ref": rec["output"], "input": rec["input"]}
```

Each evaluation row stores the full prompt alongside the prediction and the
reference, so the LLM judge can run afterwards from the CSV alone (D-20).

### 3.5 Additional metrics (section 10b)

```python
def rouge_1(pred, ref):
    p, r = _C(pred.lower().split()), _C(ref.lower().split())
    overlap = sum((p & r).values())
    ...
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)
```

ROUGE-1 F1: word counts for the prediction and reference, their multiset
intersection, then precision, recall and their harmonic mean.

```python
rep = classification_report(cl.true_theme, cl.pred_theme, output_dict=True, zero_division=0)
summary["classify_theme_macroP"] = round(rep["macro avg"]["precision"], 3)
```

Per-class precision and recall for theme classification via scikit-learn,
macro-averaged so that rare themes count as much as common ones.

```python
_, _, f1 = bscore(
    list(gen.pred), list(gen.ref), lang="en", model_type="distilbert-base-uncased", device="cpu"
)
```

BERTScore compares meaning rather than surface wording, using contextual
embeddings from DistilBERT. It runs on the CPU because the GPU is occupied by
the model under evaluation.

---

## 4. Judge evaluation

`evaluation/run_judge.py` computes the project's primary metric,
`claims_supported_rate`, together with 1–5 rubric scores.

```python
from ijeoma_build import RE_REFUSAL, extract_passage
```

The refusal pattern and passage extraction are imported from the pipeline, so
evaluation and validation share one definition of a refusal.

```python
JUDGE_PROMPT = """You are grading a travel assistant's answer ...
Also list every atomic factual claim in the answer (one fact each: a date, a
builder, an event and a location are separate claims) and mark whether the
passage directly supports it. ..."""
```

The judge is asked to **list** each atomic claim with a supported/unsupported
mark, rather than to report counts. In testing, directly requested counts were
unreliable; counting a list in code is not.

```python
def call_judge(task, passage, answer, model, api_key, retries=4) -> dict | None:
```

Sends one item to the judge with JSON output enforced and retries transient
network failures with exponential backoff. The passage is truncated at 6,000
characters.

```python
def judge_item(row, model, api_key) -> dict | None:
    prompt = row["input"]
    task_text = prompt[: prompt.find("PASSAGE:")].strip()
    verdict = call_judge(task_text, extract_passage(prompt), str(row["pred"]), model, api_key)
    ...
    claims = verdict.get("claims") or []
    return {
        ...
        "claims_total": len(claims),
        "claims_supported": sum(bool(c.get("supported")) for c in claims),
        "unsupported": json.dumps([c.get("claim") for c in claims if not c.get("supported")]),
    }
```

The stored prompt is split into the task text (before `PASSAGE:`) and the
passage. The verdict is flattened into one record per item; unsupported claims
are kept verbatim for manual review.

```python
is_refusal = df["ref"].astype(str).map(lambda s: bool(RE_REFUSAL.search(s)))
eligible = df[df["task"].isin(GROUNDED_TASKS) & ~is_refusal]
```

Only answers, digests and summaries whose reference is not a refusal are judged;
claim support is undefined for a refusal.

```python
total, supported = results.claims_total.sum(), results.claims_supported.sum()
"claims_supported_rate": round(float(supported / total), 3)
```

The primary metric is pooled over all judged items: supported claims divided by
all claims (261 / 268 = 0.974 for the reported run).

---

## 5. Demo application

| Module | Responsibility |
|---|---|
| `app/prompts.py` | Builds the exact prompt for each task |
| `app/model.py` | Loads Gemma and the adapter; generates text |
| `app/ijeoma_app.py` | Page layout, event handlers, command-line interface |

### 5.1 Prompt construction (`prompts.py`)

```python
from ijeoma_pairs import ANSWER_INSTRUCTION, CLASSIFY_INSTRUCTIONS, LABEL_MENU, TASK_TAGS
from ijeoma_summarize import INSTR_NOTABLE, INSTR_ONE_LINE, INSTR_SUMMARY
from ijeoma_teacher import DIALOGUE_INSTRUCTION, DIGEST_INSTRUCTIONS
```

Instruction wording is imported from the dataset pipeline, not restated. A
fine-tuned 1B model is sensitive to paraphrase, so the demo must send the
trained formats exactly; all six prompt types were verified byte-for-byte
against training inputs.

```python
def _passage(text: str | None) -> str:
    text = (text or "").strip()
    return text or NO_PASSAGE
```

An empty passage becomes `(none available)`, the marker used in training for
questions with no source.

```python
def dialogue(question: str, passage: str, history: list[tuple[str, str]]) -> str:
    lines = []
    for q, a in history:
        lines += [f"Traveller: {q}", f"Guide: {a}"]
    convo = "\n".join(lines) if lines else "(start of conversation)"
```

Renders the chat history in the `Traveller:` / `Guide:` format used by
`dialogue_pairs` during dataset construction.

```python
TASKS = {
    "Classify": lambda p, q, s: classify(p),
    "Ask a question": lambda p, q, s: answer(q, p),
    "Traveller digest": lambda p, q, s: digest(p),
    "Summarise": lambda p, q, s: summarize(p, s),
}
```

Maps each interface label to a builder with a uniform signature (passage,
question, style), so the interface can dispatch on the selected task.

### 5.2 Model loading and generation (`model.py`)

```python
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
```

Keras selects its backend at first import, so this is set before Keras loads.
JAX matches the training configuration; the second variable lets JAX use up to
90% of GPU memory.

```python
def chat_template(prompt: str) -> str:
    return f"{START_OF_TURN}user\n{prompt}{END_OF_TURN}\n{START_OF_TURN}model\n"
```

Gemma's chat format: a user turn containing the prompt, followed by an open model
turn for Gemma to complete. Identical to the string used in training.

```python
def extract_reply(generated: str, prompt: str) -> str:
```

KerasHub returns the prompt followed by the completion. This removes the prompt
and truncates at the first end-of-turn or end-of-sequence marker.

```python
class IjeomaModel:
    def __init__(self, preset, adapter_path, lora_rank=8):
        ...
        self._models = {"ijeoma": self._load(with_adapter=True)}
```

Holds the loaded models. The fine-tuned model loads at start-up; the baseline
loads on first use by the comparison page (`ensure_loaded`). A lock serialises
access to the GPU.

```python
model = keras_hub.models.Gemma3CausalLM.from_preset(self.preset)
if with_adapter:
    model.backbone.enable_lora(rank=self.lora_rank)
    archive = np.load(self.adapter_path)
    saved = {str(name): archive[f"w{i}"] for i, name in enumerate(archive["names"])}
```

Loads the base model, attaches empty rank-8 LoRA layers (initialised so the model
is unchanged), and reads the adapter archive into a name-to-array map: the
inverse of the checkpoint callback in §3.3.

```python
    for weight in model.weights:
        key = getattr(weight, "path", None) or weight.name
        if key in saved:
            weight.assign(saved[key])
            loaded += 1
    if loaded != len(saved):
        raise RuntimeError(...)
```

Copies each trained matrix into the corresponding model weight, then verifies
that all 104 were restored. A partial match indicates the wrong rank or base
model, and the loader fails rather than serving a partially adapted model.

```python
model.compile(sampler="greedy")
```

Greedy decoding selects the most probable token at each step, making outputs
deterministic and matching the evaluation setting.

```python
needed = n_prompt + max_new_tokens
max_length = next((b for b in LENGTH_BUCKETS if b >= needed), LENGTH_BUCKETS[-1])
```

KerasHub compiles generation separately for each distinct `max_length`. Rounding
up to one of four fixed lengths (768, 1024, 1536, 2048) bounds the number of
compilations.

```python
with self._lock:
    generated = model.generate(text, max_length=max_length, stop_token_ids=model.stop_token_ids_)
return extract_reply(generated, text), elapsed
```

Generation stops at Gemma's end-of-turn token, and requests are processed one
at a time.

`StubModel` implements the same interface with canned outputs, so the interface
can be tested without a GPU (`--stub`).

### 5.3 Pages and event handling (`ijeoma_app.py`)

```python
def run_task(task, passage, question, style, variant="ijeoma"):
    if task == "Ask a question" and not (question or "").strip():
        return "Enter a question first.", "", ""
    prompt = prompts.TASKS[task](passage, question or "", style)
    output, seconds = model.generate(prompt, variant=variant)
    ...
    return output, prompt, f"{seconds:.1f} s"
```

The shared handler for both the Workbench and Compare pages: validate input,
build the prompt, generate, and return the output, the prompt (shown in the
interface for transparency) and the generation time.

```python
picker.change(sample_passage, picker, passage)
```

A Gradio event binding: when the destination selection changes, the matching
passage is loaded into the text box.

```python
with app.route("Chat", "/chat"):
```

`Blocks.route` adds a separate page under its own URL; Gradio renders the
navigation menu between pages.

```python
def respond(text, history, passage, withhold):
    exchanges, pending = [], None
    for turn in history:
        if turn["role"] == "user":
            pending = turn["content"]
        elif pending is not None:
            exchanges.append((pending, turn["content"]))
            pending = None
    prompt = prompts.dialogue(text, "" if withhold else passage, exchanges)
```

The chat component stores history as a list of `user` and `assistant` messages.
This loop pairs each question with its reply to rebuild the
`(traveller, guide)` history, then builds the dialogue prompt, with an empty
passage if the user has chosen to withhold it. Changing the destination or
starting a new conversation clears the history, so a dialogue never mixes two
passages.

```python
def compare(task, passage, question, style):
    base, _, base_s = run_task(task, passage, question, style, variant="base")
    tuned, _, tuned_s = run_task(task, passage, question, style, variant="ijeoma")
```

The comparison page runs the identical prompt through both models.

```python
build_app(model).queue(default_concurrency_limit=1).launch(
    server_name=args.host, server_port=args.port, share=args.share
)
```

Requests are queued and processed one at a time, since a single GPU cannot
serve concurrent generations efficiently. `share=True` requests a public
`gradio.live` URL.

---

## 6. Glossary

| Term | Definition |
|---|---|
| Adapter | The saved LoRA weights (`.npz`, 5 MB); combined with base Gemma to give Ijeoma AI |
| Atomic claim | A single factual statement, such as one date or one location |
| BERTScore | Semantic similarity between output and reference, computed from contextual embeddings |
| Callback | Code that Keras invokes automatically at defined points during training |
| Checkpoint | Weights saved part-way through training |
| Claims supported rate | Supported claims ÷ all claims; the primary metric (target 0.90) |
| Epoch | One pass over the full training set |
| Example | One training item: an input prompt and its target output |
| False refusal rate | Share of answerable questions the model declined |
| Greedy decoding | Choosing the most probable token at every step; deterministic |
| Grounded | Every name and number in an output appears in the source passage |
| Held-out set | Examples from passages never used in training |
| Invention rate | Share of no-passage prompts for which the model fabricated an answer |
| JAX / Keras / KerasHub | The numerical backend, deep-learning library and model library used |
| JSONL | One JSON record per line; the dataset file format |
| LLM judge | A language model that scores outputs against a rubric |
| LoRA | Low-rank adaptation: training small added matrices instead of the full model |
| Macro average | A metric averaged over classes with equal weight per class |
| Passage | The source text describing one destination |
| Precision / recall / F1 | Correct share of predictions / found share of correct items / their harmonic mean |
| Rank | The inner dimension of the LoRA matrices (8 here); governs adapter capacity |
| Refusal recall | Share of unanswerable questions the model correctly declined |
| Regular expression | A text pattern used for matching, such as refusal phrasing |
| ROUGE-1 / ROUGE-L | Word overlap with the reference: single words / longest common subsequence |
| Route | A separate page of the Gradio application with its own URL |
| Teacher model | A larger model used to generate training examples |
| Token | The unit of text the model reads and writes, roughly three-quarters of a word |
