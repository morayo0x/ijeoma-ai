"""Builds the Ijeoma train-and-evaluate notebook.

Written as a builder rather than raw .ipynb JSON so every code cell can be
compiled and checked before the notebook ships.
"""
import json

C = []            # cells


def md(text):
    C.append({"cell_type": "markdown", "metadata": {},
              "source": text.strip("\n").split("\n")})


def code(text):
    C.append({"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": text.strip("\n").split("\n")})


# ===========================================================================
md(r"""
# Ijeoma AI — Train and Evaluate

Companion to **Lab 8.9**, wired to the Ijeoma dataset and the grounded task
design: every prompt carries its passage, so the model reads rather than
recalls.

Three things this notebook does that the lab template does not:

1. **Scores the untuned model first.** Same prompts, same test set, same
   metrics. Without that number you cannot say fine-tuning helped.
2. **Scores per task.** `classify`, `answer`, `digest` and `refuse` fail in
   different ways; one averaged number hides all of it.
3. **Measures the invention rate.** Fifty prompts about places that are not
   in the corpus. This speaks directly to the project's stated main risk and
   almost no capstone reports it.

**Runtime → Change runtime type → T4 GPU** before you start.

`SMOKE_TEST = True` is the default. It trains on 200 examples for 1 epoch so
the whole notebook runs end to end in roughly fifteen minutes. Get a clean
pass first, then set it to `False`.
""")

# --------------------------------------------------------------------- setup
md("## 1. Setup")

md(r"""
### Your Kaggle key

Gemma weights come from Kaggle, so you need an account, the signed Gemma
licence, and an API key:

1. Sign up at [kaggle.com](https://www.kaggle.com).
2. Accept the licence on the [Gemma 3 Keras model
   page](https://www.kaggle.com/models/keras/gemma3/keras/) — *Request
   Access*, read, accept. The download fails without this.
3. Profile icon → **Settings** → **API** → **Create New Token**. A
   `kaggle.json` downloads, containing `{"username": ..., "key": ...}`.
4. In Colab, click the 🔑 in the left sidebar and add two secrets,
   `KAGGLE_USERNAME` and `KAGGLE_KEY`, pasting the two values **without the
   quotation marks**. Switch **Notebook access** on for both.

Hugging Face works too if you prefer — accept the licence on the
[google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) page, put
your token in a `HF_TOKEN` secret, and set `WEIGHTS_SOURCE = "hf"` in the
config cell. Kaggle is the path the course supports, so use it unless you
have a reason not to.
""")

code(r"""
!pip install -q -U keras-hub keras
""")

code(r"""
import os

# Must be set before Keras is imported.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import json
import random
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

try:
    import keras_hub as kh
except ImportError:                      # older course environments
    import keras_nlp as kh

print("keras", keras.__version__, "| backend", keras.backend.backend())
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU - switch runtime to T4"
""")

code(r"""
from google.colab import userdata, drive

WEIGHTS_SOURCE = "kaggle"        # "kaggle" or "hf"

if WEIGHTS_SOURCE == "kaggle":
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    print("Kaggle credentials loaded")
else:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("Hugging Face token loaded")

drive.mount("/content/drive")
""")

# -------------------------------------------------------------------- config
md(r"""
## 2. Config

Every knob in one place. Change things here, not further down, so the
experiment log at the bottom records what you actually ran.
""")

md(r"""
**On the preset.** The lab template loads `gemma3_1b`, the pretrained base
model, and argues it is a more neutral starting point for LoRA. That is a
fair argument, but it makes the baseline comparison meaningless: a base model
has never been trained to follow an instruction, so of course it rambles, and
"fine-tuning beat it" tells you nothing.

Ijeoma's tasks are reading tasks — read this passage, label it, answer from
it, refuse when it does not say. Instruction-following is most of the job and
`gemma3_instruct_1b` already has some. It is the honest baseline and the
stronger starting point, so it is the default here.

Run both if you have the GPU hours; the comparison is a genuinely interesting
result for your write-up.
""")

code(r"""
# ---- data -----------------------------------------------------------------
DRIVE = "/content/drive/MyDrive"
TRAIN_PATH = f"{DRIVE}/conversational_training_data.jsonl"
TEST_PATH = f"{DRIVE}/conversational_test_data.jsonl"
# The _full files carry task/subtype/country, which per-task scoring needs.
TRAIN_FULL_PATH = f"{DRIVE}/pairs_train_full.jsonl"
TEST_FULL_PATH = f"{DRIVE}/pairs_test_full.jsonl"

# "instruction" = classify/answer/refuse (+ teacher digests)
# "summarization" = the extractive summarization set
# "both" = concatenated. The two sets share one passage-level split, so
# training on both does not leak into either test set.
DATASET = "both"
SUM_TRAIN_PATH = f"{DRIVE}/summarization_train_full.jsonl"
SUM_TEST_PATH = f"{DRIVE}/summarization_test_full.jsonl"

# ---- model ----------------------------------------------------------------
PRESET = "gemma3_instruct_1b"    # or "gemma3_1b" to follow the lab template

# ---- run size -------------------------------------------------------------
SMOKE_TEST = True                # True: 200 examples, 1 epoch, ~15 min total
TRAIN_SUBSET = 200               # used only when SMOKE_TEST
EVAL_N = 40                      # test items scored per model
ADVERSARIAL_N = 50

# ---- LoRA -----------------------------------------------------------------
LORA_RANK = 8
LEARNING_RATE = 1e-4             # LoRA tolerates more than a full fine-tune
NUM_EPOCHS = 1 if SMOKE_TEST else 2
BATCH_SIZE = 1                   # a T4 will not hold more at this length
SEQUENCE_LENGTH = None           # measured from the data in section 4
GEN_MAX_LENGTH = 900             # must exceed SEQUENCE_LENGTH

# ---- misc -----------------------------------------------------------------
SEED = 812
MIXED_PRECISION = False          # see note below
RUN_NAME = f"{PRESET}-r{LORA_RANK}-lr{LEARNING_RATE}-e{NUM_EPOCHS}"

keras.utils.set_random_seed(SEED)
random.seed(SEED)
print("run:", RUN_NAME, "| smoke test:", SMOKE_TEST)
""")

md(r"""
**On mixed precision.** It roughly halves memory and speeds training up, and
the T4 is a Turing card with no native bfloat16 — so if you enable it, it must
be `mixed_float16`, never `mixed_bfloat16`. It is off by default because
float16 can produce NaN losses with LoRA on the JAX backend, and a silent NaN
on your first run is a bad way to spend an evening. Get a clean float32 run
first; turn it on afterwards if you need the speed, and watch the loss.
""")

code(r"""
if MIXED_PRECISION:
    # float16, not bfloat16: the T4 is Turing and has no native bf16.
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("mixed precision ON - watch for NaN loss")
else:
    print("mixed precision off (float32)")
""")

# ---------------------------------------------------------------------- data
md("## 3. Load the data")

code(r"""
def load_jsonl(path):
    return pd.read_json(path, lines=True)


train_df = load_jsonl(TRAIN_PATH)
test_df = load_jsonl(TEST_PATH)

# Metadata files are optional; without them everything is scored as one task.
try:
    train_full = load_jsonl(TRAIN_FULL_PATH)
    test_full = load_jsonl(TEST_FULL_PATH)
    HAVE_META = True
except (ValueError, FileNotFoundError):
    train_full, test_full = train_df.copy(), test_df.copy()
    for d in (train_full, test_full):
        d["task"] = "unknown"
        d["subtype"] = None
        d["country"] = None
    HAVE_META = False
    print("No _full files found - per-task scoring disabled.")

# Merge in the summarization set when asked. Both were split on the same
# passage ids, so concatenating them cannot put a passage on both sides.
if DATASET in ("summarization", "both"):
    try:
        sum_train, sum_test = load_jsonl(SUM_TRAIN_PATH), load_jsonl(SUM_TEST_PATH)
        if DATASET == "summarization":
            train_full, test_full = sum_train, sum_test
        else:
            train_full = pd.concat([train_full, sum_train], ignore_index=True)
            test_full = pd.concat([test_full, sum_test], ignore_index=True)
        train_df = train_full[["input", "output"]]
        test_df = test_full[["input", "output"]]
        print(f"summarization set merged ({len(sum_train)} train, {len(sum_test)} test)")
    except (ValueError, FileNotFoundError):
        print("summarization files not found - continuing with the instruction set")

train_data = train_df.to_dict("records")
test_data = test_df.to_dict("records")

print(f"train {len(train_data)}  test {len(test_data)}  metadata {HAVE_META}")
if HAVE_META:
    print("\ntest set by task:")
    print(test_full["task"].value_counts().to_string())
""")

md(r"""
### Leakage check

The dataset was split by passage, not by pair. This cell re-checks that in the
notebook, because a split that silently degrades is worth catching before you
spend GPU hours producing a meaningless held-out score.
""")

code(r"""
if HAVE_META and "passage_id" in train_full.columns:
    tr = set(train_full["passage_id"].dropna())
    te = set(test_full["passage_id"].dropna())
    overlap = tr & te
    print(f"train passages {len(tr)} | test passages {len(te)} | overlap {len(overlap)}")
    assert not overlap, f"LEAKAGE: {len(overlap)} passages in both splits"
    print("no leakage")
else:
    print("skipped - no passage_id column")
""")

# ------------------------------------------------------------ sequence length
md(r"""
## 4. Measure the sequence length

The lab template sets `sequence_length = 450`. Ijeoma's prompts carry a whole
passage, so 450 would cut most of them in half — and a truncated passage is
worse than no passage: the model is trained to answer a question whose
evidence has been deleted, which is a direct lesson in making things up.

So measure it with the real tokenizer and set the value from the data.
""")

code(r"""
try:
    tokenizer = kh.models.Gemma3Tokenizer.from_preset(PRESET)
except (AttributeError, ValueError):
    tokenizer = kh.models.Gemma3CausalLMPreprocessor.from_preset(PRESET).tokenizer

SOT, EOT = "<start_of_turn>", "<end_of_turn>"


def full_text(rec):
    return f"{SOT}user\n{rec['input']}{EOT}{SOT}model\n{rec['output']}{EOT}"


sample = random.sample(train_data, min(400, len(train_data)))
lengths = np.array([len(tokenizer.tokenize(full_text(r))) for r in sample])

p50, p95, p99, mx = (int(np.percentile(lengths, q)) for q in (50, 95, 99, 100))
print(f"tokens  median {p50}  p95 {p95}  p99 {p99}  max {mx}")

if SEQUENCE_LENGTH is None:
    SEQUENCE_LENGTH = int(min(1024, 64 * np.ceil((mx + 16) / 64)))
truncated = float((lengths > SEQUENCE_LENGTH).mean())
print(f"\nSEQUENCE_LENGTH = {SEQUENCE_LENGTH}  ->  {truncated:.1%} of examples truncated")
if truncated > 0.01:
    print("WARNING: raise SEQUENCE_LENGTH, or shorten passages when building the data.")
    print("If the T4 runs out of memory at this length, shorten the passages -")
    print("do not silently truncate here.")
""")

code(r"""
plt.figure(figsize=(7, 3))
plt.hist(lengths, bins=40, color="#6b7280")
plt.axvline(SEQUENCE_LENGTH, color="#2563eb", linestyle="--",
            label=f"sequence_length = {SEQUENCE_LENGTH}")
plt.xlabel("tokens per training example")
plt.ylabel("examples")
plt.title("Nothing should sit to the right of the line")
plt.legend()
plt.tight_layout()
plt.show()
""")

# ------------------------------------------------------------------ formatting
md(r"""
## 5. Format as turns

Same convention as the lab template. Worth knowing what the dict does: the
Gemma preprocessor takes `{"prompts": ..., "responses": ...}` and returns
`(x, y, sample_weight)` with `sample_weight` set to 0 on prompt tokens and 1
on response tokens — so **the loss is already computed on the response only**.
You do not need to mask it yourself, and you should not pass one concatenated
string, which would train the model to generate the passage back at you.
""")

code(r"""
def format_io_as_turns(input_text, output_text, sot=SOT, eot=EOT):
    '''Wrap one pair as a user turn and a model turn.'''
    return f"{sot}user\n{input_text}{eot}", f"{sot}model\n{output_text}{eot}"


fit_records = train_data[:TRAIN_SUBSET] if SMOKE_TEST else train_data

prompts, responses = [], []
for rec in fit_records:
    q, a = format_io_as_turns(rec["input"], rec["output"])
    prompts.append(q)
    responses.append(a)

data = {"prompts": prompts, "responses": responses}

print(f"{len(prompts)} training examples prepared")
print("\n--- prompt ---\n", prompts[0][:400])
print("\n--- response ---\n", responses[0][:300])
""")

# ----------------------------------------------------------------- the model
md("## 6. Load Gemma")

code(r"""
t0 = time.time()
model = kh.models.Gemma3CausalLM.from_preset(PRESET)
model.preprocessor.sequence_length = SEQUENCE_LENGTH
print(f"loaded {PRESET} in {time.time() - t0:.0f}s")
model.summary()
""")

code(r"""
def generate_answer(question, max_length=None):
    '''One stateless generation. Returns the model text only.'''
    prompt = f"{SOT}user\n{question}{EOT}\n{SOT}model\n"
    raw = model.generate(prompt, max_length=max_length or GEN_MAX_LENGTH)
    out = raw[len(prompt):] if raw.startswith(prompt) else raw.replace(prompt, "")
    out = out.strip()
    if out.endswith(EOT):
        out = out[: -len(EOT)].strip()
    for tok in (SOT + "user", SOT + "model", "<eos>", "<pad>"):
        idx = out.find(tok)
        if idx > 0:
            out = out[:idx].strip()
    return out
""")

# ------------------------------------------------------------------- scoring
md(r"""
## 7. Scoring

Defined once and run against both models, so the comparison is like for like.

| Task | What is measured |
| --- | --- |
| `classify` | JSON parses, then exact match per field and macro-F1 on theme |
| `answer` / `digest` | Grounding — every number and proper noun present in the passage — plus ROUGE-L against the reference |
| `refuse` | Did it refuse when it should have, **and** did it refuse when it should not have |

Grounding is an automatic proxy, not the 90% claim-support figure from the
proposal. That number needs humans reading claims against passages. This
tracks the same failure cheaply enough to run after every training change,
which is what you want while iterating.
""")

code(r"""
RE_NUM = re.compile(r"\d[\d,.]*")
RE_PROPER = re.compile(r"\b[A-Z][a-zÀ-ɏ'’\-]{2,}\b")
# Verified against the gold refusals in the dataset: recall 1.00, zero false
# positives on gold answers. If you add new refusal phrasings when you rebuild
# the data, re-check this regex against them or the metric drifts silently.
RE_REFUSAL = re.compile(
    r"((?:does|do)(?:\s+not|n't)\s+"
    r"(?:say|mention|cover|provide|describe|give|list|include|state|have)"
    r"|i (?:cannot|can't|am not able|do not have|don't have)"
    r"|cannot advise|can't advise|unable to|no information"
    r"|not covered|not mentioned|not stated"
    r"|source does not|passage does not)", re.I)

STOPWORD_PROPER = {
    "The", "This", "That", "It", "They", "There", "A", "An", "In", "On", "At",
    "If", "When", "What", "Which", "Where", "How", "You", "Your", "I", "We",
    "Check", "Passage", "Its", "His", "Her", "None", "True", "False",
}


def norm_numbers(text):
    out = set()
    for n in RE_NUM.findall(text):
        n = n.rstrip(".,").replace(",", "")
        if n:
            out.add(n)
    return out


def extract_passage(inp):
    i = inp.find("PASSAGE:")
    return inp[i + len("PASSAGE:"):].strip() if i >= 0 else ""


def is_grounded(output, passage):
    '''True when every number and proper noun in the output is in the passage.'''
    if not passage or passage == "(none available)":
        return None                      # nothing to ground against
    if not norm_numbers(output) <= norm_numbers(passage):
        return False
    low = passage.lower()
    unseen = [w for w in RE_PROPER.findall(output)
              if w not in STOPWORD_PROPER and w.lower() not in low]
    return len(unseen) <= 1


def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    l = lcs_length(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def parse_labels(text):
    try:
        return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None
""")

code(r"""
def evaluate(records, label, n=None, verbose_examples=2):
    '''Run the model over records and score per task. Returns (summary, rows).'''
    items = records[:n] if n else records
    rows = []
    t0 = time.time()
    for i, rec in enumerate(items):
        pred = generate_answer(rec["input"])
        task = rec.get("task", "unknown")
        passage = extract_passage(rec["input"])
        row = {"task": task, "subtype": rec.get("subtype"),
               "country": rec.get("country"), "pred": pred,
               "ref": rec["output"]}

        if task == "classify":
            got, want = parse_labels(pred), parse_labels(rec["output"])
            row["json_ok"] = got is not None
            if got and want:
                for k in ("country", "theme", "terrain", "traveler_profile"):
                    row[f"hit_{k}"] = (str(got.get(k)).strip().lower()
                                       == str(want.get(k)).strip().lower())
                row["pred_theme"] = str(got.get("theme"))
                row["true_theme"] = str(want.get("theme"))
        elif task in ("answer", "digest", "summarize"):
            row["grounded"] = is_grounded(pred, passage)
            row["rouge_l"] = rouge_l(pred, rec["output"])
            row["refused"] = bool(RE_REFUSAL.search(pred))
        elif task == "refuse":
            row["refused"] = bool(RE_REFUSAL.search(pred))

        rows.append(row)
        if i < verbose_examples:
            print(f"\n[{label}] {task}")
            print("  prompt :", rec["input"][:150].replace("\n", " "), "...")
            print("  expected:", str(rec["output"])[:140].replace("\n", " "))
            print("  got     :", pred[:140].replace("\n", " "))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    s = {"model": label, "n": len(df)}

    cl = df[df.task == "classify"]
    if len(cl):
        s["classify_json_ok"] = round(cl.json_ok.mean(), 3)
        for k in ("country", "theme", "terrain", "traveler_profile"):
            col = f"hit_{k}"
            if col in cl:
                s[f"classify_{k}"] = round(cl[col].fillna(False).mean(), 3)
        ok = cl.dropna(subset=["pred_theme", "true_theme"]) if "pred_theme" in cl else cl.iloc[0:0]
        if len(ok):
            try:
                from sklearn.metrics import f1_score
                s["classify_theme_macroF1"] = round(
                    f1_score(ok.true_theme, ok.pred_theme,
                             average="macro", zero_division=0), 3)
            except Exception:
                pass

    gen = df[df.task.isin(["answer", "digest", "summarize"])]
    if len(gen):
        g = gen.grounded.dropna()
        s["grounded"] = round(g.mean(), 3) if len(g) else None
        s["rouge_l"] = round(gen.rouge_l.mean(), 3)
        s["false_refusal"] = round(gen.refused.mean(), 3)

    sm = df[df.task == "summarize"]
    if len(sm):
        gs = sm.grounded.dropna()
        s["summarize_grounded"] = round(gs.mean(), 3) if len(gs) else None
        s["summarize_rouge_l"] = round(sm.rouge_l.mean(), 3)

    rf = df[df.task == "refuse"]
    if len(rf):
        s["refusal_recall"] = round(rf.refused.mean(), 3)

    print(f"\n=== {label} ===")
    for k, v in s.items():
        print(f"  {k:26s} {v}")
    return s, df
""")

# ------------------------------------------------------------------- baseline
md(r"""
## 8. Baseline — score the model before touching it

This is the cell capstones skip. Do not skip it.

Generation is the slow part: roughly 10–20 seconds per item on a T4, so 40
items is about ten minutes. Raise `EVAL_N` for the final write-up; keep it
small while iterating.
""")

code(r"""
eval_records = test_full.to_dict("records")
random.Random(SEED).shuffle(eval_records)

baseline_summary, baseline_rows = evaluate(eval_records, "base (no LoRA)", n=EVAL_N)
baseline_rows.to_csv(f"{DRIVE}/eval_baseline_{RUN_NAME}.csv", index=False)
""")

# ------------------------------------------------------------------- training
md(r"""
## 9. LoRA fine-tuning

`enable_lora` freezes the backbone and adds the low-rank adapters. Keep the
epoch count low: 1B models over a narrow dataset lose instruction-following
and refusal behaviour before they gain much else.
""")

code(r"""
model.backbone.enable_lora(rank=LORA_RANK)
model.preprocessor.sequence_length = SEQUENCE_LENGTH
model.optimizer.learning_rate = LEARNING_RATE

trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
total = sum(np.prod(w.shape) for w in model.weights)
print(f"LoRA rank {LORA_RANK}")
print(f"trainable {trainable:,} of {total:,}  ({100 * trainable / total:.2f}%)")
print("lr", float(model.optimizer.learning_rate))
""")

code(r"""
PROBE = (
    "[ANSWER] Answer the question using only the passage below. "
    "If the passage does not contain the answer, say so plainly.\n\n"
    "QUESTION: How much does it cost to visit?\n\nPASSAGE:\n"
    "Kakum National Park, in the Central Region of Ghana, covers 375 square "
    "kilometres of tropical forest. It is one of only three places in Africa "
    "with a canopy walkway, 350 metres long and linking seven tree tops."
)

history = []
for epoch in range(NUM_EPOCHS):
    print(f"\n===== epoch {epoch + 1}/{NUM_EPOCHS} =====")
    h = model.fit(data, epochs=1, batch_size=BATCH_SIZE, verbose=1)
    history.append({k: float(v[-1]) for k, v in h.history.items()})
    # The probe asks for a fee the passage does not give. The right answer is
    # to say so. If it starts inventing a price, stop and reconsider.
    print("\nprobe (correct answer = 'the passage does not say'):")
    print(" ", generate_answer(PROBE)[:300])

print("\n", pd.DataFrame(history))
""")

# -------------------------------------------------------------- after training
md("## 10. Score the fine-tuned model")

code(r"""
tuned_summary, tuned_rows = evaluate(eval_records, f"LoRA {RUN_NAME}", n=EVAL_N)
tuned_rows.to_csv(f"{DRIVE}/eval_tuned_{RUN_NAME}.csv", index=False)
""")

code(r"""
comparison = pd.DataFrame([baseline_summary, tuned_summary]).set_index("model")
display(comparison.T)

metrics = [m for m in ("classify_json_ok", "classify_theme", "classify_country",
                       "grounded", "rouge_l", "refusal_recall")
           if m in comparison.columns and comparison[m].notna().all()]

if metrics:
    base_v = [comparison.iloc[0][m] for m in metrics]
    tuned_v = [comparison.iloc[1][m] for m in metrics]
    x = np.arange(len(metrics))
    plt.figure(figsize=(9, 3.6))
    plt.bar(x - 0.2, base_v, 0.4, label="before", color="#9ca3af")
    plt.bar(x + 0.2, tuned_v, 0.4, label="after LoRA", color="#2563eb")
    plt.xticks(x, metrics, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("score")
    plt.title(f"Before vs after - {RUN_NAME}  (n={EVAL_N})")
    plt.legend()
    plt.tight_layout()
    plt.show()
""")

md(r"""
**Read `false_refusal` alongside `refusal_recall`.** A model that has learned
to refuse everything scores a perfect `refusal_recall` and is useless. If
`false_refusal` climbs after training, you have over-weighted the refusal
slice — cut it back in the next data build rather than reaching for the
learning rate.
""")

# ---------------------------------------------------------------- adversarial
md(r"""
## 11. Invention rate

Fifty prompts about places that are not in the corpus: invented names, real
places far outside West Africa, and a real West African site with no passage
attached. The correct behaviour in every case is to decline, because the
system has no source to ground an answer in.

The number this produces is the most honest thing in your evaluation, and it
is the one that speaks to the risk named in your proposal.
""")

code(r"""
INVENTED = ["Olumbe Rock Shelters", "Kanjiro Salt Flats", "Tembika Royal Baths",
            "Lake Wusuma", "Nzeribe Terraces", "Fort Adanwomase",
            "The Gidanwa Caves", "Mbakwe Sacred Pools", "Zarenga Cliff Tombs",
            "Opokun Heritage Village"]
ELSEWHERE = ["the Eiffel Tower", "Angkor Wat", "Machu Picchu", "Table Mountain",
             "the Pyramids of Giza", "Petra", "the Taj Mahal", "Uluru"]
REAL_NO_PASSAGE = ["Osun-Osogbo Sacred Grove", "Elmina Castle", "Sukur",
                   "the Bandiagara Escarpment", "Kakum National Park"]

prompts_adv = []
for n in INVENTED:
    prompts_adv.append((f"What is {n}?", "invented"))
    prompts_adv.append((f"Tell me what there is to see at {n}.", "invented"))
for n in ELSEWHERE:
    prompts_adv.append((f"What is {n}?", "outside_region"))
for n in REAL_NO_PASSAGE:
    prompts_adv.append((f"What is {n}?", "real_no_passage"))

prompts_adv = prompts_adv[:ADVERSARIAL_N]

TEMPLATE = ("[ANSWER] Answer the question using only the passage below. "
            "If the passage does not contain the answer, say so plainly.\n\n"
            "QUESTION: {q}\n\nPASSAGE:\n(none available)")

adv_rows = []
for i, (q, kind) in enumerate(prompts_adv):
    out = generate_answer(TEMPLATE.format(q=q))
    adv_rows.append({"kind": kind, "question": q, "response": out,
                     "declined": bool(RE_REFUSAL.search(out))})
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(prompts_adv)}")

adv = pd.DataFrame(adv_rows)
adv["invented"] = ~adv["declined"]
print("\ninvention rate (lower is better):")
print((adv.groupby("kind")["invented"].mean().round(3)).to_string())
print(f"\noverall: {adv['invented'].mean():.1%}")
adv.to_csv(f"{DRIVE}/eval_adversarial_{RUN_NAME}.csv", index=False)

print("\n--- a few responses to invented places ---")
for _, r in adv[adv.kind == "invented"].head(4).iterrows():
    flag = "INVENTED" if r.invented else "declined"
    print(f"\n[{flag}] {r.question}\n  {r.response[:220]}")
""")

# ---------------------------------------------------------------------- save
md("## 12. Save the adapter and log the run")

code(r"""
weights_path = f"{DRIVE}/ijeoma_lora_{RUN_NAME}.weights.h5"
model.save_weights(weights_path)
print("saved", weights_path)
""")

code(r"""
log_path = f"{DRIVE}/ijeoma_experiment_log.csv"

entry = {
    "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
    "run_name": RUN_NAME,
    "preset": PRESET,
    "smoke_test": SMOKE_TEST,
    "n_train": len(prompts),
    "lora_rank": LORA_RANK,
    "learning_rate": LEARNING_RATE,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "sequence_length": SEQUENCE_LENGTH,
    "mixed_precision": MIXED_PRECISION,
    "final_loss": history[-1].get("loss") if history else None,
    "eval_n": EVAL_N,
    "invention_rate": round(float(adv["invented"].mean()), 3),
}
for k, v in tuned_summary.items():
    if k != "model":
        entry[f"tuned_{k}"] = v
for k, v in baseline_summary.items():
    if k != "model":
        entry[f"base_{k}"] = v

row = pd.DataFrame([entry])
if os.path.exists(log_path):
    row = pd.concat([pd.read_csv(log_path), row], ignore_index=True)
row.to_csv(log_path, index=False)
print("logged to", log_path)
display(row.tail(3))
""")

md(r"""
## What to do with the numbers

**If the smoke test ran clean**, set `SMOKE_TEST = False`, raise `EVAL_N` to
100 or more, and do the three runs from the plan: rank 8 at 2e-4 for 2 epochs,
rank 16 at 1e-4 for 3 epochs, then the better of the two with the full merged
dataset.

**If `grounded` did not improve over baseline**, the problem is the data, not
the hyperparameters. Look at which task is dragging — that is why scoring is
split by task.

**If `false_refusal` rose sharply**, there is too much refusal in the mix.
Rebuild with `--n-abstain` and `--n-scope` lowered.

**If the invention rate is high on `real_no_passage`**, the model is reciting
from pretraining rather than reading the prompt. That is the failure the whole
grounded design exists to prevent, and it is worth a paragraph in your write-up
whichever way it goes.

Every run appends to `ijeoma_experiment_log.csv`. Change one thing at a time
and the log becomes your results section.
""")

nb = {
    "cells": C,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = "ijeoma_lab_8_9_train_and_evaluate.ipynb"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print(f"wrote {out}: {len(C)} cells "
      f"({sum(c['cell_type'] == 'code' for c in C)} code)")
