# Code walkthrough

What every piece of code we wrote or changed does, line by line, so it can be
explained to a reviewer. Read the **big picture** first; the sections after it can
be read in any order. A glossary of terms is at the end.

Code that already existed before 22 Nov (most of `pipeline/`, most of the
training notebook) is not walked through here except where we changed it.

---

## The big picture

The project is four stages. Each stage produces files the next one reads.

```
 ① BUILD DATA                ② TRAIN                    ③ EVALUATE                ④ SERVE
 pipeline/*.py               notebooks/…v0_3.ipynb      notebook + evaluation/    app/*.py
 passages ─► examples ─►     examples ─► LoRA adapter   adapter ─► scores         adapter ─► web pages
 (JSONL files)               (.npz file, 5 MB)          (CSV files, judge)        (Gradio)
```

1. **Build data.** 1,586 passages about places become ~7,800 training and test
   examples. Code templates make some; a teacher model (OpenAI) writes the rest.
   Every example is checked before it is kept.
2. **Train.** Gemma 3 1B learns from the examples. Only small add-on matrices
   (LoRA) are trained; the result is a 5 MB adapter file.
3. **Evaluate.** The model answers 100 test questions before and after training;
   code scores the answers, and a judge model checks every claim against the passage.
4. **Serve.** A small web app loads Gemma plus the adapter and lets people try it.

One idea runs through all four: **every prompt contains the source passage**,
and the model is trained to answer only from it.

---

## ① Build data

### 1.1 `pipeline/ijeoma_pairs.py` — the label list in classify prompts

**The problem it fixes:** the classify prompt never said which labels were
allowed, so the model invented its own and scored 0 on theme.

```python
THEMES = ["Town or Area Guide", "Nature & Wildlife", "Museum & Gallery", …]
```
A Python **list** of the 13 allowed theme names. These are the exact strings the
dataset uses as answers, so the model must copy one of them.

```python
TERRAINS = ["Coastal", "Riverine", "Island", …]
PROFILES = ["General, base for exploring", "Nature and wildlife, outdoor", …]
```
The same for the 9 terrains and 12 traveller profiles.

```python
LABEL_MENU = (
    "Allowed values (copy one exactly):\n"
    f"theme: {' | '.join(THEMES)}\n"
```
Builds one block of text. `' | '.join(THEMES)` glues the list into
`Town or Area Guide | Nature & Wildlife | …`. The `f"…{…}"` form is an
**f-string**: Python replaces `{…}` with the value inside. `\n` is a new line.

```python
    f"terrain: {' | '.join(TERRAINS)} | null\n"
```
Terrain may also be `null` (no terrain), because 645 passages have none.

```python
"input": (f"{TASK_TAGS['classify']} {instr}\n{LABEL_MENU}\n\n"
          f"PASSAGE:\n{row['passage']}"),
```
The classify prompt is now: tag `[CLASSIFY]` + instruction + **the label menu** +
blank line + `PASSAGE:` + the passage text. Before, the menu line was missing.

---

### 1.2 `pipeline/ijeoma_teacher.py` — OpenAI teacher and conversations

The teacher is a large model that reads a passage and writes training examples.
The file existed for Gemini; we added an OpenAI option and a conversation task.

#### Choosing the key

```python
def get_api_key(names=("GOOGLE_API_KEY", "GEMINI_API_KEY")) -> str:
```
Looks for an API key under a list of names. We made the list a parameter so the
same function can look for `OPENAI_API_KEY`. It checks Colab Secrets first, then
environment variables, and stops with a message if none is found.

#### The OpenAI client

```python
class OpenAITeacher:
```
A **class** groups data and functions together. This one talks to OpenAI and
has the same shape as the Gemini `Teacher`, so the rest of the file can use
either without knowing which.

```python
    URL = "https://api.openai.com/v1/chat/completions"
```
The web address OpenAI's chat API listens on.

```python
    def __init__(self, model: str):
        self.key = get_api_key(("OPENAI_API_KEY",))
        self.model = model
        self.lock = threading.Lock()
        self.calls = 0
```
`__init__` runs when the object is created. It stores the key and model name, a
**lock** (so two threads don't update `calls` at the same moment) and a counter
of API calls made.

```python
    def generate(self, prompt: str, retries: int = 4) -> str | None:
        import urllib.request
```
Sends one prompt, returns the text reply, or `None` if it keeps failing.
`urllib` is Python's built-in web library, so nothing extra needs installing.

```python
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }).encode()
```
Builds the request: which model, the prompt as a user message, and
`json_object`, which forces the reply to be valid JSON. All our teacher prompts
ask for JSON, so this prevents unparseable replies. `.encode()` turns the text
into bytes for sending.

```python
        delay = 2.0
        for attempt in range(retries):
            try:
                req = urllib.request.Request(self.URL, data=body, headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.load(r)
```
Tries up to 4 times. Each try sends the request with the key in the
`Authorization` header and waits up to 180 seconds for the answer.

```python
                with self.lock:
                    self.calls += 1
                return (data["choices"][0]["message"]["content"] or "").strip()
```
On success: count the call (inside the lock) and return the reply text, which
OpenAI puts at `choices[0].message.content`.

```python
            except Exception as exc:
                if attempt == retries - 1:
                    print(f"  giving up after {retries}: {exc}", file=sys.stderr)
                    return None
                time.sleep(delay)
                delay *= 2
```
On failure (rate limit, network error): wait 2 s, then 4 s, then 8 s and retry.
This is **exponential backoff**. After the last try, give up and return `None`.

#### The conversation prompt

```python
DIALOGUE_PROMPT = """…Write a realistic conversation of exactly 3 exchanges
between a curious Traveller and a knowledgeable Guide…
In the third exchange the Traveller asks something practical the passage does NOT
cover (cost, opening hours, transport, safety); the Guide says plainly the source
does not cover it…
{rules}
Return: {{"turns": [{{"traveller": "...", "guide": "..."}}, …]}}
DESTINATION: {name} ({country})
PASSAGE:
{passage}"""
```
The instruction sent to the teacher for each passage. `{rules}`, `{name}`,
`{country}` and `{passage}` are filled in later. Double braces `{{ }}` mean a
literal `{`, needed because the JSON example contains braces. `{rules}` adds the
shared rules: use only the passage, no phone numbers or prices, no marketing
words, skip if the passage is too thin.

```python
DIALOGUE_INSTRUCTION = (
    "Continue the conversation as the guide. Answer the traveller's latest "
    "question using only the passage below. …")
```
This is **not** sent to the teacher. It is the instruction placed in the
*training example* that Gemma will learn from.

#### Turning one conversation into training examples

```python
def dialogue_pairs(row, raw) -> list[dict]:
    obj = parse_json(raw)
    if not obj or obj.get("skip"):
        return []
```
`raw` is the teacher's reply. Parse it as JSON; if it failed or the teacher said
`skip`, produce nothing.

```python
    turns = [t for t in (obj.get("turns") or [])
             if str(t.get("traveller", "")).strip() and str(t.get("guide", "")).strip()]
```
Keep only turns where both the question and the answer are non-empty.

```python
    out, history = [], []
    for i, t in enumerate(turns):
        q, a = str(t["traveller"]).strip(), str(t["guide"]).strip()
        answerable = bool(t.get("answerable", True))
        convo = "\n".join(history) if history else "(start of conversation)"
```
Walk through the turns in order. `history` collects what has been said so far.
For the first turn it is empty, so the prompt says `(start of conversation)`.

```python
        out.append({
            "input": (f"{TASK_TAGS['answer']} {DIALOGUE_INSTRUCTION}\n\n"
                      f"CONVERSATION SO FAR:\n{convo}\n\n"
                      f"QUESTION: {q}\n\nPASSAGE:\n{row['passage']}"),
            "output": a,
```
One training example per guide turn. The **input** is: instruction + the
conversation so far + the new question + the passage. The **output** is the
guide's answer. This is how the model learns to use earlier turns.

```python
            "task": "answer" if answerable else "refuse",
            "subtype": f"dialogue_turn{i + 1}" if answerable else "dialogue_abstain",
```
Labels used later for scoring: an answerable turn is an `answer`; the third,
unanswerable turn is a `refuse`.

```python
        history += [f"Traveller: {q}", f"Guide: {a}"]
    return out
```
Add this exchange to the history so the next turn sees it. One conversation of 3
turns becomes 3 training examples.

#### Running the jobs

```python
    dialogue_rows = country_quota_sample(rich, min(n_dialogue, len(rich)), 0.22, rng)
```
Choose which passages get a conversation: only "rich" ones (≥ 900 characters),
with no country above 22% of the sample.

```python
    jobs = [("digest", r) for _, r in digest_rows.iterrows()] + \
           [("qa", r) for _, r in qa_rows.iterrows()] + \
           [("dialogue", r) for _, r in dialogue_rows.iterrows()]
    jobs = [(t, r) for t, r in jobs if not cache.has(r["passage_id"], t)]
```
Make a to-do list of (job type, passage), then drop any already in the cache.
The cache is why rerunning costs nothing.

```python
    teacher = (OpenAITeacher if args.provider == "openai" else Teacher)(args.model)
```
Pick the OpenAI or Gemini class depending on `--provider`, and create it.

```python
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
```
Run the jobs in parallel: 16 at once with `--workers 16`. Each `work` call sends
one prompt and saves the reply to the cache.

---

### 1.3 `pipeline/ijeoma_build.py` — the extra quality check

`validate(pair)` returns a list of reasons an example is bad; an empty list means
keep it. We added rule 6.

```python
    if str(pair.get("subtype", "")).startswith(("teacher", "dialogue")):
```
Only check teacher-written examples. Template examples are correct by construction.

```python
        refuses = bool(RE_REFUSAL.search(out))
```
`RE_REFUSAL` is a **regular expression**, a text pattern, that matches refusal
phrases such as "does not cover", "I cannot", "passage does not". `search` finds
it anywhere in the answer. `refuses` is True if the answer contains a refusal.

```python
        if task == "refuse" and not refuses:
            fails.append("refusal_label_mismatch")
        if task in GROUNDED_TASKS and refuses:
            fails.append("refusal_label_mismatch")
```
Reject an example labelled *refuse* that doesn't refuse, or labelled *answer*
that does. Either would teach the model the wrong behaviour.

```python
        words = re.findall(r"[a-z]+", out.lower())
        if words and sum(w in ENGLISH_MARKERS for w in words) / len(words) < 0.08:
            fails.append("not_english")
```
A cheap language check. Split the answer into words and count how many are
common English words (`the`, `and`, `is`…). English text is usually 25–40% such
words; below 8% it is almost certainly another language. This caught the
Italian answers.

---

## ② Train — notebook changes (`ijeoma_train_and_evaluate_v0_3.ipynb`)

### 2.1 Finding the Drive folder (section 2)

```python
_hits = [d for d in ["/content/drive/MyDrive/Ijeoma"]
         + glob.glob("/content/drive/MyDrive/*/Ijeoma")
         + glob.glob("/content/drive/MyDrive/*/*/Ijeoma")
         if os.path.exists(f"{d}/conversational_training_data.jsonl")]
```
Look for a folder called `Ijeoma` at the top of My Drive, one level down, or two
levels down (`*` matches any folder name). Keep only those that actually contain
the training file, so a different folder that happens to be called Ijeoma is ignored.

```python
if not _hits:
    raise FileNotFoundError("Could not find the 'Ijeoma' folder …")
DRIVE = _hits[0]
```
Stop with a clear message if none is found; otherwise use the first match.

### 2.2 Settings (section 2)

```python
SMOKE_TEST = False
TRAIN_SUBSET = 200 if SMOKE_TEST else None
EVAL_N = 20 if SMOKE_TEST else 100
CHECKPOINT_EVERY = 500
```
`SMOKE_TEST` switches between a quick check and the real run. In the real run,
`TRAIN_SUBSET = None` means "use all examples", 100 test items are scored, and
the adapter is saved every 500 steps.

### 2.3 Saving during training (section 9)

```python
class SaveLoRA(keras.callbacks.Callback):
```
A Keras **callback** is code that Keras calls automatically during training.

```python
    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.every and self.step % self.every == 0:
```
Called after every training step. `%` is remainder, so `step % 500 == 0` is true
at steps 500, 1000, 1500…

```python
            ws = [w for w in self.model.weights
                  if "lora" in (getattr(w, "path", None) or w.name).lower()]
```
Pick out only the LoRA weights (their names contain "lora"). These are the
1.3 million numbers training changes; the other billion stay as they were.

```python
            np.savez(path, names=np.array([...]), step=self.step,
                     **{f"w{j}": np.asarray(w) for j, w in enumerate(ws)})
```
Write them to one `.npz` file on Drive: the weight names, the step number, and
each weight as `w0`, `w1`, …. That file is the checkpoint.

### 2.4 Keeping the prompt in the results (section 7)

```python
row = {"task": task, "subtype": …, "country": …, "pred": pred, "ref": rec["output"],
       "input": rec["input"]}
```
Each scored item now also stores its `input` (the full prompt). That lets the
judge run later, outside Colab, without an OpenAI key in Colab.

### 2.5 Extra metrics (section 10b)

```python
def rouge_1(pred, ref):
    p, r = _C(pred.lower().split()), _C(ref.lower().split())
    overlap = sum((p & r).values())
```
**ROUGE-1** measures word overlap between the model's answer and the reference.
`Counter` counts each word; `p & r` keeps the words in both, with the smaller
count. `overlap` is how many words they share.

```python
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)
```
Precision: share of the answer's words that are in the reference. Recall: share
of the reference's words that are in the answer. F1 combines the two.

```python
rep = classification_report(cl.true_theme, cl.pred_theme, output_dict=True, …)
summary["classify_theme_macroP"] = round(rep["macro avg"]["precision"], 3)
```
scikit-learn computes precision and recall per theme. The **macro average** treats
every theme equally, however rare.

```python
from bert_score import score as bscore
_, _, f1 = bscore(list(gen.pred), list(gen.ref), lang="en",
                  model_type="distilbert-base-uncased", device="cpu")
```
**BERTScore** compares meaning rather than exact words, using another small
model (DistilBERT). It runs on CPU because Gemma is already using the GPU.

The judge code in this section is the same as `evaluation/run_judge.py` below.

---

## ③ Evaluate — `evaluation/run_judge.py`

Scores the model's answers using another model as a judge, and computes the
proposal's success measure: **share of claims supported by the passage**.

```python
MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
GROUNDED = ("answer", "digest", "summarize")
```
Which judge model to use (overridable), and which tasks to judge: the ones that
produce free text from a passage.

```python
PROMPT = """You are grading a travel assistant's answer…
Score it against the SOURCE PASSAGE only, from 1 (poor) to 5 (excellent), on:
- factual_accuracy …  - cultural_nuance …  - helpfulness …
Also list every atomic factual claim in the answer … and mark whether the
passage directly supports it…"""
```
The judge's instructions. **Atomic claim** means one fact each: "built in 1652"
and "built by the Portuguese" are two claims. We ask for a list rather than a
count because the judge counted unreliably when asked for numbers directly.

```python
def judge(task, passage, answer):
    body = json.dumps({"model": MODEL, "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": PROMPT.format(
                           task=task, passage=passage[:6000], answer=answer)}]}).encode()
```
Fills the prompt with the question, the passage (cut at 6,000 characters) and
the model's answer, and requests JSON. The retry loop below it works like the teacher's.

```python
def main(path):
    df = pd.read_csv(path)
    items = df[df.task.isin(GROUNDED) & ~df.ref.astype(str).str.contains(RE_REFUSAL)]
```
Read the results CSV. Keep rows whose task is in `GROUNDED` **and** (`&`) whose
correct answer is **not** (`~`) a refusal. Judging a refusal for "claims" makes no sense.

```python
    def work(row):
        inp = row["input"]
        cut = inp.find("PASSAGE:")
        s = judge(inp[:cut].strip(), inp[cut + 8:].strip(), str(row["pred"]))
```
Split the stored prompt into the part before `PASSAGE:` (the question) and the
part after it (the passage). `+ 8` skips the 8 characters of "PASSAGE:".

```python
        claims = s.get("claims") or []
        return {…, "claims_total": len(claims),
                "claims_supported": sum(bool(c.get("supported")) for c in claims),
                "unsupported": json.dumps([c.get("claim") for c in claims
                                           if not c.get("supported")])}
```
Count the claims, count the supported ones (`True` counts as 1), and keep the
text of unsupported claims so a person can check them.

```python
    with ThreadPoolExecutor(8) as ex:
        out = [r for r in ex.map(work, [r for _, r in items.iterrows()]) if r]
```
Judge 8 answers at a time; drop any the judge failed on.

```python
    tot, sup = res.claims_total.sum(), res.claims_supported.sum()
    … "claims_supported_rate": round(sup / tot, 3) …
```
**The headline number**: all supported claims ÷ all claims, across every judged
answer. 261 ÷ 268 = 0.974.

---

## ④ Serve — the test pages (`app/`)

Three files with one job each:

| File | Job |
|---|---|
| `prompts.py` | builds the exact prompt text for each task |
| `model.py` | loads Gemma + adapter, and turns a prompt into an answer |
| `ijeoma_app.py` | the three web pages and the start-up options |

### 4.1 `app/prompts.py`

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
```
Tells Python to also look in the `pipeline/` folder when importing.
`__file__` is this file's path; `dirname` is its folder; `".."` goes up one level.

```python
from ijeoma_pairs import ANSWER_INSTRUCTION, CLASSIFY_INSTRUCTIONS, LABEL_MENU, TASK_TAGS
from ijeoma_summarize import INSTR_NOTABLE, INSTR_ONE_LINE, INSTR_SUMMARY
from ijeoma_teacher import DIALOGUE_INSTRUCTION, DIGEST_INSTRUCTIONS
```
**The key design choice:** the instruction wording is imported from the code
that built the training data, not retyped. A fine-tuned 1B model is sensitive to
wording, so the app must send exactly what the model trained on. We verified all
six prompt types match training examples character for character.

```python
NO_PASSAGE = "(none available)"

def _passage(p):
    p = (p or "").strip()
    return p if p else NO_PASSAGE
```
If the passage box is empty, send `(none available)`, the same marker used in
training for questions with no source.

```python
def classify(passage):
    return (f"{TASK_TAGS['classify']} {CLASSIFY_INSTRUCTIONS[0]}\n{LABEL_MENU}\n\n"
            f"PASSAGE:\n{_passage(passage)}")
```
Same shape as training: tag, instruction, label menu, passage. The other builders
(`answer`, `digest`, `summarize`) follow the same pattern.

```python
def dialogue(question, passage, history):
    lines = []
    for q, a in history:
        lines += [f"Traveller: {q}", f"Guide: {a}"]
    convo = "\n".join(lines) if lines else "(start of conversation)"
```
Turns the chat history into the `Traveller: … / Guide: …` lines the model saw in
training. Same logic as `dialogue_pairs` in the teacher.

```python
TASKS = {
    "Classify": lambda p, q, s: classify(p),
    "Ask a question": lambda p, q, s: answer(q, p),
    …
}
```
A **dictionary** from the button label to a function. `lambda p, q, s:` is a
one-line function taking passage, question and style. Every entry takes the same
three inputs, so the page can call `TASKS[task](passage, question, style)` whatever the task.

### 4.2 `app/model.py`

```python
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
```
Must run before Keras is imported. Keras can run on JAX, TensorFlow or PyTorch;
we use **JAX**, as in training. The second line lets JAX use up to 90% of GPU
memory. `setdefault` leaves the value alone if it is already set.

```python
SOT, EOT = "<start_of_turn>", "<end_of_turn>"
BUCKETS = (768, 1024, 1536, 2048)
```
Gemma's special markers for the start and end of a chat turn. `BUCKETS` are the
allowed total lengths, explained under `generate` below.

```python
def chat_template(prompt):
    return f"{SOT}user\n{prompt}{EOT}\n{SOT}model\n"
```
Wraps the prompt the way Gemma expects a chat: a *user* turn containing the
prompt, then an open *model* turn for Gemma to complete. Identical to training.

```python
def _strip(raw, prompt):
    out = raw[len(prompt):] if raw.startswith(prompt) else raw.replace(prompt, "")
    …
    for tok in (EOT, "<eos>", "<pad>"):
        i = out.find(tok)
        if i >= 0:
            out = out[:i].strip()
```
Keras returns the prompt **plus** the answer. This cuts off the prompt, then cuts
everything after the first end marker, leaving only the answer.

```python
class Ijeoma:
    def __init__(self, preset, adapter_path, lora_rank=8):
        …
        self.lock = threading.Lock()
        self.models = {}
        self.models["ijeoma"] = self._load(with_adapter=True)
```
Holds the loaded models in a dictionary. The fine-tuned model loads at start-up;
the base model loads only when the Compare page first needs it. The lock stops
two requests using the GPU at the same moment.

```python
    def _load(self, with_adapter):
        m = kh.models.Gemma3CausalLM.from_preset(self.preset)
```
Downloads (first time only) and loads Gemma 3 1B. `preset` is
`hf://google/gemma-3-1b-it`, its address on Hugging Face.

```python
        if with_adapter:
            m.backbone.enable_lora(rank=self.rank)
```
Adds empty LoRA layers to the model: small matrices alongside the attention
layers. They start at zero, so the model still behaves like the base.

```python
            z = np.load(self.adapter_path)
            saved = {str(n): z[f"w{i}"] for i, n in enumerate(z["names"])}
```
Opens the 5 MB `.npz` adapter and builds a name → numbers dictionary, the
reverse of what `SaveLoRA` wrote.

```python
            for w in m.weights:
                key = getattr(w, "path", None) or w.name
                if key in saved:
                    w.assign(saved[key])
                    hit += 1
            if hit != len(saved):
                raise RuntimeError(…)
```
Go through every weight in the model; where the name matches a saved LoRA
weight, copy the trained numbers in. Then check that **all 104** were loaded. A
mismatch means the wrong rank or base model, and silently running a half-loaded
model would be worse than stopping.

```python
        m.compile(sampler="greedy")
```
**Greedy** decoding always picks the most likely next word, so the same question
gives the same answer every time. This is the setting used in evaluation.

```python
        stops = [int(tok.token_to_id(t)) for t in (EOT, "<eos>") …]
        m._ijeoma_stops = stops or "auto"
```
Looks up the number IDs of the end markers so generation stops as soon as Gemma
finishes its turn, instead of running on to the length limit.

```python
    def generate(self, prompt, which="ijeoma", max_new_tokens=320):
        self.ensure(which)
        m = self.models[which]
        text = chat_template(prompt)
        n = len(m.preprocessor.tokenizer.tokenize(text))
```
Load the requested model if needed, wrap the prompt, and count its **tokens**
(word pieces; roughly ¾ of a word each).

```python
        need = n + max_new_tokens
        max_length = next((b for b in BUCKETS if b >= need), BUCKETS[-1])
```
Room needed = prompt + up to 320 new tokens, rounded up to the next bucket.
Keras compiles generation separately for each distinct length, which takes
time. Using four fixed sizes means at most four compiles, not one per question.

```python
        with self.lock:
            t0 = time.time()
            raw = m.generate(text, max_length=max_length, stop_token_ids=m._ijeoma_stops)
            dt = time.time() - t0
        return _strip(raw, text), dt
```
Generate, one request at a time, timing it. Return the cleaned answer and the
seconds taken.

```python
class FakeIjeoma:
```
A stand-in with the same functions but no model, returning canned text. Used
with `--fake` to test the pages on a machine without a GPU.

### 4.3 `app/ijeoma_app.py`

```python
SAMPLES = json.load(open(os.path.join(HERE, "sample_passages.json"), encoding="utf-8"))
CHOICES = [f"{s['name']} ({s['country']})" for s in SAMPLES]
BY_LABEL = dict(zip(CHOICES, SAMPLES))
```
Loads the 40 sample destinations (all from the test set, never trained on),
makes dropdown labels like "Great Mosque of Porto-Novo (Benin)", and maps each
label back to its passage.

```python
def build(engine):
```
Builds all three pages. `engine` is either the real `Ijeoma` or `FakeIjeoma`;
the pages don't need to know which.

```python
    def run(task, passage, question, style, which="ijeoma"):
        if task == "Ask a question" and not (question or "").strip():
            return "Type a question first.", "", ""
        prompt = P.TASKS[task](passage, question or "", style)
        out, dt = engine.generate(prompt, which=which)
        if task == "Classify":
            out = f"```json\n{out}\n```"
        return out, prompt, f"{dt:.1f}s"
```
What the Run button does: refuse an empty question, build the prompt, generate,
format classify output as JSON, and return the answer, the prompt (shown under
"Exact prompt sent") and the time.

```python
    def passage_inputs():
        pick = gr.Dropdown(CHOICES, value=CHOICES[0], label="Destination (held-out sample)")
        passage = gr.Textbox(passage_for(CHOICES[0]), lines=9, …)
        pick.change(passage_for, pick, passage)
        return pick, passage
```
Creates the destination dropdown and passage box. `pick.change(...)` is an
**event**: when the dropdown changes, call `passage_for` with its value and put
the result in the passage box. Reused on all three pages.

```python
    def task_inputs():
        task = gr.Radio(list(P.TASKS), value="Ask a question", label="Task")
        …
        def toggle(t):
            return gr.update(visible=t == "Ask a question"), gr.update(visible=t == "Summarise")
        task.change(toggle, task, [question, style])
```
The task buttons. Choosing a task shows the question box only for "Ask a
question" and the style dropdown only for "Summarise".

```python
    with gr.Blocks(title="Ijeoma AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(...)
        with gr.Row():
            with gr.Column(): …inputs…
            with gr.Column(): …outputs…
        go.click(run, [task, passage, question, style], [out, sent, took])
```
**Page 1, Workbench.** `gr.Blocks` is the app; `Row` and `Column` lay out two
side-by-side columns. `go.click(run, inputs, outputs)` wires the button: read
the inputs, call `run`, put its three results into the three outputs.

```python
    with demo.route("Chat", "/chat"):
```
**Page 2, Chat**, served at `/chat`. `demo.route` creates a separate page in the
same app; Gradio adds the navigation menu at the top.

```python
        def respond(message, history, passage, hide):
            pairs, q = [], None
            for turn in history:
                if turn["role"] == "user":
                    q = turn["content"]
                elif q is not None:
                    pairs.append((q, turn["content"]))
                    q = None
```
The chat box stores history as a list of messages, each `user` or `assistant`.
This loop pairs each user message with the reply after it, giving
`[(question, answer), …]`.

```python
            prompt = P.dialogue(message, "" if hide else passage, pairs)
            reply, _ = engine.generate(prompt)
            history = history + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": reply}]
            return "", history
```
Build the conversation prompt; if "Hide the passage" is ticked, send no passage.
Generate, add both messages to the history, and return an empty string to clear
the typing box.

```python
        msg.submit(respond, [msg, chat, passage_c, no_src], [msg, chat])
        clear.click(lambda: [], None, chat)
        pick_c.change(lambda: [], None, chat)
```
Pressing Enter sends a message. "New conversation" empties the chat, and so does
choosing a different destination, so one conversation never mixes two places.

```python
    with demo.route("Compare", "/compare"):
        …
        def compare(task, passage, question, style):
            b, _, tb = run(task, passage, question, style, which="base")
            i, _, ti = run(task, passage, question, style, which="ijeoma")
            return b, tb, i, ti
```
**Page 3, Compare.** Runs the same `run` twice, once with the base model and once
with Ijeoma AI, and shows the results side by side. The `_` discards the prompt,
which is the same for both.

```python
def main():
    ap = argparse.ArgumentParser(…)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--host", default="0.0.0.0", …)
    ap.add_argument("--share", action="store_true", …)
    ap.add_argument("--fake", action="store_true", …)
    a = ap.parse_args()
```
Reads command-line options. `0.0.0.0` means accept connections from other
computers on the network, not just this one. `store_true` options are on/off switches.

```python
    if a.fake:
        from model import FakeIjeoma as Engine
    else:
        if not os.path.exists(a.adapter):
            sys.exit(f"adapter not found: {a.adapter}")
        from model import Ijeoma as Engine
    engine = Engine(a.preset, a.adapter)
```
Choose the fake or real engine; stop early if the adapter file is missing.
Creating the real engine loads Gemma, which is the slow start-up step.

```python
    build(engine).queue(default_concurrency_limit=1).launch(
        server_name=a.host, server_port=a.port, share=a.share)
```
Build the pages and start the web server. `queue(… 1)` handles one request at a
time: one GPU can't run two generations at once, so others wait in line.
`share=True` asks Gradio for a public `gradio.live` link.

---

## Glossary

| Term | Meaning |
|---|---|
| **Passage** | The source text about one place, from Wikipedia, Wikivoyage or OpenStreetMap |
| **Example / pair** | One training item: an input (prompt) and the output we want |
| **Prompt** | The text given to the model: instruction + question + passage |
| **Teacher model** | A large model (here OpenAI) used to write training examples |
| **Template** | A fixed pattern filled in by code to make examples |
| **JSONL** | A file with one JSON record per line; our data format |
| **Gemma 3 1B** | Google's open model with 1 billion parameters; our base model |
| **Parameters / weights** | The numbers inside a model that training adjusts |
| **LoRA** | Training small add-on matrices instead of the whole model; we train 1.3M of 1B numbers (0.13%) |
| **Rank (8)** | The size of the LoRA matrices; bigger means more capacity and more memory |
| **Adapter** | The saved LoRA weights (`.npz`, 5 MB); added to base Gemma to get Ijeoma AI |
| **Epoch** | One pass through all training examples; we trained one |
| **Loss** | How wrong the model's predictions are during training; lower is better |
| **Checkpoint** | A saved copy of the weights partway through training |
| **Token** | A word piece the model reads and writes; about ¾ of a word |
| **Greedy decoding** | Always pick the most likely next token; same input → same output |
| **Keras / JAX** | The libraries the model runs on (Keras on top of JAX) |
| **Colab / T4** | Google's free notebook service and the GPU it provides |
| **Callback** | Code Keras runs automatically at points during training |
| **Regex** | A text pattern used to find phrases, such as refusals |
| **Grounded** | Every name and number in the answer appears in the passage |
| **Refusal recall** | Share of should-decline questions the model actually declined |
| **False refusal** | Share of answerable questions the model wrongly declined |
| **ROUGE-1 / ROUGE-L** | Word overlap with the reference answer (single words / longest matching sequence) |
| **BERTScore** | Meaning similarity with the reference, measured by another model |
| **Precision / recall / F1** | Of what was predicted, how much was right / of what was right, how much was found / their balance |
| **Macro average** | Average over classes, each class counted equally |
| **LLM judge** | Another model grading answers against a rubric |
| **Claims supported rate** | Supported claims ÷ all claims; the proposal's success measure (target 90%) |
| **Invention rate** | How often the model makes up an answer when it has no passage |
| **Held-out / test set** | Examples from passages never used in training |
| **Gradio** | A Python library for building simple web pages around a model |
| **Route** | A separate page in the Gradio app, at its own URL (`/chat`, `/compare`) |
