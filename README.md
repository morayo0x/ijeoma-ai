# Ijeoma AI

**A discovery assistant for West African destinations.** Ijeoma AI is Gemma 3 1B
fine-tuned with LoRA to do three things with a passage about a place:

- **classify** it for a travel catalogue (country, theme, terrain, traveller profile, World Heritage status)
- **summarise** it as a short traveller digest
- **answer** a traveller's questions about it, including follow-ups in a conversation

It answers only from the passage it is given. When the passage does not cover
a question, or the question needs live information (visas, prices, safety,
transport), it says so and points to official sources instead of guessing.

Capstone project, *AI Research Foundations for Higher Education* (Google DeepMind ×
FATE Foundation), Nigeria Cohort 1. Team Ijeoma AI, University of Lagos.

---

## Results

Held-out test set of 100 items from passages never seen in training.
Base = Gemma 3 1B-it before fine-tuning.

| | Base | **Ijeoma AI** |
|---|---|---|
| **Claims supported by the source passage** (proposal target ≥ 90%) | 86.0% | **97.4%** |
| Declines out-of-scope questions (refusal recall) | 44% | **92%** |
| Wrongly refuses answerable questions | 3% | 3% |
| Answers use only names/numbers found in the passage | 82% | **100%** |
| Theme classification accuracy (8 items) | 12.5% | **75%** |
| Judge score: factual accuracy (1–5) | 3.81 | **4.60** |
| Judge score: cultural nuance (1–5) | 4.30 | **4.61** |
| Judge score: helpfulness (1–5) | 3.81 | 3.76 |
| BERTScore F1 | 0.84 | **0.92** |
| Invents an answer when given **no** passage | 95%* | 65% |

\*From the earlier smoke-test run. Full table, caveats and per-item scores:
[`results/run2_full/RESULTS.md`](results/run2_full/RESULTS.md).

**What this shows:** fine-tuning taught the model to stay inside its source and to
decline what it cannot answer, without making it refuse more often. It did not
make answers more *helpful*, and with no passage at all it still invents. That is
why the design gives the model its source passage every time (see
[Design](#design)).

---

## Repository layout

```
notebooks/
  ijeoma_train_and_evaluate_v0_3.ipynb   ← train + evaluate (Colab, T4)
  archive/                               earlier notebook version
pipeline/                                builds the dataset from the raw scrape
  out/                                   ← the built dataset (ready to use)
models/
  ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz   ← trained LoRA adapter (5 MB)
evaluation/
  run_judge.py                           LLM-as-judge + claim-support scoring
results/
  run2_full/                             final run: scores, predictions, judge output
  run1_smoke/                            first 200-example test run
docs/
  DECISIONS.md                           every design decision, with reasons and evidence
  capstone-proposal-ijeoma-ai_SUBMITTED.docx
data/raw/                                original dataset bundle
```

---

## How to run it

There are four things you might want to do. Most people only need **1**.

| | Task | Needs | Time |
|---|---|---|---|
| **1** | Train and evaluate the model | Google account, Colab T4 GPU, Hugging Face token | ~3 h |
| **2** | Use the trained adapter without retraining | same as 1 | ~15 min |
| **3** | Score answers with the LLM judge | Python, OpenAI API key | ~5 min |
| **4** | Rebuild the dataset from scratch | Python, OpenAI API key | ~1 h |

### Before you start: Hugging Face access to Gemma

Gemma is downloaded from Hugging Face and needs a one-time licence acceptance.

1. Sign in at [huggingface.co](https://huggingface.co) and open
   [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it). Accept the licence.
2. Profile → **Settings → Access Tokens → Create new token**, type **Read**. Copy it (`hf_…`).
3. In Colab, open the **🔑 Secrets** panel (left sidebar) → **Add new secret**:
   name `HF_TOKEN`, value your token, and switch **Notebook access** on.

Never paste the token into a notebook cell.

### 1. Train and evaluate

1. **Put the data in Google Drive.** Create a folder called **`Ijeoma`** directly in
   *My Drive* and upload these six files from [`pipeline/out/`](pipeline/out/):

   ```
   conversational_training_data.jsonl   conversational_test_data.jsonl
   pairs_train_full.jsonl               pairs_test_full.jsonl
   summarization_train_full.jsonl       summarization_test_full.jsonl
   ```

   The notebook finds the folder automatically, including through a Drive
   shortcut up to two folders deep, and stops with a clear message if it can't.

2. **Open the notebook in Colab.** In Colab: **File → Open notebook → GitHub**, paste
   this repository's URL and pick `notebooks/ijeoma_train_and_evaluate_v0_3.ipynb`.
   (Or download it and use **File → Upload notebook**.)

3. **Choose a GPU.** **Runtime → Change runtime type → T4 GPU**.

4. **Run everything.** **Runtime → Run all**, and allow Drive access when asked.

What you'll see, in order:

| Stage | Takes | Look for |
|---|---|---|
| Setup and model download | ~5 min | `Tesla T4`, `project folder: …/Ijeoma`, the HF secret found |
| Base model scored on 100 test items | ~9 min | `=== base (no LoRA) ===` table |
| Training, ~6,500 steps | ~2–2.5 h | `checkpoint at step 500 …` every 500 steps |
| Fine-tuned model scored | ~9 min | `=== LoRA … ===` table and a before/after chart |
| Extra metrics, invention test, save | ~10 min | `logged to …/ijeoma_experiment_log.csv` |

Keep the tab open: free Colab disconnects idle sessions. If it does, the latest
checkpoint (`ijeoma_lora_<run>_ckpt.npz`) is already on Drive; load it as in [task 2](#2-use-the-trained-adapter-without-retraining).

Everything the run produces lands in the `Ijeoma` folder: the adapter
(`ijeoma_lora_<run>.npz`), `eval_baseline_<run>.csv`, `eval_tuned_<run>.csv`,
`eval_adversarial_<run>.csv` and `ijeoma_experiment_log.csv`.

**Settings** live in section 2 (Config). The defaults reproduce the
reported run. For a quick end-to-end check, set `SMOKE_TEST = True`
(200 examples, ~15 min in total).

### 2. Use the trained adapter without retraining

The adapter from the reported run is in
[`models/`](models/ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz).
Upload it to the `Ijeoma` Drive folder, run sections 1–2 of the notebook (setup and config), then:

```python
import numpy as np, keras_hub as kh

model = kh.models.Gemma3CausalLM.from_preset("hf://google/gemma-3-1b-it")
model.backbone.enable_lora(rank=8)
model.compile(sampler="greedy")

z = np.load("/content/drive/MyDrive/Ijeoma/ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz")
saved = {str(n): z[f"w{i}"] for i, n in enumerate(z["names"])}
for w in model.weights:
    key = getattr(w, "path", None) or w.name
    if key in saved:
        w.assign(saved[key])

prompt = ("[ANSWER] Answer the question using only the passage below. "
          "If the passage does not contain the answer, say so plainly.\n\n"
          "QUESTION: How much does it cost to visit?\n\nPASSAGE:\n"
          "Kakum National Park, in the Central Region of Ghana, covers 375 square "
          "kilometres of tropical forest. It is one of only three places in Africa "
          "with a canopy walkway.")
print(model.generate(f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
                     max_length=320))
```

Prompts must follow the formats the model was trained on; see
[Prompt formats](#prompt-formats).

### 3. Score answers with the LLM judge

The judge reads an eval CSV (which stores each prompt), scores every answerable
answer, digest and summary 1–5 on factual accuracy, cultural nuance and
helpfulness, and checks each factual claim against the source passage.

```bash
pip install pandas
export OPENAI_API_KEY=sk-...        # never commit this
python3 evaluation/run_judge.py path/to/eval_tuned_<run>.csv
```

It prints the summary, including `claims_supported_rate` (the proposal's success
criterion), and writes per-item scores to `…_judged.csv`. The judge model
defaults to `gpt-5.4-mini`; set `JUDGE_MODEL` to change it. The notebook can
also run the judge itself if `OPENAI_API_KEY` is in Colab Secrets.

### 4. Rebuild the dataset

You only need this to change the data; the built dataset is already in
`pipeline/out/`.

```bash
cd pipeline
pip install -r requirements.txt

# template pairs: classify, extractive answers, refusals (free, deterministic)
python3 ijeoma_pairs.py --passages out/passages.jsonl --out out

# teacher pairs: digests, abstractive Q&A, multi-turn dialogues (OpenAI API)
export OPENAI_API_KEY=sk-...
python3 ijeoma_teacher.py --provider openai --passages out/passages.jsonl --out out \
    --n-digest 600 --n-qa 400 --n-dialogue 400 --limit 3      # smoke test first
python3 ijeoma_teacher.py --provider openai --passages out/passages.jsonl --out out \
    --n-digest 600 --n-qa 400 --n-dialogue 400 --workers 16

# validate, split by passage (no passage in both train and test), write files
python3 ijeoma_build.py --out out --teacher out/pairs_teacher.jsonl
python3 ijeoma_summarize.py --passages out/passages.jsonl --out out
python3 ijeoma_build_summarization.py --out out
python3 test_validator.py
```

Teacher output is cached per passage in `out/teacher_cache.jsonl`, so a rerun
costs nothing; the cache from the reported run is included. To start from the
raw scrape instead of `passages.jsonl`, run `ijeoma_clean.py` first (see
[`pipeline/run_all.sh`](pipeline/run_all.sh)).

---

## Prompt formats

Every prompt starts with a task tag and ends with `PASSAGE:` followed by the source
text (or `(none available)`).

| Task | Prompt shape | Output |
|---|---|---|
| Classify | `[CLASSIFY] Classify this destination…` + allowed-label list + passage | JSON |
| Answer | `[ANSWER] Answer the question using only the passage below…` + `QUESTION:` + passage | 1–3 sentences, or a refusal |
| Conversation | `[ANSWER] Continue the conversation as the guide…` + `CONVERSATION SO FAR:` + `QUESTION:` + passage | 1–3 sentences, or a refusal |
| Digest | `[DIGEST] Give me a traveller-facing summary…` + passage | 60–100 words |
| Summarise | `[SUMMARIZE] Give a short summary of the passage below…` + passage | 1–3 sentences |

Example, classify:

```
[CLASSIFY] Tag this place for a travel catalogue. Output JSON: country, theme, terrain, traveler_profile, world_heritage.
Allowed values (copy one exactly):
theme: Town or Area Guide | Nature & Wildlife | Museum & Gallery | … (13)
terrain: Coastal | Riverine | Island | … | null
…
PASSAGE:
Uno is an island in the Bissagos archipelago off the coast of Guinea-Bissau…
```
```json
{"country": "Guinea-Bissau", "theme": "Nature & Wildlife", "terrain": "Coastal",
 "traveler_profile": "Nature and wildlife, outdoor", "world_heritage": false}
```

---

## Data

1,947 West African destination records scraped from Wikipedia, Wikivoyage and
OpenStreetMap, cleaned to **1,586 passages** across 16 countries.

| | Train | Test |
|---|---|---|
| Classify | 1,027 | 173 |
| Answer (incl. multi-turn) | 2,390 | 477 |
| Refuse | 1,133 | 238 |
| Digest | 456 | 70 |
| Summarise | 1,548 | 283 |
| **Total** | **6,554** | **1,241** |

- **Split by passage:** no destination appears in both train and test (checked in code).
- **Country cap:** no country exceeds 22% of a task's sample; Nigeria is 45% of the raw corpus.
- **Two sources of examples:** 2,498 from code templates, and 3,476 written by a
  teacher model (`gpt-5.4-mini`) from the passage and kept only if every name and
  number appears in the passage, the label matches the content, and the text is
  in English (260 rejected).

Details: [`pipeline/out/build_report.json`](pipeline/out/build_report.json),
[`pipeline/out/teacher_report.json`](pipeline/out/teacher_report.json).

---

## Design

**The passage goes in the prompt.** The proposal ruled out retrieval, but a 1B
model asked about a place from memory invents fluent, wrong answers (95% of the
time in our first run). Fine-tuning teaches format and register, not facts. So
every task takes its source passage as input, and the model learns to read rather
than recall. In deployment the passage would come from a lookup step, which is the
"retrieval grounding" named in the proposal's governance section.

**Gemma 3 1B with LoRA** (rank 8, 0.13% of weights trained, 1 epoch, lr 2e-4). It
trains on a free Colab T4 in about two hours without 4-bit quantisation.

Every decision, with its reason and evidence, is in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

### Differences from the submitted proposal

| Proposal | This build | Why |
|---|---|---|
| Gemma 4 E4B with QLoRA | Gemma 3 1B with LoRA | Team decision; fits a free T4 (D-18) |
| No external retrieval | Passage supplied in every prompt | 1B model invents without a source (D-02) |
| Text, images, tabular | Text only | Time |
| Teacher once per passage | 1,026 passages, some sent 2–3 times (one per task) | Separate prompts per task (D-17) |
| HF Space demo, Model Card, Data Card | Not yet built | Time |

## Limitations

- Classification results rest on 8 test items and digests on 2; treat them as indicative.
- Answers are faithful but often minimal; helpfulness did not improve with training.
- Without a passage the model still invents 65% of the time. Do not deploy it without a source lookup.
- The judge is from the same model family as the teacher, which may favour that style.
- Some passages still carry Wikivoyage listing text or opinionated wording from the source.
- Not for safety, security, visa, health or border advice, ever.

## Licence and attribution

Source text: Wikipedia and Wikivoyage (CC BY-SA 4.0), OpenStreetMap (ODbL). The
derived dataset is shared under CC BY-SA 4.0. Model use is governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Team

John Hicks · Prof Chika Yinka-Banjo · Usman Kolawole (University of Lagos)
