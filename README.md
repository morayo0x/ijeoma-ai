# Ijeoma AI

Ijeoma AI is a grounded question-answering and summarisation model for West
African destinations. It is Gemma 3 1B instruction-tuned, fine-tuned with LoRA
to perform three tasks on a passage describing a place:

- **Classification** for a travel catalogue: country, theme, terrain, traveller
  profile and World Heritage status.
- **Summarization** as a short traveller digest or a one- to three-sentence summary.
- **Question answering**, including multi-turn conversation.

The model answers only from the passage it is given. Questions the passage does
not cover, or that require live information (visas, prices, safety, transport),
are declined with a pointer to official sources.

Developed as the capstone project for *AI Research Foundations for Higher
Education* (Google DeepMind and FATE Foundation), Nigeria Cohort 1, University
of Lagos.

---

## Results

Evaluated on 100 held-out items whose passages were not seen in training. The
baseline is `gemma-3-1b-it` before fine-tuning.

| Metric | Baseline | Ijeoma AI |
|---|---|---|
| **Claims supported by the source passage** (target ≥ 0.90) | 0.860 | **0.974** |
| Refusal recall (declines out-of-scope questions) | 0.44 | **0.92** |
| False refusal rate (declines answerable questions) | 0.03 | 0.03 |
| Grounded outputs (all names and numbers present in the passage) | 0.82 | **1.00** |
| Theme accuracy (n = 8) | 0.125 | **0.75** |
| Judge: factual accuracy (1–5) | 3.81 | **4.60** |
| Judge: cultural nuance (1–5) | 4.30 | **4.61** |
| Judge: helpfulness (1–5) | 3.81 | 3.76 |
| BERTScore F1 | 0.84 | **0.92** |
| Invention rate with no passage (n = 20) | 0.95¹ | 0.65 |

¹ Measured in the pipeline-validation run. Full results, sample sizes and
per-item outputs: [`results/run2_full/RESULTS.md`](results/run2_full/RESULTS.md).

Fine-tuning made the model substantially more faithful to its source and more
reliable at declining unanswerable questions, without increasing false
refusals. It did not improve helpfulness, and without a passage the model still
fabricates in most cases, which is why every prompt supplies the source passage
(see [Method](#method)).

---

## Repository structure

```
app/                         chat interface, and demo with Workbench, Chat and Compare pages
data/raw/                    original dataset bundle
docs/
  COLAB_DEMO.md              step-by-step guide to the Colab demo
  capstone-proposal-ijeoma-ai_SUBMITTED.docx
evaluation/
  run_judge.py               LLM-judge scoring and claim-support rate
models/
  ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz   trained LoRA adapter (5 MB)
notebooks/
  ijeoma_train_and_evaluate_v0_3.ipynb           training and evaluation (Colab)
  ijeoma_chat_colab.ipynb                        chat launcher (Colab)
  ijeoma_demo_colab.ipynb                        demo launcher (Colab)
  archive/                                       earlier notebook revision
pipeline/                    dataset construction
  out/                       built dataset
tests/                       routing tests for the chat interface
results/
  run1_smoke/                pipeline-validation run (200 examples)
  run2_full/                 reported run (6,522 examples)
```

---

## Quick start on Google Colab

Both notebooks run on a free Colab T4 GPU and need only a Hugging Face token
(see [Requirements](#requirements)); the code, adapter and data are fetched from
this repository.

| Notebook | Purpose | Time |
|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/morayo0x/ijeoma-ai/blob/main/notebooks/ijeoma_chat_colab.ipynb) **Chat** | Chat-only interface: ask about any destination; sources are found and cited automatically | ~5 min |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/morayo0x/ijeoma-ai/blob/main/notebooks/ijeoma_demo_colab.ipynb) **Demo** | Interactive Workbench, Chat and Compare pages behind a public link | ~5 min |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/morayo0x/ijeoma-ai/blob/main/notebooks/ijeoma_train_and_evaluate_v0_3.ipynb) **Train and evaluate** | Full fine-tuning run and evaluation, reproducing the reported results | ~3 h |

In Colab: **Runtime → Change runtime type → T4 GPU**, add the `HF_TOKEN` secret,
then **Runtime → Run all**.

---

## Requirements

- **Training and inference:** a GPU with at least 12 GB of memory (a free Colab
  T4 is sufficient), and a Hugging Face account that has accepted the
  [Gemma licence](https://huggingface.co/google/gemma-3-1b-it).
- **Dataset construction and judge evaluation:** Python 3.10+, `pandas`, and an
  OpenAI API key.

Create a Hugging Face access token (Settings → Access Tokens, type *Read*). In
Colab, store it as a secret named `HF_TOKEN` (key icon in the left sidebar) with
notebook access enabled; elsewhere, export it as `HF_TOKEN`. Do not place tokens
in notebook cells or commit them.

---

## Usage

| Task | Section | Time |
|---|---|---|
| Train and evaluate | [Training](#training) | ~3 h on a T4 |
| Run the trained adapter | [Inference](#inference) | ~5 min |
| Score outputs with the LLM judge | [Judge evaluation](#judge-evaluation) | ~5 min |
| Rebuild the dataset | [Dataset construction](#dataset-construction) | ~1 h |
| Interactive demo | [Demo](#demo) | ~5 min |

### Training

Open [`notebooks/ijeoma_train_and_evaluate_v0_3.ipynb`](https://colab.research.google.com/github/morayo0x/ijeoma-ai/blob/main/notebooks/ijeoma_train_and_evaluate_v0_3.ipynb) in Colab,
select a T4 GPU runtime, and run all cells. The notebook mounts Google Drive and
uses a folder named `Ijeoma` in *My Drive* (or reachable through a shortcut up to
two levels deep); if none exists, it creates one and downloads the dataset from
this repository.

| Stage | Duration | Output |
|---|---|---|
| Setup and model download | ~5 min | GPU, Drive folder and token are reported |
| Baseline evaluation (100 items) | ~9 min | baseline metrics |
| Fine-tuning (~6,500 steps) | 2–2.5 h | checkpoint written every 500 steps |
| Fine-tuned evaluation | ~9 min | metrics and before/after chart |
| Additional metrics, invention test, logging | ~10 min | `ijeoma_experiment_log.csv` |

Outputs are written to the same `Ijeoma` folder: the adapter
(`ijeoma_lora_<run>.npz`), `eval_baseline_<run>.csv`, `eval_tuned_<run>.csv`,
`eval_adversarial_<run>.csv` and the experiment log. If the session disconnects,
the latest checkpoint (`ijeoma_lora_<run>_ckpt.npz`) can be loaded as described
under [Inference](#inference).

Hyperparameters are set in section 2 of the notebook; the defaults reproduce
the reported run. `SMOKE_TEST = True` runs a 200-example check in about 15 minutes.

### Inference

The adapter from the reported run is in [`models/`](models/). To load it with
KerasHub on the JAX backend:

```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras_hub
import numpy as np

model = keras_hub.models.Gemma3CausalLM.from_preset("hf://google/gemma-3-1b-it")
model.backbone.enable_lora(rank=8)
model.compile(sampler="greedy")

archive = np.load("models/ijeoma_lora_gemma-3-1b-it-r8-lr0.0002-e1.npz")
saved = {str(name): archive[f"w{i}"] for i, name in enumerate(archive["names"])}
for weight in model.weights:
    key = getattr(weight, "path", None) or weight.name
    if key in saved:
        weight.assign(saved[key])

prompt = (
    "[ANSWER] Answer the question using only the passage below. "
    "If the passage does not contain the answer, say so plainly.\n\n"
    "QUESTION: How much does it cost to visit?\n\nPASSAGE:\n"
    "Kakum National Park, in the Central Region of Ghana, covers 375 square "
    "kilometres of tropical forest. It is one of only three places in Africa "
    "with a canopy walkway."
)
chat = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
print(model.generate(chat, max_length=512))
```

Inputs must follow the trained [prompt formats](#prompt-formats); paraphrased
instructions degrade performance. `app/prompts.py` provides builders for each task.

### Judge evaluation

`evaluation/run_judge.py` scores each answerable answer, digest and summary in an
evaluation CSV for factual accuracy, cultural nuance and helpfulness (1–5), and
checks every factual claim against the source passage.

```bash
pip install pandas
export OPENAI_API_KEY=...
python evaluation/run_judge.py results/run2_full/eval_tuned_with_input.csv
```

The script prints a JSON summary, including `claims_supported_rate`, and writes
per-item scores to `<input>_judged.csv`. The judge defaults to `gpt-5.4-mini`
(`--model` or `JUDGE_MODEL` to change it).

### Dataset construction

The built dataset is included in `pipeline/out/`; rebuilding is only needed to
change it.

```bash
cd pipeline
pip install -r requirements.txt

# Template examples: classification, extractive answers, refusals (deterministic)
python ijeoma_pairs.py --passages out/passages.jsonl --out out

# Teacher examples: digests, abstractive Q&A, multi-turn dialogues
export OPENAI_API_KEY=...
python ijeoma_teacher.py --provider openai --passages out/passages.jsonl --out out \
    --n-digest 600 --n-qa 400 --n-dialogue 400 --workers 16

# Validation, passage-level split, output files
python ijeoma_build.py --out out --teacher out/pairs_teacher.jsonl
python ijeoma_summarize.py --passages out/passages.jsonl --out out
python ijeoma_build_summarization.py --out out
python test_validator.py
```

Teacher responses are cached in `out/teacher_cache.jsonl`; with the included
cache, a rebuild makes no API calls and reproduces the published files byte for
byte. To start from the raw records, run `ijeoma_clean.py` first (see
[`pipeline/run_all.sh`](pipeline/run_all.sh)).

### Chat

A single chat window for end users. Each question is routed to a destination
passage before the model answers: by place name; by recommendation for general
requests ("where should I visit in Ghana?", "somewhere for wildlife"), ranked by
UNESCO World Heritage status and theme; by keyword search for other broad
requests; or kept on the current destination for follow-up questions. Lists are
offered as numbered choices. Depending on the request, the model answers
conversationally, writes a traveller digest ("why should I visit?"), summarises
("in one sentence", "what is it known for?") or classifies the place ("what kind
of place is it?"). Each reply cites its source and links to the location on
OpenStreetMap where coordinates are available. A pasted description of a place
that is not in the corpus (40 words or more, in sentences) is classified and
summarised, and later questions are answered from that description. Questions
about languages or climate are answered from the destination's passage if it
covers them, otherwise from the nearest passage that does (within 150 km) or the
country's regional guide, and the reply names that source; requests for current
weather are declined. Country-level questions ("tell me about Ghana", "what
languages are spoken in Mali?") are answered from Wikivoyage country guides
(`pipeline/out/country_guides.jsonl`, fetched with `pipeline/fetch_country_guides.py`),
which are used only by the chat and are not part of the training or test data. Places that are not in the corpus are reported as not
found rather than guessed, visa, safety and similar questions are declined, and
every answer cites its source. Launch it on Colab with
[`notebooks/ijeoma_chat_colab.ipynb`](https://colab.research.google.com/github/morayo0x/ijeoma-ai/blob/main/notebooks/ijeoma_chat_colab.ipynb), or locally with
`python app/chat_app.py --share`. Routing tests: `python -m pytest tests`.

### Demo

A Gradio application with three pages: **Workbench** (any task on any passage),
**Chat** (multi-turn conversation) and **Compare** (baseline and fine-tuned
model side by side). On Colab, run
[`notebooks/ijeoma_demo_colab.ipynb`](https://colab.research.google.com/github/morayo0x/ijeoma-ai/blob/main/notebooks/ijeoma_demo_colab.ipynb), which prints a public
link on a free T4 GPU (step-by-step guide: [`docs/COLAB_DEMO.md`](docs/COLAB_DEMO.md)).
On a Linux GPU host:

```bash
bash run_demo.sh                # sets up a venv, asks for the HF token, starts the app
```

See [`app/README.md`](app/README.md) for options.

---

## Prompt formats

Each prompt begins with a task tag and ends with `PASSAGE:` followed by the
source text, or `(none available)`.

| Task | Prompt structure | Output |
|---|---|---|
| Classification | `[CLASSIFY]` instruction, permitted label values, passage | JSON object |
| Question answering | `[ANSWER]` instruction, `QUESTION:`, passage | 1–3 sentences or a refusal |
| Conversation | `[ANSWER]` instruction, `CONVERSATION SO FAR:`, `QUESTION:`, passage | 1–3 sentences or a refusal |
| Digest | `[DIGEST]` instruction, passage | 60–100 words |
| Summary | `[SUMMARIZE]` instruction, passage | 1–3 sentences |

Example (classification, abridged):

```
[CLASSIFY] Tag this place for a travel catalogue. Output JSON: country, theme, terrain, traveler_profile, world_heritage.
Allowed values (copy one exactly):
theme: Town or Area Guide | Nature & Wildlife | Museum & Gallery | ...
terrain: Coastal | Riverine | Island | ... | null
...
PASSAGE:
Uno is an island in the Bissagos archipelago off the coast of Guinea-Bissau...
```

```json
{"country": "Guinea-Bissau", "theme": "Nature & Wildlife", "terrain": "Coastal",
 "traveler_profile": "Nature and wildlife, outdoor", "world_heritage": false}
```

---

## Method

**Grounded design.** Fine-tuning adapts format and register but does not
reliably add factual knowledge, particularly at 1B parameters. In the first run,
the model fabricated answers about places it was given no passage for in 95% of
cases. Every task therefore takes its source passage as input, and the model is
trained to read rather than recall. In deployment, passages would be supplied by
a lookup or retrieval step.

**Model and training.** `gemma-3-1b-it` with LoRA (rank 8; 1.3M trainable
parameters, 0.13% of the model), trained for one epoch with AdamW (learning rate
2e-4, weight decay 0.01, effective batch size 4) and a maximum sequence length
of 960 tokens. Training runs in full precision on a single T4.

**Evaluation.** The baseline and fine-tuned models are scored on the same 100
held-out items with greedy decoding, broken down by task. Metrics include exact
match on classification labels, grounding, refusal recall and false-refusal
rate, ROUGE-1/L, BERTScore and an LLM-judge rubric. The primary metric,
`claims_supported_rate`, is the share of atomic factual claims in generated
answers and summaries that the source passage supports. A separate adversarial
set measures fabrication when no passage is given.

---

## Data

1,947 West African destination records collected from Wikipedia, Wikivoyage and
OpenStreetMap were cleaned to 1,586 passages covering 16 countries.

| Task | Train | Test |
|---|---|---|
| Classification | 1,027 | 173 |
| Question answering (incl. multi-turn) | 2,390 | 477 |
| Refusal | 1,133 | 238 |
| Digest | 456 | 70 |
| Summary | 1,548 | 283 |
| **Total** | **6,554** | **1,241** |

- **Passage-level split:** no passage appears in both training and test data.
- **Country balancing:** no country exceeds 22% of a task's sample (Nigeria is
  45% of the cleaned corpus).
- **Example sources:** 2,498 examples from deterministic templates and 3,476
  generated by a teacher model (`gpt-5.4-mini`). Teacher examples are kept only
  if every name and number appears in the passage, the label agrees with the
  content, and the text is in English; 260 were rejected.

Build statistics: [`pipeline/out/build_report.json`](pipeline/out/build_report.json),
[`pipeline/out/teacher_report.json`](pipeline/out/teacher_report.json).

---

## Deviations from the proposal

| Submitted proposal | This work | Reason |
|---|---|---|
| Gemma 4 E4B with QLoRA | Gemma 3 1B with LoRA, full precision | Fine-tunes on a single free T4 without quantisation; the smallest model that meets the targets |
| No external retrieval | Source passage supplied in every prompt | Without a source, the model fabricated answers in 95% of cases |
| Text, image and tabular data | Text only | Project timeline |
| Each passage sent to the teacher once | 1,026 passages; 486 sent two or three times | Digests, question-answer pairs and dialogues used separate prompts |
| Model Card, Data Card, hosted demo | Demo application built; cards pending | Project timeline |

## Limitations

- Classification and digest results rest on 8 and 2 test items respectively and
  should be read as indicative.
- Outputs are faithful but often minimal; helpfulness did not improve with
  fine-tuning, and 30% of outputs reproduce the reference verbatim.
- Without a source passage the model fabricates in 65% of cases. It must not be
  deployed without a retrieval step.
- The judge model belongs to the same family as the teacher model and may favour
  its style.
- Some passages retain listing text or opinionated wording from the sources.
- The model must not be used for safety, security, visa, health or border advice.

## Licence

Source text is from Wikipedia and Wikivoyage (CC BY-SA 4.0) and OpenStreetMap
(ODbL); the derived dataset is distributed under CC BY-SA 4.0. Use of the model
is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Authors

Prof Chika Yinka-Banjo · John Hicks · Usman Kolawole — University of Lagos

