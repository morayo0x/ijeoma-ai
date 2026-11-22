# Design decisions

This log records the decisions that shaped Ijeoma AI: what was decided, why,
what was traded away, and the evidence. Entries are numbered in the order they
were made and are not rewritten afterwards; a reversed decision gets a new
entry that supersedes the old one.

Status values: **active**, **open** (identified, not yet resolved), or
**superseded by D-nn**.

A summary of deviations from the submitted proposal is at the
[end of this document](#deviations-from-the-submitted-proposal).

---

## D-01 · Scope: West Africa

- **Date:** November 2026 (proposal stage)
- **Decision:** Train and evaluate on West African destinations only.
- **Rationale:** The source corpus (1,947 records) covers 16 West African
  countries. Continental coverage was not feasible within the project timeline.
- **Consequences:** Per-region sampling quotas become per-country quotas within
  one region. Results are claims about West Africa, not Africa.
- **Evidence:** `pipeline/out/clean_report.json`.
- **Status:** active

## D-02 · Every prompt includes its source passage

- **Date:** 14 Nov 2026
- **Decision:** All tasks take the source passage as input. The model is trained
  to read and reformulate, not to recall facts from its weights.
- **Rationale:** Fine-tuning adapts format and register; it does not reliably
  install facts, least of all at 1B parameters. Supplying the passage turns
  factual accuracy into a reading-comprehension problem the model can solve.
- **Consequences:** The model is not an open-domain assistant. A deployment must
  supply the passage through a lookup or retrieval step.
- **Evidence:** In the first run, asked about places with no passage, the model
  produced an invented answer 95% of the time (for example, it placed the
  fictitious "Olumbe Rock Shelters" in the United Kingdom).
  `results/run1_smoke/eval_adversarial_*.csv`.
- **Status:** active

## D-03 · Clean the corpus before generating examples

- **Date:** 14–15 Nov 2026
- **Decision:** Remove commercial listings (phone numbers, opening hours,
  "updated" stamps) and inline source markers; drop passages below a length
  floor, duplicates and non-places; normalise free-text attraction types into
  closed label sets (13 themes, 9 terrains, 12 traveller profiles).
- **Rationale:** Training on listings teaches the model to emit live logistics,
  which is out of scope. Multi-valued free-text types cannot serve as
  classification targets.
- **Consequences:** 1,947 records reduce to 1,586 passages (249 too short after
  cleaning, 53 duplicates, 52 non-places, 7 boilerplate). 645 passages have no
  terrain label.
- **Evidence:** `pipeline/out/clean_report.json`.
- **Status:** active

## D-04 · Cap each country's share of the sample

- **Date:** 14–15 Nov 2026
- **Decision:** Sample passages per task with a 22% cap per country; report
  results per country.
- **Rationale:** Nigeria accounts for 45% of cleaned passages and Nigeria plus
  Ghana for 58% of the raw corpus. Uniform sampling would make aggregate scores
  predominantly Nigerian.
- **Consequences:** Liberia, Togo, Guinea and Guinea-Bissau still contribute only
  15–33 passages each; per-country results for them are indicative only.
- **Evidence:** `pipeline/out/build_report.json` (`test_by_country`).
- **Status:** active

## D-05 · Split train and test by passage

- **Date:** 15 Nov 2026
- **Decision:** Hold out whole passages (about 15%), stratified by country. The
  summarisation set inherits the same split.
- **Rationale:** Several examples are derived from each passage; a per-example
  split would expose test passages during training.
- **Evidence:** `passage_leakage: 0` and `cross_dataset_leakage: 0` in the build
  reports.
- **Status:** active

## D-06 · Baseline-first, per-task evaluation with an invention test

- **Date:** 15–16 Nov 2026
- **Decision:** Score the base model and the fine-tuned model on identical
  items with identical metrics, broken down by task, and add an adversarial set
  of fictitious and out-of-region places presented without a passage.
- **Rationale:** Without a baseline there is no evidence that fine-tuning
  helped; a single blended score hides which task is failing.
- **Status:** active

## D-07 · Run 1: pipeline validation

- **Date:** 16 Nov 2026
- **Configuration:** 200 training examples, 1 epoch, LoRA rank 8, learning rate
  2e-4, 20 evaluation items.
- **Outcome:** The pipeline ran end to end. Scores were close to the baseline,
  as expected at this size: country 0.857 → 0.857, theme 0.000 → 0.000,
  terrain 0.071 → 0.286, ROUGE-L 0.466 → 0.389, refusal recall 1.0, invention
  rate 0.95.
- **Note:** The experiment log contains three identical rows from repeated runs
  of the same configuration; they are one data point.
- **Evidence:** `results/run1_smoke/ijeoma_experiment_log.csv`.
- **Status:** active (reference point)

## D-08 · Standalone, reproducible repository

- **Date:** 22 Nov 2026
- **Decision:** Keep code, built data, trained adapter, evaluation outputs and
  documentation in a single repository, with each training run in its own
  `results/` folder.
- **Rationale:** Every reported number can be traced to a file and regenerated
  from the repository alone.
- **Status:** active

## D-09 · List the allowed labels in the classification prompt

- **Date:** 22 Nov 2026
- **Decision:** Every `[CLASSIFY]` prompt includes the closed set of permitted
  values for theme, terrain, traveller profile and World Heritage status.
- **Rationale:** Theme accuracy in run 1 was exactly 0.000 for both models. The
  prompt never named the permitted labels, so the model produced its own
  ("Nature", "Hiking and Trekking"), which exact-match scoring rejects.
- **Alternative rejected:** Fuzzy label matching at scoring time. It would mask
  the problem, and a catalogue requires the exact label.
- **Consequences:** Classification prompts are about 120 tokens longer.
- **Evidence:** `results/run1_smoke/eval_tuned_*.csv` (`pred_theme` against
  `true_theme`); `LABEL_MENU` in `pipeline/ijeoma_pairs.py`.
- **Status:** active

## D-10 · OpenAI teacher model

- **Date:** 22 Nov 2026
- **Decision:** Generate teacher examples with `gpt-5.4-mini` (`--provider
  openai`). The Gemini code path is retained.
- **Rationale:** The planned Gemini pass had not run (4 of 2,502 examples came
  from the teacher, including 3 digests). An OpenAI model was available.
- **Consequences:** Departs from the originally planned teacher (Gemini).
  Prompts and validation rules are unchanged.
- **Sizing:** 600 digest, 400 question-answer and 400 dialogue passages, with
  25% over-generation: 1,741 API calls.
- **Status:** active

## D-11 · Multi-turn dialogues

- **Date:** 22 Nov 2026
- **Decision:** The teacher writes three-exchange traveller–guide conversations.
  Each guide turn becomes one training example whose input contains the
  conversation so far. The third exchange is always a practical question the
  passage cannot answer (cost, hours, transport, safety), which the guide
  declines.
- **Rationale:** The proposal commits to conversational follow-up; the existing
  data was single-turn.
- **Design note:** One example per turn keeps the single input/output format the
  training notebook already uses, while still conditioning each reply on the
  dialogue history.
- **Status:** active

## D-12 · Reject teacher examples whose label contradicts their content

- **Date:** 22 Nov 2026
- **Decision:** The validator rejects teacher examples labelled as refusals that
  do not refuse, examples labelled as answers that do refuse, and non-English
  outputs.
- **Rationale:** The first teacher pass contained 40 mislabelled examples and 4
  non-English outputs. Mislabelled refusals train the model to refuse
  answerable questions, or the reverse.
- **Evidence:** `pipeline/out/teacher_report.json` (`refusal_label_mismatch: 40`,
  `not_english: 4`). The refusal pattern is shared with the evaluation code.
- **Status:** active

## D-13 · Run 2: full dataset, one epoch, periodic checkpoints

- **Date:** 22 Nov 2026
- **Decision:** Train on all 6,522 examples for one epoch (LoRA rank 8, learning
  rate 2e-4, effective batch 4), writing the adapter to Drive every 500 steps.
- **Rationale:** At about 1.3 s per step on a T4, one epoch takes 2–2.5 hours,
  within a single Colab session. Checkpoints bound the loss from a disconnect.
- **Alternative considered:** A 1,500-example subset (about 35 minutes).
- **Status:** active

## D-14 · Additional evaluation metrics

- **Date:** 22 Nov 2026
- **Decision:** Add per-class precision, recall and F1 for theme; ROUGE-1;
  BERTScore F1; and an LLM-judge rubric (1–5 for factual accuracy, cultural
  nuance and helpfulness).
- **Choices:** BERTScore uses `distilbert-base-uncased` on CPU, since the GPU is
  held by the model under test; the backbone is reported alongside the score.
- **Not implemented:** readability scores.
- **Status:** active

## D-15 · The submitted proposal is the reference

- **Date:** 22 Nov 2026
- **Decision:** Deviations are measured against
  `docs/capstone-proposal-ijeoma-ai_SUBMITTED.docx` (Nigeria Cohort 1,
  1 Nov 2026).
- **Note:** The submitted proposal excludes external retrieval in §1 and names
  retrieval grounding as the main risk control in §7. The passage-in-prompt
  design (D-02) is consistent with §7.
- **Status:** active

## D-16 · Measure the success criterion directly

- **Date:** 22 Nov 2026
- **Decision:** Report `claims_supported_rate`: an LLM judge lists every atomic
  factual claim in each answer or summary and marks whether the source passage
  supports it; the rate is supported claims over all claims.
- **Rationale:** The proposal's success criterion is that 90% of claims in
  answers and summaries are supported by the source passage. The existing
  `grounded` check tests only that numbers and proper nouns appear in the
  passage.
- **Validation:** On a reference digest the judge marked 18/18 claims supported.
  With an invented sentence appended ("built in 1652 by the Portuguese … jazz
  festival every June") it marked 9/11, flagging both invented sentences, and
  factual accuracy fell from 5 to 1.
- **Known limitation:** The judge occasionally merges two facts into one claim,
  which slightly favours the model.
- **Status:** active

## D-17 · Teacher coverage

- **Date:** 22 Nov 2026
- **Finding:** The proposal commits to running synthetic generation once over
  800–1,200 passages, with no passage sent to the teacher twice. The run covered
  1,026 unique passages, but because digests, question-answer pairs and
  dialogues used separate prompts, 248 passages were sent twice and 238 three
  times.
- **Decision:** Keep the data and report the deviation. A single combined prompt
  per passage would meet the commitment in a rebuild.
- **Status:** active

## D-18 · Base model: Gemma 3 1B

- **Date:** 22 Nov 2026
- **Decision:** Fine-tune `gemma-3-1b-it` with LoRA in full precision, in place
  of the Gemma 4 E4B with QLoRA named in the submitted proposal (§4).
- **Rationale:** The 1B model fine-tunes in full precision on a single free Colab
  T4 (1.3M trainable parameters, 0.13% of the model) without 4-bit
  quantisation, and is consistent with the proposal's commitment to use the
  smallest Gemma variant that meets the evaluation targets.
- **Consequences:** Less capacity for reading and reformulation. Plain LoRA in
  place of QLoRA follows from the model size and is not a separate choice.
- **Status:** active

## D-19 · Location-independent notebook paths

- **Date:** 22 Nov 2026
- **Decision:** The training notebook locates its working folder (`Ijeoma`) by
  searching Google Drive, directly or through a shortcut up to two levels deep,
  and fails with an explicit message if it is absent.
- **Rationale:** A hard-coded personal Drive path makes the notebook unusable by
  anyone else.
- **Status:** active

## D-20 · Judge evaluation runs outside the training notebook

- **Date:** 22 Nov 2026
- **Decision:** Evaluation CSVs store each item's full prompt (`input` column).
  The judge (`evaluation/run_judge.py`) runs on these files after training. The
  notebook still runs the judge itself if an OpenAI key is present in Colab
  Secrets.
- **Rationale:** API keys should not be placed in a shared notebook, and Colab
  Secrets are per account, so every person running the notebook would otherwise
  need their own key.
- **Status:** active

## D-21 · Data quality findings from manual review

- **Date:** 22 Nov 2026
- **Finding:** Reviewing sampled examples from every task found:
  1. Residual listing text in some passages (Igbo-Ukwu: "Daylight hours with
     family permission … Tip for local guides expected"), contrary to D-03.
  2. Questionable automatic labels (Ile Oluji, an agricultural town, labelled
     terrain "Urban").
  3. Extractive summaries: references are the passage's opening sentences, and a
     request for two to three sentences sometimes yields two.
  4. Opinionated source text carried through (Gorom-Gorom: "The guides are
     pests"), a representational risk identified in the proposal.
- **Decision:** Retain the data for run 2 and report these as limitations. A
  rebuild should tighten the listing filter and review the terrain rules.
- **Status:** open

## D-22 · Run 2 results

- **Date:** 22 Nov 2026
- **Configuration:** As D-13. Final training loss 0.027, validation loss 0.028.
- **Headline:** 97.4% of claims in answers, digests and summaries are supported
  by the source passage (base model: 86.0%), against a 90% target. Refusal
  recall rose from 0.44 to 0.92 with false refusals unchanged at 0.03. Theme
  accuracy rose from 0.125 to 0.75 (8 items).
- **Method note:** The run used a notebook revision that did not export prompts.
  The test order was reconstructed from the fixed seed (812) and verified
  against all 100 saved reference answers before judging.
- **Limitations:** Classification rests on 8 items and digests on 2; helpfulness
  did not improve (3.81 → 3.76); 30% of tuned outputs reproduce the reference
  verbatim; the judge shares a model family with the teacher; with no passage
  the model still invents 65% of the time (from 0.95), which supports D-02 and
  implies that a deployment requires retrieval.
- **Evidence:** `results/run2_full/RESULTS.md`.
- **Status:** active

## D-23 · Interactive demo

- **Date:** 22 Nov 2026
- **Decision:** A Gradio application (`app/`) with three routes: `/` runs any
  task, `/chat` holds a multi-turn conversation, and `/compare` shows the base
  and fine-tuned models side by side. It runs on Colab with a public link
  (`notebooks/ijeoma_demo_colab.ipynb`) or on any NVIDIA GPU host.
- **Design notes:** Prompt wording is imported from the pipeline, and all six
  prompt types were verified byte-for-byte against training inputs. The 40
  sample passages are drawn from the test split, two to three per country.
  Base and fine-tuned models are separate instances, the base loaded on first
  use, so switching never triggers recompilation. Generation lengths are
  bucketed to bound recompilation. Requests are served sequentially.
- **Verification:** Routing, chat history and the comparison layout were tested
  in a headless browser with a stub model (`--stub`). End-to-end inference with
  the real model on GPU is pending.
- **Status:** active

## D-24 · Self-contained Colab notebooks

- **Date:** 22 Nov 2026
- **Decision:** Both notebooks run from the public repository with no manual
  setup beyond a Hugging Face token. The training notebook creates the `Ijeoma`
  Drive folder if absent and downloads the dataset from GitHub; the demo
  notebook clones the repository. The README links both with "Open in Colab"
  buttons.
- **Rationale:** The repository is shared with collaborators and reviewers who
  do not have access to the team's Drive folder.
- **Status:** active

---

## Deviations from the submitted proposal

| Submitted proposal | This work | Entry |
|---|---|---|
| Gemma 4 E4B with QLoRA | Gemma 3 1B with LoRA, full precision | D-18 |
| No external retrieval (§1); retrieval grounding (§7) | Source passage supplied in every prompt | D-02, D-15 |
| 90% of claims supported by the source | Measured: 97.4% | D-16, D-22 |
| Text, image and tabular data | Text only | open |
| Teacher runs once over 800–1,200 passages, none twice | 1,026 passages; 486 sent two or three times | D-17 |
| One baseline plus three hyperparameter runs | Pipeline-validation run plus one full run | D-07, D-13 |
| Hugging Face Space, Model Card, Data Card, error reporting | Demo application built; cards and reporting channel not yet | D-23, open |
| Example labels "Sacred & Cultural Landscape", "Riverine forest" | "Sacred & Ceremonial", "Riverine" | D-03 |
| 3,500 examples | 6,554 training and 1,241 test examples | — |

## Open issues

- Smaller countries contribute few test passages; their per-country scores are noisy.
- The no-passage refusals in training cover logistics questions only (visas,
  safety, flights), not general "what is X?" questions, which likely explains
  the remaining invention rate.
- Data quality findings in D-21.

## Entry template

```
## D-nn · <title>

- **Date:**
- **Decision:**
- **Rationale:**
- **Alternatives / consequences:**
- **Evidence:**
- **Status:** active | open | superseded by D-nn
```
