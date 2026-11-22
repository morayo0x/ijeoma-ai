# Ijeoma AI — Decision Log

Every decision that shapes the project, in order. Each entry records what we
decided, why, what we gave up, and the evidence. This is the source for the
report's methodology, limitations and design-rationale sections.

**How to add an entry:** copy the template at the bottom, give it the next
number, and fill every field. Never edit an old entry to change history: if a
decision is reversed, add a new entry that says so and points back to it.

Status values: `active` · `superseded by D-nn` · `open`.

---

## D-01 · Scope the project to West Africa
- **Date:** before 14 Nov 2026 (proposal)
- **Decision:** Train and evaluate on West African destinations only, though
  the proposal frames the problem continent-wide.
- **Why:** The scraped corpus (1,947 rows) covers 16 West African countries.
  Five-region coverage was out of reach in the time available.
- **Trade-off:** The "five sub-region quota" safeguard in the proposal becomes
  a per-country quota inside one region. Claims in the report must say
  "West Africa", not "Africa".
- **Evidence:** proposal §5 (Accountability Measures, item 1); `pipeline/out/clean_report.json`.
- **Status:** active

## D-02 · Put the source passage in every prompt (grounded design)
- **Date:** 14 Nov 2026 (Dataset Assessment doc)
- **Decision:** Every task takes its source passage as input. The model reads
  and reformulates; it is not expected to recall facts.
- **Why:** The proposal promised factual accuracy while ruling out retrieval.
  At 1B parameters those conflict: fine-tuning teaches format and register,
  not facts. Grounding turns factual accuracy into a reading-comprehension
  problem a 1B model can handle.
- **Trade-off:** This is not an open-domain chatbot. In a deployed system the
  passage must come from somewhere (a lookup or a retrieval step), which
  partly walks back the proposal's "no retrieval" architecture (§5, §7).
- **Evidence:** Run 1 invention rate 0.95: with no passage, the model invented
  answers about fictitious places (e.g. placed "Olumbe Rock Shelters" in the
  UK). `results/eval_adversarial_*.csv`.
- **Status:** active. The report must state this as a deliberate change from
  the proposal.

## D-03 · Clean the corpus before generating anything
- **Date:** 14–15 Nov 2026
- **Decision:** Strip Wikivoyage commercial listings (phone numbers, opening
  hours, "updated" stamps) and `[Additional source]` markers; drop passages
  under the length floor, duplicate names and non-places; normalise
  `attraction_type` into closed label sets (13 themes, 9 terrains, 12
  traveller profiles).
- **Why:** Training on listings teaches the model to emit live logistics, which
  the proposal puts out of scope. Multi-valued free-text types cannot serve as
  classification targets.
- **Trade-off:** 1,947 → 1,586 passages (249 too short after cleaning, 53
  duplicates, 52 not places, 7 boilerplate). 645 passages have no terrain label.
- **Evidence:** `pipeline/out/clean_report.json`.
- **Status:** active

## D-04 · Cap any one country's share of the training data
- **Date:** 14–15 Nov 2026
- **Decision:** Sample passages with a per-country cap of 22% and report results
  per country.
- **Why:** Nigeria and Ghana are 58% of the raw corpus (Nigeria alone 45% of
  cleaned passages). Uniform sampling would make the headline score a
  Nigeria score.
- **Trade-off:** Liberia, Togo, Guinea and Guinea-Bissau still have only 15–33
  passages each: per-country results for them rest on a handful of documents.
- **Evidence:** `pipeline/out/build_report.json` (`test_by_country`).
- **Status:** active

## D-05 · Split train/test by passage, not by pair
- **Date:** 15 Nov 2026
- **Decision:** Hold out whole passages (≈15%), stratified by country. The
  summarisation set inherits the same split.
- **Why:** Several pairs come from one passage; splitting by pair would let the
  model see the test passage during training.
- **Evidence:** `passage_leakage: 0`, `cross_dataset_leakage: 0` in the build reports.
- **Status:** active

## D-06 · Evaluate the untuned model first, per task, plus an invention test
- **Date:** 15–16 Nov 2026 (notebook design)
- **Decision:** Score base Gemma 3 1B and the LoRA model on the same items with
  the same metrics, broken down by task (classify, answer, digest, summarise,
  refuse), and add an adversarial set of invented and out-of-region places.
- **Why:** Without a baseline there is no evidence fine-tuning helped. A single
  blended score hides which task is failing.
- **Status:** active

## D-07 · Run 1: smoke test only
- **Date:** 16 Nov 2026 (re-logged 22 Nov)
- **Decision:** First run at 200 training examples, 1 epoch, rank 8, lr 2e-4,
  20 eval items, to prove the pipeline end to end.
- **Result:** Pipeline works. Scores near the baseline (expected at this size):
  country 0.857 → 0.857, theme 0.000 → 0.000, terrain 0.071 → 0.286,
  ROUGE-L 0.466 → 0.389, refusal recall 1.0, invention rate 0.95.
- **Note:** The log holds three identical rows: the same configuration was run
  three times. Treat them as one data point.
- **Evidence:** `results/ijeoma_experiment_log.csv`.
- **Status:** active (baseline for comparison)

## D-08 · Separate Ijeoma AI from the Quanta AI repository
- **Date:** 22 Nov 2026
- **Decision:** The capstone lives in its own project folder and git repository
  (`ijeoma-ai`), unrelated to Quanta AI.
- **Status:** active. GitHub remote to be added once created.

## D-09 · List the allowed labels in the classify prompt
- **Date:** 22 Nov 2026
- **Decision:** Every `[CLASSIFY]` prompt now carries the closed label menu for
  theme, terrain, traveller profile and world heritage.
- **Why:** Theme accuracy was exactly 0.000 for both base and tuned models.
  Cause: the prompt never named the allowed labels, so the model invented its
  own ("Nature", "Hiking and Trekking") and exact-match scoring rejected them.
- **Alternative rejected:** Fuzzy-matching labels at scoring time. It would hide
  the problem rather than fix it, and a catalogue needs the exact label.
- **Trade-off:** Classify prompts are ~120 tokens longer, so training is slightly slower.
- **Evidence:** `results/eval_tuned_*.csv` (`pred_theme` vs `true_theme`);
  `pipeline/ijeoma_pairs.py` (`LABEL_MENU`).
- **Status:** active

## D-10 · Use OpenAI as the teacher model instead of Gemini
- **Date:** 22 Nov 2026
- **Decision:** Run the teacher step on `gpt-5.4-mini` via an OpenAI API key
  (`--provider openai`). The Gemini path stays in the code.
- **Why:** The proposal named Gemini 1.5, but the Gemini pass never ran (4 of
  2,502 pairs came from the teacher; 3 digests in total). An OpenAI key was
  available on deadline day.
- **Trade-off:** Departs from the proposal's named teacher. Report it as a
  substitution of convenience; the prompts and validation rules are unchanged.
- **Sizing:** 600 digest passages, 400 QA passages, 400 dialogue passages
  (×1.25 over-generation) = 1,741 API calls. Sized to finish within the hour.
- **Status:** active

## D-11 · Add multi-turn guide dialogues
- **Date:** 22 Nov 2026
- **Decision:** The teacher writes 3-exchange traveller–guide conversations
  per passage. Each guide turn becomes one training pair whose input holds the
  conversation so far. The third exchange is always an out-of-scope practical
  question (cost, hours, transport, safety) that the guide declines.
- **Why:** The proposal promises multi-turn dialogue (§2, task 3). The
  existing "conversational" data was single-turn only.
- **Why one pair per turn:** It keeps the input/output format the notebook
  already trains on, while still teaching follow-ups in context.
- **Status:** active

## D-12 · Reject teacher pairs whose label contradicts the output
- **Date:** 22 Nov 2026
- **Decision:** The validator now rejects teacher pairs labelled "refuse" whose
  output answers, pairs labelled "answer" whose output refuses, and non-English
  outputs.
- **Why:** 40 mislabelled pairs and 4 non-English outputs (one in Italian) were
  found in the first teacher pass. Mislabelled refusals teach the model to
  refuse answerable questions, or the reverse.
- **Evidence:** `pipeline/out/teacher_report.json` (`refusal_label_mismatch: 40`,
  `not_english: 4`). Uses the same refusal pattern as the notebook's scorer.
- **Status:** active

## D-13 · Train on the full dataset for one epoch, with checkpoints
- **Date:** 22 Nov 2026 (submission 23:59 WAT)
- **Decision:** `TRAIN_SUBSET = None` (≈6,500 examples), 1 epoch, LoRA rank 8,
  lr 2e-4, effective batch 4. Adapter written to Drive every 500 steps.
- **Why:** At ~1.3 s/step on a T4 that is 2.5–3 h, which fits the day.
  Checkpoints protect against a Colab disconnect late in the run.
- **Alternative considered:** 1,500-example subset (~35 min). Rejected once it
  was clear the day had room for the full run.
- **Status:** active

## D-14 · Add the remaining metrics promised in the proposal
- **Date:** 22 Nov 2026
- **Decision:** Add per-class precision/recall/F1 (theme), ROUGE-1, BERTScore
  F1 and an LLM-as-judge rubric (1–5: factual accuracy, cultural nuance,
  helpfulness) to both base and tuned summaries and to the experiment log.
- **Choices:** BERTScore uses `distilbert-base-uncased` on CPU, because JAX
  holds the GPU and `roberta-large` is slow on CPU; report the model next to
  the score. Judge: `gpt-5.4-mini`, 40 grounded items per model, scored
  against the source passage.
- **Not done:** readability scores (proposal §3).
- **Status:** active

## D-15 · The submitted proposal is the reference document
- **Date:** 22 Nov 2026
- **Decision:** Measure the build against `docs/capstone-proposal-ijeoma-ai_SUBMITTED.docx`
  (Nigeria Cohort 1, submitted 1 Nov 2026 by John Hicks; team: Prof Chika
  Yinka-Banjo, Usman Kolawole). The longer draft
  (`docs/African_Tourism_Capstone_Proposal.docx`) is background only.
- **Effect on earlier entries:** D-01's "five sub-regions" gap does not apply:
  the submitted proposal is West Africa only. D-02 still stands, but note the
  submitted proposal contradicts itself: §1 rules out external retrieval while
  §7 names "retrieval grounding" as the main risk control. The
  passage-in-prompt design is consistent with §7.
- **Status:** active

## D-16 · Measure the proposal's success criterion directly
- **Date:** 22 Nov 2026
- **Decision:** Add `claims_supported_rate` to the notebook: the LLM judge lists
  every atomic factual claim in an answer or summary and marks whether the
  source passage supports it; the rate is supported ÷ total across judged items.
- **Why:** The submitted proposal's success criterion is "90% of the claims in
  its answers and summaries are supported by the source passage", and its
  metric list includes Factual Consistency. Nothing in the notebook measured
  either. The existing `grounded` check only tests that numbers and proper
  nouns appear in the passage.
- **Validation:** Live test on one gold digest: 18/18 supported. The same digest
  with an invented sentence ("built in 1652 by the Portuguese… jazz festival
  every June"): 9/11, both invented claims flagged, accuracy score 5 → 1.
- **Known weakness:** The judge sometimes merges two facts into one claim
  (it counted "built in 1652 by the Portuguese" as one), so the rate slightly
  flatters the model. Judged on 40 items per model: report it with that n.
- **Status:** active

## D-17 · Teacher calls: count met, "never twice" not met
- **Date:** 22 Nov 2026
- **Finding:** The submitted proposal commits to synthetic generation "once
  over 800 to 1,200 passages… no passage reaches the teacher model twice".
  The run covered 1,026 unique passages (within range) in 1,750 calls: 540
  passages were sent once, 248 twice and 238 three times, because digest, QA
  and dialogue were separate prompts.
- **Decision:** Keep the data and report this honestly. A single combined prompt
  per passage would meet the commitment in a rebuild.
- **Status:** active

## D-18 · Base model is Gemma 3 1B, not Gemma 4 E4B
- **Date:** confirmed by the team 22 Nov 2026 (already the target in the
  14 Nov Dataset Assessment doc)
- **Decision:** Fine-tune Gemma 3 1B instruction-tuned (`gemma-3-1b-it`) with
  LoRA, as the team intended, rather than Gemma 4 E4B with QLoRA as written in
  the submitted proposal (§4).
- **Why:** Team decision. *Reason to fill in for the report.* Supporting
  facts from the runs: LoRA on the 1B model fits a free Colab T4 in full
  precision (1.3M trainable parameters, 0.13%) with no 4-bit quantisation
  needed, and the proposal's own environmental commitment is to "use the
  smallest Gemma variant that clears our evaluation targets".
- **Trade-off:** A smaller model has less capacity for the reading and
  reformulation tasks; the invention rate without a passage (0.95, D-07) is
  partly a function of size. Plain LoRA instead of QLoRA is a consequence of
  the smaller model, not a separate choice.
- **Status:** active

---

## Known gaps against the submitted proposal (for the Limitations section)

| Submitted proposal says | What we did | Entry |
|---|---|---|
| Gemma 4 E4B with QLoRA | Gemma 3 1B (team decision), plain LoRA in Keras/JAX, full precision | D-18 |
| No external retrieval (§1) / retrieval grounding (§7) | Passage supplied in every prompt | D-02, D-15 |
| 90% of claims supported by the source | Now measured as `claims_supported_rate` | D-16 |
| Text, images and tabular data | Text only | open |
| Teacher once over 800–1,200 passages, never twice | 1,026 passages; 486 sent 2–3 times | D-17 |
| 1 baseline + 3 hyperparameter runs | Smoke test (logged 3×) + full run | D-07, D-13 |
| Hugging Face Space demo, Model Card, Data Card, error reporting, attribution | Not built yet | open |
| Example labels "Sacred & Cultural Landscape", "Riverine forest" | "Sacred & Ceremonial", "Riverine" | D-03 |
| Teacher model unspecified (draft: Gemini) | OpenAI `gpt-5.4-mini` | D-10 |
| 3,500 examples | ~6,550 train + ~1,240 test | exceeded |

## Open questions
- Nigeria is still ~22% of the test set; per-country scores for small countries are noisy.
- Refusals are ~17% of training data. Watch `false_refusal` after the run.
- The no-passage invention test asks "What is X?"; training refusals with no
  passage only cover logistics (visa, safety, flights). The invention rate may
  stay high. If it does, report it as a limitation.

---

## Template

```
## D-nn · <short title>
- **Date:**
- **Decision:**
- **Why:**
- **Alternatives / trade-off:**
- **Evidence:** <file, metric or run>
- **Status:** active | superseded by D-nn | open
```

---

## D-19 · Move the project to Usman's Drive folder "Ijeoma"
- **Date:** 22 Nov 2026
- **Decision:** Training data and notebook now live in the folder "Ijeoma"
  (Usman Kolawole's My Drive), replacing "Google Deep mind/Google Deepmind Tour
  Guide IJEOMA". Notebook v0.3 finds the folder automatically under
  `/content/drive/MyDrive` (directly, or via a shortcut up to two levels deep)
  and stops with a clear message if it is missing.
- **Why:** Team moved the working folder. Hard-coding one person's path breaks
  the notebook for every other team member.
- **Note:** v0.3 is v0.2 plus this path change only. The earlier experiment log
  stays in `results/`; the new folder starts a fresh `ijeoma_experiment_log.csv`.
- **Status:** active
