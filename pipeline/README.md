> **Historical note:** this README describes the pipeline as first built (template pairs only). The teacher pass has since been run on OpenAI, adding digests, abstractive QA and multi-turn dialogues. Current numbers are in the top-level README and `out/build_report.json`.

# Ijeoma AI — dataset augmentation pipeline

Turns the raw West Africa scrape (1,947 rows of free text) into an
instruction-tuning dataset for Gemma 3 1B, in the exact format the course
lab's Activity 4 expects.

Everything except the teacher pass is deterministic, free and reproducible:
same input, same seed, same output, every time.

## What it produced

| Stage | Result |
| --- | --- |
| Raw rows in | 1,947 |
| Passages kept | **1,586** |
| Rejected | 249 too short after cleaning · 53 duplicate names · 52 not a place · 7 web boilerplate |
| Instruction pairs | **2,502** — train 2,127 / test 375 |
| Summarization pairs | **1,831** — train 1,550 / test 281 |
| Built without a single API call | **all of it** |
| Passage leakage, within and across both sets | **0** (asserted in code) |
| Validation failures | 0 |

The two datasets share one passage-level split, so you can train on either or
on both without a passage crossing from one set's training side to the
other's test side.

Longest example is ~771 tokens, so `max_seq_len=1024` truncates nothing.

### Pair mix

| Task | Count | Output comes from |
| --- | --- | --- |
| `classify` | 1,200 | Normalised labels — exact, no model involved |
| `answer` | 899 | Spans lifted verbatim from the passage |
| `refuse` | 398 | 248 abstentions (checked against the passage first) + 150 scope refusals |
| `digest` | teacher pass | Abstractive, 60–100 words — needs an API key |

Run the teacher pass to add ~1,000 digests and ~700 abstractive QA pairs and
you land at roughly 3,500 — the target in the proposal.

### Summarization set

Separate files, separate task tag (`[SUMMARIZE]`), same passages.

| Variant | Count | Mean length | What it teaches |
| --- | --- | --- | --- |
| `summary` | 944 | 55 words | Two to three sentences — the main task |
| `one_sentence` | 667 | 17 words | Compression to a single line |
| `notable` | 220 | 32 words | Selecting for significance, not position |

Targets are **extractive**: every sentence is lifted verbatim from its
passage. That makes each pair faithful by construction and free to produce,
and it teaches *selection* — deciding what matters in a destination passage,
which is the harder half of summarization. It does **not** teach paraphrase.
For that, run the teacher pass and merge its abstractive digests in; the
intended path is selection first, rewording second.

Three failure modes the generator guards against, because the first draft hit
all of them:

- **Dangling pronouns.** "It is a Kwa language spoken by 800,000 people"
  is a fine sentence and a false summary — lifted out of context the pronoun
  reattaches to whatever precedes it. A pronoun-subject sentence is only kept
  when its antecedent is immediately before it in the summary.
- **Topic drift.** A passage about Guembeul Natural Reserve mentions Saloum
  Delta National Park, and a centrality score will happily pick that sentence.
  Candidates must name the subject or continue a sentence already chosen.
- **Broken brackets.** Splitting on full stops inside "(French: Réserve
  spéciale …)" leaves a fragment opening on a stray bracket. Sentences are
  re-joined until brackets balance.

## Running it

```bash
./run_all.sh  WIP_West_African_Tourist_Locations_-_West_Africa_Spots.csv  out
```

Or stage by stage:

```bash
python3 ijeoma_clean.py  --in raw.csv --out out        # 1  clean + normalise
python3 ijeoma_pairs.py  --out out                     # 2  deterministic pairs
python3 ijeoma_teacher.py --out out --limit 5          # 2b smoke test (costs ~nothing)
python3 ijeoma_teacher.py --out out                    # 2b full teacher pass
python3 ijeoma_build.py  --out out                     # 3  validate + split
python3 ijeoma_summarize.py --out out                  # 4  summarization pairs
python3 ijeoma_build_summarization.py --out out        #    validate + split those
python3 test_validator.py                              #    validator self-test
```

## Files

| File | What it is |
| --- | --- |
| `ijeoma_clean.py` | Cleaning, non-place filtering, label normalisation |
| `ijeoma_pairs.py` | Deterministic pair generation |
| `ijeoma_teacher.py` | Teacher pass — digests and abstractive QA (needs a key) |
| `ijeoma_build.py` | Validator, passage-level split, training files |
| `ijeoma_summarize.py` | Extractive summarization pair generation |
| `ijeoma_build_summarization.py` | Validates and splits it against the same passage split |
| `test_validator.py` | Nine deliberately broken pairs the validator must catch |
| `make_reference_digests.py` | Four hand-written digests: plumbing test and quality yardstick |
| `run_all.sh` | All of the above in order |

Outputs land in `out/`:

- `conversational_training_data.jsonl` / `conversational_test_data.jsonl` —
  `{"input", "output"}` only, the lab's filenames
- `pairs_train_full.jsonl` / `pairs_test_full.jsonl` — same pairs plus
  `task`, `subtype`, `passage_id`, `country`, `name`, for per-task and
  per-country evaluation
- `passages.jsonl` — the cleaned passage table, one row per destination
- `rejected.csv`, `review.csv` — see "Needs your eyes" below
- `*_report.json` — counts for the data card

## Design decisions worth knowing

**Every task is passage-grounded.** The passage sits in the input; the output
is derivable from it. Nothing asks the model to recall a place from memory,
because a 1B model cannot and will invent instead. This is what makes the
90% claim-support target reachable.

**The split is by passage, not by pair.** `train_test_split(df)` on the pair
table would put a digest of Osun-Osogbo in train and a question about
Osun-Osogbo in test. `ijeoma_build.py` splits unique `passage_id`s,
stratified by country, then expands — and asserts zero overlap.

**Nigeria is capped at 22% of each slice.** It is 46% of the raw corpus.
Without the quota every metric would be a Nigeria metric.

**Live logistics never reach an output.** Phone numbers, prices, opening
hours and date stamps are stripped from passages, and the validator rejects
any output containing them.

**Abstentions are verified, not assumed.** A "the passage doesn't say" pair
is only emitted after checking the passage really lacks that information.

**Passage ids are content-derived hashes.** They were positional at first,
which meant any change to the cleaning rules renumbered every passage and
silently invalidated an existing train/test split. They are now a hash of
name, country and url, so the split survives a re-clean.

**Terrain is null 37% of the time.** Inferring one from a single incidental
word would be a guess. The model is trained to return `null` rather than
invent a landscape.

## Label taxonomy

13 themes: Sacred & Ceremonial · Palace & Royal Site · Fort & Castle ·
Museum & Gallery · Archaeological Site · Monument & Memorial · Coast & Beach ·
Nature & Wildlife · Landscape & Scenery · Historic Site · Town or Area Guide ·
Family & Leisure · General Attraction.

`world_heritage` is a separate boolean, not a theme — it cuts across all of
them (77 passages).

9 terrains: Coastal · Island · Riverine · Lake or wetland · Forest ·
Mountain or highland · Desert or Sahel · Savannah · Urban.

Each passage records `theme_source`: `type` (1,403 — from the source's own
tag, reliable), `keyword` (123 — inferred from the text), `fallback` (66 —
nothing matched). Answers only assert a theme when the source is `type`.

## Needs your eyes

**`out/rejected.csv`** — the non-place filter is deliberately conservative,
but it still caught about five real places whose scraped text is a biography
of the person they are named after: **Kalakuta Republic**, **Villa Karo**,
**Boubou Hama National Museum**, **Emotan Statue**, **CERCOPAN**. Restore
them with better source text if you want them in. Ten minutes.

**`out/review.csv`** (30 rows) — kept but flagged: mostly town guides whose
history sections read like biographies, plus a few monuments named after
people. Skim, don't agonise.

## Before the teacher run

Read `make_reference_digests.py`. Those four digests are what good output
looks like. Run `ijeoma_teacher.py --limit 5` first, put its output next to
them, and only launch the full pass once the quality matches — a bad prompt
discovered after 1,700 calls is a wasted budget.

Record in the data card: the model name, the date, the prompt, and the
rejection rate from `out/teacher_report.json`.
