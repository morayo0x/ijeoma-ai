# Ijeoma AI

A West African travel discovery assistant: Gemma 3 1B fine-tuned with LoRA to
classify destinations, write traveller digests, and answer questions about them
from a source passage, declining when the passage does not say.

Capstone project for the Google DeepMind / FATE Foundation AI programme.
The proposal is in `docs/`.

## Layout

```
docs/        proposal, DECISIONS.md (decision log)
pipeline/    data build: clean -> template pairs -> teacher pairs -> split
  out/       built datasets (upload these to Drive for the notebook)
notebooks/   train-and-evaluate notebook (Colab, T4, Keras + JAX)
results/     eval CSVs and the experiment log from each run
data/raw/    original dataset bundles from Drive
```

## Rebuild the data

```bash
cd pipeline
python3 ijeoma_pairs.py --passages out/passages.jsonl --out out
OPENAI_API_KEY=... python3 ijeoma_teacher.py --provider openai --passages out/passages.jsonl --out out \
    --n-digest 600 --n-qa 400 --n-dialogue 400
python3 ijeoma_build.py --out out --teacher out/pairs_teacher.jsonl
python3 ijeoma_summarize.py --passages out/passages.jsonl --out out
python3 ijeoma_build_summarization.py --out out
```

The teacher step caches per passage (`out/teacher_cache.jsonl`), so a rerun
does not re-spend.

## Design decision: passage in the prompt

The proposal planned pure fine-tuning with no retrieval. Run 1 showed why that
does not hold at 1B parameters: asked about places with no passage, the model
invented an answer 95% of the time. Every task therefore takes its source
passage as input, and the model is trained to read rather than recall.

## Changes in v0.2

- Classify prompts now list the allowed labels. Run 1 scored 0.000 on theme
  because the model invented label wording that exact-match scoring rejected.
- Teacher step runs on OpenAI (`--provider openai`) and adds 3-turn
  traveller-guide dialogues, each ending in an out-of-scope question.
- Notebook v0.2: full-data run (1 epoch, adapter checkpointed to Drive every 500 steps, 100 eval items) and the
  proposal's remaining metrics: per-class P/R/F1, ROUGE-1, BERTScore and an
  LLM judge (1-5 on factual accuracy, cultural nuance, helpfulness).

## Licensing

Source text is from Wikipedia and Wikivoyage (CC BY-SA 4.0) and OpenStreetMap
(ODbL). Model use is governed by the Gemma Terms of Use.
