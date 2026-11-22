# Ijeoma AI — notes for Claude

Capstone: Gemma 3 1B + LoRA as a grounded West African travel assistant. Unrelated to Quanta AI.

- **Every decision goes in `docs/DECISIONS.md`.** Add a new numbered entry
  (template in the file) whenever data, training, evaluation or scope changes.
  Never rewrite old entries: supersede them.
- Each training run gets its own folder under `results/` (eval CSVs, experiment
  log, RESULTS.md). The notebook writes these to the `Ijeoma` Drive folder.
- Training runs in Colab (T4); this repo holds the pipeline, data, adapter and notebook.
- The refusal regex in `pipeline/ijeoma_build.py`, the notebook and
  `evaluation/run_judge.py` must stay identical.
