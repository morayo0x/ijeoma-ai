# Ijeoma AI — notes for Claude

Capstone: Gemma 3 1B + LoRA as a grounded West African travel assistant. Unrelated to Quanta AI.

- **Every decision goes in `docs/DECISIONS.md`.** Add a new numbered entry
  (template at the bottom of the file) whenever data, training, evaluation or
  scope changes. Never rewrite old entries: supersede them.
- Every training run is appended to `results/ijeoma_experiment_log.csv` by the
  notebook; copy the latest log and eval CSVs from Drive into `results/`.
- Training runs in Colab (T4); this repo holds the pipeline, data and notebook.
- The refusal regex in `pipeline/ijeoma_build.py` and in the notebook must stay identical.
