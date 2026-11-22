# Run 2 — full training run (22 Nov 2026)

Gemma 3 1B-it · LoRA rank 8 · lr 2e-4 · 1 epoch · 6,522 training examples ·
sequence length 960 · final loss 0.027 / val loss 0.028 · T4, Colab.
Evaluated on 100 held-out items (passages never seen in training).

## Before vs after training

| Metric | Base | LoRA | n |
|---|---|---|---|
| **Claims supported by source (target ≥ 0.90)** | 0.860 | **0.974** | 67 items (399 / 268 claims) |
| Judge: factual accuracy (1–5) | 3.81 | 4.60 | 67 |
| Judge: cultural nuance (1–5) | 4.30 | 4.61 | 67 |
| Judge: helpfulness (1–5) | 3.81 | 3.76 | 67 |
| Refusal recall | 0.44 | 0.92 | 25 |
| False refusal | 0.03 | 0.03 | answerable items |
| Grounded (names/numbers in passage) | 0.82 | 1.00 | 67 |
| Summary grounded | 0.62 | 1.00 | 26 |
| ROUGE-L | 0.34 | 0.70 | 67 |
| ROUGE-1 | 0.40 | 0.72 | 67 |
| BERTScore F1 (distilbert, CPU) | 0.84 | 0.92 | 67 |
| Classify: country | 0.75 | 1.00 | **8** |
| Classify: theme | 0.125 | 0.75 | **8** |
| Classify: terrain | 0.375 | 0.75 | **8** |
| Classify: traveller profile | 0.125 | 0.625 | **8** |
| Theme macro-F1 / P / R | 0.10 / 0.10 / 0.10 | 0.48 / 0.47 / 0.50 | **8** |
| Invention rate (no passage, 20 prompts) | 0.95 (run 1) | 0.65 | 20 |

## Caveats
- Classification rests on 8 items and digests on 2: directional only.
- The judge (gpt-5.4-mini) is from the same model family as the teacher that
  wrote much of the training data; it may favour that style.
- 20 of 67 tuned outputs reproduce the reference word for word (summaries 38%,
  answers 26%). Many references are extractive, so ROUGE partly measures copying.
- Helpfulness did not improve: answers are faithful but often minimal or
  off-target (e.g. reciting area statistics when asked about history).
- Without a passage the model still invents 65% of the time, including about
  places that do not exist ("Zarenga Cliff Tombs… in the Philippines").

## Unsupported claims (tuned, 7 of 268)
- Kagara: claimed the passage had no industrial interest; it mentions a talc
  processing plant (a wrong refusal).
- Labelled a theme "historical site & memorial" (not in the passage).
- Djenné: "centre for Fulani and Dogon culture", "Macina inland delta".
- Ouahigouya: claimed no guides/tours information.
- A town "market, lorry station, and historic sites".
- "Obetsebi is in Ghana".

Files: `eval_*_with_input.csv` (predictions + prompts), `*_judged.csv`
(per-item judge scores), `eval_adversarial.csv`, `ijeoma_experiment_log.csv`.
Judge script: `evaluation/run_judge.py`.
