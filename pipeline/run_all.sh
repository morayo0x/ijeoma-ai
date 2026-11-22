#!/usr/bin/env bash
# Ijeoma AI - full dataset build.
#
# Stages 1, 2 and 3 are deterministic and free; run them as often as you like.
# Stage 2b is the only one that costs anything, and it caches, so an
# interrupted run resumes without re-spending.
set -euo pipefail

RAW="${1:-WIP_West_African_Tourist_Locations_-_West_Africa_Spots.csv}"
OUT="${2:-out}"

echo "== 1. clean, filter, normalise labels"
python3 ijeoma_clean.py --in "$RAW" --out "$OUT"

echo
echo "== 2. deterministic pairs (classification, grounded QA, abstention, refusal)"
python3 ijeoma_pairs.py --passages "$OUT/passages.jsonl" --out "$OUT"

echo
echo "== 2b. teacher pairs (digests + abstractive QA)"
echo "   skipped unless GOOGLE_API_KEY is set; run it yourself with:"
echo "   python3 ijeoma_teacher.py --passages $OUT/passages.jsonl --out $OUT --limit 5   # smoke test"
echo "   python3 ijeoma_teacher.py --passages $OUT/passages.jsonl --out $OUT             # full pass"
if [ -n "${GOOGLE_API_KEY:-}" ]; then
  python3 ijeoma_teacher.py --passages "$OUT/passages.jsonl" --out "$OUT"
fi

echo
echo "== 3. validate, split by passage, write training files"
python3 ijeoma_build.py --out "$OUT" --teacher "$OUT/pairs_teacher.jsonl"

echo
echo "== 4. summarization dataset (extractive, no API calls)"
python3 ijeoma_summarize.py --passages "$OUT/passages.jsonl" --out "$OUT"
python3 ijeoma_build_summarization.py --out "$OUT"

echo
echo "== validator self-test"
python3 test_validator.py

echo
echo "Training files:"
echo "  $OUT/conversational_training_data.jsonl   <- the lab's Activity 4 filename"
echo "  $OUT/conversational_test_data.jsonl"
echo "  $OUT/pairs_train_full.jsonl               <- same pairs plus metadata for eval slicing"
echo "  $OUT/pairs_test_full.jsonl"
echo "  $OUT/summarization_training_data.jsonl  / summarization_test_data.jsonl"
echo "  $OUT/summarization_train_full.jsonl     / summarization_test_full.jsonl"
