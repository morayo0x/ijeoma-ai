"""
Ijeoma AI - validate and split the summarization dataset.

Splits consistently with the main instruction dataset. That consistency is
the whole point of this script: if a passage sits in the QA training set and
its summary sits in the summarization test set, then training on both and
evaluating on either gives a number inflated by a passage the model has
already read. The split here inherits the existing assignment for every
passage the main build already placed, and only decides the leftovers.

Usage:
    python3 ijeoma_build_summarization.py --out out
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ijeoma_build import validate  # noqa: E402


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def inherited_sides(out_dir):
    """Map passage_id -> 'train' | 'test' from the main build, if present."""
    side = {}
    for name, label in (("pairs_train_full.jsonl", "train"),
                        ("pairs_test_full.jsonl", "test")):
        for r in read_jsonl(os.path.join(out_dir, name)):
            pid = r.get("passage_id")
            if pid:
                side[pid] = label
    return side


def run(out_dir, test_frac, seed):
    pairs = read_jsonl(os.path.join(out_dir, "pairs_summarization.jsonl"))
    if not pairs:
        raise SystemExit("run ijeoma_summarize.py first")

    # ---- validate
    fails = Counter()
    kept, rejected = [], []
    for p in pairs:
        f = validate(p)
        if f:
            fails.update(f)
            p["_fails"] = f
            rejected.append(p)
        else:
            kept.append(p)

    # ---- drop near-duplicate outputs
    seen, deduped, dup = set(), [], 0
    for p in kept:
        key = re.sub(r"\W+", " ", p["output"].lower()).strip()[:160]
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        deduped.append(p)

    # ---- split, inheriting the main build's assignment where it exists
    side = inherited_sides(out_dir)
    inherited = sum(1 for p in deduped if p["passage_id"] in side)

    by_country = defaultdict(set)
    for p in deduped:
        if p["passage_id"] not in side:
            by_country[str(p["country"])].add(p["passage_id"])

    rng = random.Random(seed)
    for country, pids in by_country.items():
        pids = sorted(pids)
        rng.shuffle(pids)
        k = max(1, round(len(pids) * test_frac)) if len(pids) >= 3 else 0
        for i, pid in enumerate(pids):
            side[pid] = "test" if i < k else "train"

    train = [p for p in deduped if side.get(p["passage_id"]) != "test"]
    test = [p for p in deduped if side.get(p["passage_id"]) == "test"]
    rng.shuffle(train)
    rng.shuffle(test)

    # ---- leakage checks, within this set and against the main one
    tr_pids = {p["passage_id"] for p in train}
    te_pids = {p["passage_id"] for p in test}
    assert not (tr_pids & te_pids), "passage in both summarization splits"

    main_train = {r["passage_id"] for r in
                  read_jsonl(os.path.join(out_dir, "pairs_train_full.jsonl"))
                  if r.get("passage_id")}
    main_test = {r["passage_id"] for r in
                 read_jsonl(os.path.join(out_dir, "pairs_test_full.jsonl"))
                 if r.get("passage_id")}
    cross_a = te_pids & main_train      # summarise-test seen in QA-train
    cross_b = tr_pids & main_test       # QA-test seen in summarise-train
    assert not cross_a, f"{len(cross_a)} test passages are in the main train split"
    assert not cross_b, f"{len(cross_b)} train passages are in the main test split"

    # ---- write
    def dump(rows, name, lab_format):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for r in rows:
                rec = ({"input": r["input"], "output": r["output"]}
                       if lab_format else r)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dump(train, "summarization_training_data.jsonl", True)
    dump(test, "summarization_test_data.jsonl", True)
    dump(train, "summarization_train_full.jsonl", False)
    dump(test, "summarization_test_full.jsonl", False)
    if rejected:
        dump(rejected, "summarization_rejected.jsonl", False)

    report = {
        "pairs_in": len(pairs),
        "validation_failures": dict(fails),
        "rejected": len(rejected),
        "near_duplicates_dropped": dup,
        "train": len(train),
        "test": len(test),
        "test_fraction_actual": round(len(test) / max(1, len(train) + len(test)), 3),
        "passages_inherited_from_main_split": inherited,
        "passages_newly_assigned": len(deduped) - inherited,
        "train_by_variant": dict(Counter(p["subtype"] for p in train)),
        "test_by_variant": dict(Counter(p["subtype"] for p in test)),
        "cross_dataset_leakage": 0,
    }
    with open(os.path.join(out_dir, "summarization_build_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args()
    print(json.dumps(run(a.out, a.test_frac, a.seed), indent=2, ensure_ascii=False))
