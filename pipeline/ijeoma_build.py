"""
Ijeoma AI - Stage 3: validate, split and emit the training files.

Merges the deterministic pairs with any teacher-generated pairs, runs every
pair through the validator, splits by PASSAGE (not by pair) stratified by
country, and writes the two JSONL files the course lab expects.

The passage-level split is the point of this script. Splitting the pair
table directly - as `train_test_split(df)` in the lab notebook does - puts
a digest of Osun-Osogbo in train and a question about Osun-Osogbo in test,
and the held-out score stops meaning anything.

Usage:
    python3 ijeoma_build.py --out out [--teacher out/pairs_teacher.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------

RE_PHONE = re.compile(r"☏|\+\d{2,3}[\s\-]?\d[\d\s\-]{5,}")
RE_URL = re.compile(r"https?://|www\.")
RE_EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
RE_MONEY = re.compile(r"(?:₦|US\$|\$|€|£|CFA|GH₵|XOF|NGN|GHS)\s?\d")
RE_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b", re.I)
RE_NUM = re.compile(r"\d[\d,.]*")
RE_PROPER = re.compile(r"\b[A-Z][a-zÀ-ɏ'’\-]{2,}\b")

LENGTH_BOUNDS = {  # words
    "classify": (5, 120),
    "answer": (3, 90),
    "digest": (40, 140),
    "summarize": (8, 110),  # spans one-liners through three-sentence summaries
    "refuse": (8, 90),
}

# Tasks whose output is free text grounded in the passage, so the number and
# entity checks apply.
GROUNDED_TASKS = ("answer", "digest", "summarize")

CLASSIFY_KEYS = {"country", "theme", "terrain", "traveler_profile", "world_heritage"}

# Words that routinely open a sentence or name a concept and so are not
# evidence that the model invented an entity.
STOPWORD_PROPER = {
    "The",
    "This",
    "That",
    "It",
    "They",
    "There",
    "A",
    "An",
    "In",
    "On",
    "At",
    "If",
    "When",
    "What",
    "Which",
    "Where",
    "How",
    "You",
    "Your",
    "I",
    "We",
    "Check",
    "Hours",
    "Passage",
    "Its",
    "His",
    "Her",
    "African",
    "West",
    "Africa",
    "None",
    "True",
    "False",
    "Nature",
    "Wildlife",
    "Culture",
    "Museum",
    "Gallery",
    "Historic",
    "Site",
    "Monument",
    "Memorial",
    "Coast",
    "Beach",
    "Landscape",
    "Scenery",
    "Fort",
    "Castle",
    "Palace",
    "Royal",
    "Archaeological",
    "Town",
    "Area",
    "Guide",
    "General",
    "Attraction",
    "Family",
    "Leisure",
    "Heritage",
    "World",
}


def normalise_numbers(text: str) -> set[str]:
    out = set()
    for n in RE_NUM.findall(text):
        n = n.rstrip(".,").replace(",", "")
        if n:
            out.add(n)
            out.add(n.lstrip("0") or "0")
    return out


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_passage(inp: str) -> str:
    idx = inp.find("PASSAGE:")
    return inp[idx + len("PASSAGE:") :].strip() if idx >= 0 else ""


def validate(pair: dict) -> list[str]:
    """Return a list of failure codes; empty means the pair is usable."""
    fails: list[str] = []
    task = pair.get("task", "")
    out = str(pair.get("output", "")).strip()
    passage = extract_passage(str(pair.get("input", "")))
    has_passage = bool(passage) and passage != "(none available)"

    if not out:
        return ["empty_output"]

    # 1 - structure
    if task == "classify":
        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            fails.append("json_parse")
        else:
            if set(obj) != CLASSIFY_KEYS:
                fails.append("json_keys")

    # 2 - length
    lo, hi = LENGTH_BOUNDS.get(task, (3, 200))
    n_words = len(out.split())
    if n_words < lo or n_words > hi:
        fails.append("length")

    # 3 - live-logistics patterns must never appear in an output
    if RE_PHONE.search(out) or RE_EMAIL.search(out) or RE_URL.search(out):
        fails.append("contact_details")
    if RE_MONEY.search(out) or RE_TIME.search(out):
        fails.append("live_logistics")

    # 4 - every number in the output must be in the passage.
    # Normalise first: a trailing full stop and a thousands separator are
    # punctuation, not a different number ("1992." and "8,000" must match
    # "1992" and "8000" in the passage).
    if has_passage and task in GROUNDED_TASKS:
        p_nums = normalise_numbers(passage)
        if not normalise_numbers(out) <= p_nums:
            fails.append("number_not_in_passage")

    # 5 - proper nouns must be grounded in the passage
    if has_passage and task in GROUNDED_TASKS:
        p_lower = passage.lower()
        unseen = [
            w
            for w in RE_PROPER.findall(out)
            if w not in STOPWORD_PROPER and w.lower() not in p_lower
        ]
        if len(unseen) > 1:
            fails.append("entity_not_in_passage")

    # 6 - teacher pairs: the label must agree with what the output does.
    # The teacher sometimes marks an answerable question as unanswerable (or
    # refuses mid-dialogue); either way the pair teaches the wrong behaviour.
    if str(pair.get("subtype", "")).startswith(("teacher", "dialogue")):
        refuses = bool(RE_REFUSAL.search(out))
        if task == "refuse" and not refuses:
            fails.append("refusal_label_mismatch")
        if task in GROUNDED_TASKS and refuses:
            fails.append("refusal_label_mismatch")
        words = re.findall(r"[a-z]+", out.lower())
        if words and sum(w in ENGLISH_MARKERS for w in words) / len(words) < 0.08:
            fails.append("not_english")

    return fails


# Same pattern the notebook scores refusals with - keep the two in sync.
RE_REFUSAL = re.compile(
    r"((?:does|do)(?:\s+not|n't)\s+"
    r"(?:say|mention|cover|provide|describe|give|list|include|state|have)"
    r"|i (?:cannot|can't|am not able|do not have|don't have)"
    r"|cannot advise|can't advise|unable to|no information"
    r"|not covered|not mentioned|not stated"
    r"|source does not|passage does not)",
    re.I,
)

ENGLISH_MARKERS = {
    "the",
    "a",
    "an",
    "is",
    "and",
    "of",
    "in",
    "it",
    "to",
    "was",
    "are",
    "for",
    "on",
    "with",
    "as",
    "by",
    "that",
    "at",
}


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def split_passages(pairs, test_frac, seed):
    """Hold out whole passages, stratified by country."""
    rng = random.Random(seed)
    by_country: dict[str, set] = defaultdict(set)
    for p in pairs:
        if p.get("passage_id"):
            by_country[str(p.get("country"))].add(p["passage_id"])

    test_ids: set[str] = set()
    for ids in by_country.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        k = max(1, round(len(ids) * test_frac)) if len(ids) >= 3 else 0
        test_ids.update(ids[:k])
    return test_ids


def run(out_dir, teacher_path, test_frac, seed):
    det_path = os.path.join(out_dir, "pairs_deterministic.jsonl")
    pairs = read_jsonl(det_path)
    n_det = len(pairs)

    n_teacher = 0
    if teacher_path and os.path.exists(teacher_path):
        teach = read_jsonl(teacher_path)
        pairs += teach
        n_teacher = len(teach)

    # ---- validate
    fail_counts: Counter = Counter()
    kept, rejected = [], []
    for p in pairs:
        fails = validate(p)
        if fails:
            fail_counts.update(fails)
            p["_fails"] = fails
            rejected.append(p)
        else:
            kept.append(p)

    # ---- near-duplicate outputs (teacher models converge on phrasing)
    seen: dict[str, int] = {}
    deduped = []
    dup = 0
    for p in kept:
        # Classification outputs are label sets: two different places sharing
        # a label set is correct, not a duplicate. Dedup on the passage only.
        if p.get("task") == "classify":
            key = f"classify|{p.get('passage_id')}"
            if key in seen:
                dup += 1
                continue
            seen[key] = 1
            deduped.append(p)
            continue
        key = re.sub(r"\W+", " ", str(p["output"]).lower()).strip()[:160]
        if p.get("task") == "refuse":
            key = f"{p.get('subtype')}|{key}"  # refusals are meant to repeat
            if seen.get(key, 0) >= 40:
                dup += 1
                continue
        elif key in seen:
            dup += 1
            continue
        seen[key] = seen.get(key, 0) + 1
        deduped.append(p)

    # ---- split
    test_ids = split_passages(deduped, test_frac, seed)
    rng = random.Random(seed + 1)
    train, test = [], []
    for p in deduped:
        pid = p.get("passage_id")
        if pid is None:  # scope refusals carry no passage
            (test if rng.random() < test_frac else train).append(p)
        else:
            (test if pid in test_ids else train).append(p)

    rng.shuffle(train)
    rng.shuffle(test)

    # ---- write
    def dump(rows, name, lab_format):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                rec = {"input": r["input"], "output": r["output"]} if lab_format else r
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    dump(train, "conversational_training_data.jsonl", True)
    dump(test, "conversational_test_data.jsonl", True)
    dump(train, "pairs_train_full.jsonl", False)
    dump(test, "pairs_test_full.jsonl", False)
    dump(rejected, "pairs_rejected.jsonl", False)

    # ---- leakage assertion
    train_ids = {p["passage_id"] for p in train if p.get("passage_id")}
    test_pids = {p["passage_id"] for p in test if p.get("passage_id")}
    overlap = train_ids & test_pids
    assert not overlap, f"passage leakage between splits: {len(overlap)}"

    report = {
        "pairs_in": {"deterministic": n_det, "teacher": n_teacher},
        "validation_failures": dict(fail_counts),
        "rejected": len(rejected),
        "near_duplicates_dropped": dup,
        "train": len(train),
        "test": len(test),
        "test_fraction_actual": round(len(test) / max(1, len(train) + len(test)), 3),
        "train_by_task": dict(Counter(p.get("task") for p in train)),
        "test_by_task": dict(Counter(p.get("task") for p in test)),
        "test_by_country": dict(Counter(str(p.get("country")) for p in test).most_common()),
        "passages_train": len(train_ids),
        "passages_test": len(test_pids),
        "passage_leakage": len(overlap),
    }
    with open(os.path.join(out_dir, "build_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--teacher", default="out/pairs_teacher.jsonl")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args()
    print(json.dumps(run(a.out, a.teacher, a.test_frac, a.seed), indent=2, ensure_ascii=False))
