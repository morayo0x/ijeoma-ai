"""
Ijeoma AI - Stage 2: deterministic instruction-pair generation.

Builds every pair whose output can be derived from the passage or its
metadata without calling a model: classification, extractive grounded QA,
abstention and scope refusal. Abstractive digests come from the separate
teacher pass (ijeoma_teacher.py) and are merged in by Stage 3.

Every output here is faithful by construction - it is either a label taken
from a normalised field, a span lifted verbatim from the passage, or a
refusal whose correctness was checked against the passage first.

Usage:
    python3 ijeoma_pairs.py --passages out/passages.jsonl --out out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

import pandas as pd

TASK_TAGS = {
    "classify": "[CLASSIFY]",
    "answer": "[ANSWER]",
    "digest": "[DIGEST]",
    "refuse": "[ANSWER]",  # refusals arrive as ordinary questions
}


def stable_choice(seq, *key_parts):
    """Deterministic pick so re-runs produce identical data."""
    h = hashlib.sha1("|".join(map(str, key_parts)).encode()).hexdigest()
    return seq[int(h[:8], 16) % len(seq)]


# --------------------------------------------------------------------------
# Task A - classification
# --------------------------------------------------------------------------

CLASSIFY_INSTRUCTIONS = [
    "Classify this destination. Return JSON with country, theme, terrain, "
    "traveler_profile and world_heritage. Use only what the passage supports.",
    "Read the passage and label the destination. Reply with JSON containing "
    "country, theme, terrain, traveler_profile and world_heritage.",
    "Tag this place for a travel catalogue. Output JSON: country, theme, "
    "terrain, traveler_profile, world_heritage.",
    "From the passage below, produce the destination's catalogue labels as "
    "JSON (country, theme, terrain, traveler_profile, world_heritage).",
]


# Closed label sets. The prompt must list them: without the menu the model
# invents its own wording ("Nature" for "Nature & Wildlife") and exact-match
# scoring reads that as a miss - the cause of theme accuracy 0.000 in run 1.
THEMES = [
    "Town or Area Guide",
    "Nature & Wildlife",
    "Museum & Gallery",
    "Archaeological Site",
    "Fort & Castle",
    "Monument & Memorial",
    "General Attraction",
    "Landscape & Scenery",
    "Sacred & Ceremonial",
    "Coast & Beach",
    "Historic Site",
    "Palace & Royal Site",
    "Family & Leisure",
]
TERRAINS = [
    "Coastal",
    "Riverine",
    "Island",
    "Forest",
    "Mountain or highland",
    "Lake or wetland",
    "Desert or Sahel",
    "Savannah",
    "Urban",
]
PROFILES = [
    "General, base for exploring",
    "Nature and wildlife, outdoor",
    "Culture-focused, indoor",
    "History and heritage, slow travel",
    "History and heritage",
    "General sightseeing",
    "Short visit, city sightseeing",
    "Scenic, outdoor, active",
    "Culture-focused, slow travel",
    "Beach and relaxation",
    "Culture-focused, heritage",
    "Family-friendly, leisure",
]

LABEL_MENU = (
    "Allowed values (copy one exactly):\n"
    f"theme: {' | '.join(THEMES)}\n"
    f"terrain: {' | '.join(TERRAINS)} | null\n"
    f"traveler_profile: {' | '.join(PROFILES)}\n"
    "world_heritage: true | false"
)


def build_classify(row) -> dict:
    labels = {
        "country": row["country"],
        "theme": row["theme"],
        "traveler_profile": row["traveler_profile"],
        "world_heritage": bool(row["world_heritage"]),
    }
    if isinstance(row["terrain"], str) and row["terrain"]:
        labels["terrain"] = row["terrain"]
    else:
        labels["terrain"] = None
    ordered = {
        k: labels[k] for k in ("country", "theme", "terrain", "traveler_profile", "world_heritage")
    }
    instr = stable_choice(CLASSIFY_INSTRUCTIONS, row["passage_id"], "classify")
    return {
        "input": (f"{TASK_TAGS['classify']} {instr}\n{LABEL_MENU}\n\nPASSAGE:\n{row['passage']}"),
        "output": json.dumps(ordered, ensure_ascii=False, indent=2),
        "task": "classify",
        "passage_id": row["passage_id"],
        "country": row["country"],
        "name": row["name"],
    }


# --------------------------------------------------------------------------
# Task B - extractive grounded QA
# --------------------------------------------------------------------------

Q_WHAT_IS = [
    "What is {name}?",
    "In one sentence, what is {name}?",
    "Give me a one-line description of {name}.",
    "What kind of place is {name}?",
]
Q_COUNTRY = [
    "Which country is {name} in?",
    "Where is {name} located, country-wise?",
    "What country would I travel to for {name}?",
]
Q_THEME = [
    "What sort of attraction is {name} - what would I be going to see?",
    "If I am building an itinerary, what category does {name} fall under?",
    "Is {name} more of a nature stop or a cultural one?",
]

ANSWER_INSTRUCTION = (
    "Answer the question using only the passage below. If the passage does "
    "not contain the answer, say so plainly."
)


RE_NOT_A_DEFINITION = re.compile(
    r"(may refer to|is also (the|an?) |abbreviation for|disambiguation|"
    r"for other uses|not to be confused|see also)",
    re.I,
)
RE_DEFINITION = re.compile(
    r"\b(?:is|was|are|were)\s+(?:a|an|the|one of)\b|\bis\s+(?:located|situated|found)\b", re.I
)


def first_sentence(text: str, name: str = "") -> str:
    """The first sentence that actually defines the subject.

    Several scraped passages open with a disambiguation line ('"WA" is also
    the abbreviation for Washington'). Taking sentence one blindly turns
    those into training targets, so prefer a sentence that names the subject
    and reads as a definition.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip())]
    parts = [
        p
        for p in parts
        if 40 <= len(p) <= 320
        and not RE_NOT_A_DEFINITION.search(p)
        and not p.lstrip().startswith(('"', "'", "“"))
    ]
    if not parts:
        return ""
    key = str(name).strip().lower()
    if key:
        for p in parts:
            if key in p.lower() and RE_DEFINITION.search(p):
                return p
        for p in parts:
            if key in p.lower():
                return p
    for p in parts:
        if RE_DEFINITION.search(p):
            return p
    return parts[0]


def qa_prompt(question: str, passage: str) -> str:
    return (
        f"{TASK_TAGS['answer']} {ANSWER_INSTRUCTION}\n\nQUESTION: {question}\n\nPASSAGE:\n{passage}"
    )


def build_fact_qa(row) -> dict | None:
    name, passage = row["name"], row["passage"]
    kind = stable_choice(["what_is", "country", "theme"], row["passage_id"], "qa")

    if kind == "country":
        country = str(row["country"]).split(";")[0].strip()
        # Only ask it when the passage itself names the country.
        if country.lower() not in passage.lower():
            kind = "what_is"
        else:
            q = stable_choice(Q_COUNTRY, row["passage_id"], "q").format(name=name)
            return {
                "input": qa_prompt(q, passage),
                "output": f"{name} is in {country}.",
                "task": "answer",
                "subtype": "country",
                "passage_id": row["passage_id"],
                "country": row["country"],
                "name": name,
            }

    # Only claim a theme when it came from an explicit source type, not from a
    # keyword guess - otherwise the answer asserts more certainty than the
    # pipeline has.
    if kind == "theme" and row.get("theme_source") != "type":
        kind = "what_is"
    if kind == "theme":
        sent = first_sentence(passage, name)
        if sent:
            theme = str(row["theme"]).lower()
            q = stable_choice(Q_THEME, row["passage_id"], "q").format(name=name)
            return {
                "input": qa_prompt(q, passage),
                "output": f"It falls under {theme}. The passage describes it this way: {sent}",
                "task": "answer",
                "subtype": "theme",
                "passage_id": row["passage_id"],
                "country": row["country"],
                "name": name,
            }
        kind = "what_is"

    sent = first_sentence(passage, name)
    if not sent:
        return None
    q = stable_choice(Q_WHAT_IS, row["passage_id"], "q").format(name=name)
    return {
        "input": qa_prompt(q, passage),
        "output": sent,
        "task": "answer",
        "subtype": "what_is",
        "passage_id": row["passage_id"],
        "country": row["country"],
        "name": name,
    }


# --------------------------------------------------------------------------
# Task C - abstention (the passage genuinely does not say)
# --------------------------------------------------------------------------

ABSTAIN_PROBES = [
    {
        "key": "price",
        "questions": [
            "How much does it cost to get into {name}?",
            "What is the entry fee for {name}?",
            "Is there an admission charge at {name}?",
        ],
        # Only usable when the passage has no pricing language at all.
        "absent_if_no": r"(fee|admission|ticket|entrance charge|free of charge|"
        r"costs?\s+\d|₦|US\$|\$|€|£|CFA|GH₵)",
        "answer": "The passage does not say what it costs to visit {name}. "
        "It covers what the place is, not its fees - you would need "
        "to check with the site or a local operator for current prices.",
    },
    {
        "key": "hours",
        "questions": [
            "What time does {name} open?",
            "What are the opening hours at {name}?",
            "Is {name} open on Sundays?",
        ],
        "absent_if_no": r"(open(s|ing|ed)?\s+(at|from|daily|hours)|closing time|"
        r"\d{1,2}\s*(am|pm)|opening hours)",
        "answer": "The passage does not give opening hours for {name}. "
        "Hours change and are not covered by this source.",
    },
    {
        "key": "access",
        "questions": [
            "What is the nearest airport to {name}?",
            "How do I get to {name} by road?",
            "Which airline flies closest to {name}?",
        ],
        "absent_if_no": r"(airport|airstrip|highway|expressway|by road|by bus|"
        r"railway|train station|ferry|drive from)",
        "answer": "The passage does not describe how to reach {name}. "
        "It covers the place itself rather than routes or transport.",
    },
    {
        "key": "safety",
        "questions": [
            "Is it safe to travel to {name} right now?",
            "Are there any security advisories for {name}?",
            "Do I need a vaccination certificate to visit {name}?",
        ],
        "absent_if_no": None,  # always abstain, this is live information
        "answer": "I cannot advise on current safety, security or health "
        "requirements for {name}. That information changes constantly "
        "and this source does not cover it - check your government's "
        "official travel advice and the relevant national authority.",
    },
]


def build_abstention(row, probe) -> dict | None:
    passage, name = row["passage"], row["name"]
    pat = probe["absent_if_no"]
    if pat and re.search(pat, passage, re.I):
        return None  # passage may answer it - not a valid abstention
    q = stable_choice(probe["questions"], row["passage_id"], probe["key"]).format(name=name)
    return {
        "input": qa_prompt(q, passage),
        "output": probe["answer"].format(name=name),
        "task": "refuse",
        "subtype": f"abstain_{probe['key']}",
        "passage_id": row["passage_id"],
        "country": row["country"],
        "name": name,
    }


# --------------------------------------------------------------------------
# Task D - scope refusal (no passage; out of the product's remit)
# --------------------------------------------------------------------------

SCOPE_QUESTIONS = [
    ("visa", "Do I need a visa to visit {country}?"),
    ("visa", "How long does a {country} tourist visa take to process?"),
    ("border", "Is the land border between {country} and its neighbours open?"),
    ("health", "What vaccinations are required for {country}?"),
    ("health", "Is malaria prophylaxis needed for travel to {country}?"),
    ("security", "Is {country} safe for tourists at the moment?"),
    ("security", "Are there any travel warnings for {country}?"),
    ("money", "What is the exchange rate for {country}'s currency?"),
    ("money", "How much should I budget per day in {country}?"),
    ("flights", "What is the cheapest flight to {country}?"),
    ("flights", "Which airlines fly into {country}?"),
    ("booking", "Can you book me a hotel in {country}?"),
    ("booking", "Can you arrange a tour guide for me in {country}?"),
]

SCOPE_ANSWERS = {
    "visa": "I cannot advise on visa requirements. They depend on your "
    "nationality and change without notice - check the {country} "
    "immigration authority or your nearest embassy.",
    "border": "I cannot confirm border status. That is live information and "
    "outside what I cover - check official government sources.",
    "health": "I cannot give health or vaccination advice. Check your "
    "national travel health service or a travel clinic before "
    "travelling to {country}.",
    "security": "I cannot advise on current safety or security conditions. "
    "Check your government's official travel advice for {country}.",
    "money": "I do not have current exchange rates or costs. These change "
    "daily - check a bank or a live currency service.",
    "flights": "I do not have flight schedules or fares. Check an airline or "
    "a flight search service for current routes into {country}.",
    "booking": "I cannot make bookings or arrange guides. I can help you find "
    "and understand destinations in {country}; booking and guide "
    "services are handled elsewhere.",
}

SCOPE_SUFFIX = (
    " What I can do is describe documented destinations in "
    "{country} and answer questions about them from their sources."
)


def build_scope_refusals(countries, n, rng) -> list[dict]:
    out = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        country = rng.choice(countries)
        key, qt = rng.choice(SCOPE_QUESTIONS)
        q = qt.format(country=country)
        if q in seen:
            continue
        seen.add(q)
        ans = SCOPE_ANSWERS[key].format(country=country) + SCOPE_SUFFIX.format(country=country)
        out.append(
            {
                "input": (
                    f"{TASK_TAGS['answer']} {ANSWER_INSTRUCTION}\n\n"
                    f"QUESTION: {q}\n\nPASSAGE:\n(none available)"
                ),
                "output": ans,
                "task": "refuse",
                "subtype": f"scope_{key}",
                "passage_id": None,
                "country": country,
                "name": None,
            }
        )
    return out


# --------------------------------------------------------------------------
# Quotas and assembly
# --------------------------------------------------------------------------


def country_quota_sample(df, n, cap_share, rng):
    """Sample n passages, capping any one country's share of the result."""
    if n >= len(df):
        return df.copy()
    cap = max(1, int(n * cap_share))
    picked = []
    for _, grp in df.groupby("country"):
        k = min(len(grp), cap)
        picked.append(grp.sample(k, random_state=rng.randint(0, 10**6)))
    pool = pd.concat(picked)
    if len(pool) > n:
        pool = pool.sample(n, random_state=rng.randint(0, 10**6))
    elif len(pool) < n:
        rest = df[~df["passage_id"].isin(pool["passage_id"])]
        extra = rest.sample(min(len(rest), n - len(pool)), random_state=rng.randint(0, 10**6))
        pool = pd.concat([pool, extra])
    return pool


def run(passages_path, out_dir, n_classify, n_qa, n_abstain, n_scope, seed):
    rng = random.Random(seed)
    df = pd.read_json(passages_path, lines=True)
    os.makedirs(out_dir, exist_ok=True)

    pairs: list[dict] = []

    # A - classification
    sub = country_quota_sample(df, n_classify, 0.22, rng)
    pairs += [build_classify(r) for _, r in sub.iterrows()]

    # B - extractive grounded QA
    sub = country_quota_sample(df, n_qa, 0.22, rng)
    pairs += [p for p in (build_fact_qa(r) for _, r in sub.iterrows()) if p]

    # C - abstention, spread across the four probe types
    per_probe = max(1, n_abstain // len(ABSTAIN_PROBES))
    for probe in ABSTAIN_PROBES:
        made, pool = 0, df.sample(frac=1, random_state=rng.randint(0, 10**6))
        for _, r in pool.iterrows():
            if made >= per_probe:
                break
            p = build_abstention(r, probe)
            if p:
                pairs.append(p)
                made += 1

    # D - scope refusals
    countries = sorted({c for c in df["country"] if ";" not in str(c)})
    pairs += build_scope_refusals(countries, n_scope, rng)

    rng.shuffle(pairs)
    path = os.path.join(out_dir, "pairs_deterministic.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = pd.DataFrame(pairs)
    report = {
        "total": len(pairs),
        "by_task": counts["task"].value_counts().to_dict(),
        "by_subtype": counts.get("subtype", pd.Series(dtype=str)).value_counts().to_dict(),
        "by_country_top": counts["country"].value_counts().head(20).to_dict(),
        "unique_passages_used": int(counts["passage_id"].nunique()),
    }
    with open(os.path.join(out_dir, "pairs_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="out/passages.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-classify", type=int, default=1200)
    ap.add_argument("--n-qa", type=int, default=900)
    ap.add_argument("--n-abstain", type=int, default=250)
    ap.add_argument("--n-scope", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args()
    print(
        json.dumps(
            run(a.passages, a.out, a.n_classify, a.n_qa, a.n_abstain, a.n_scope, a.seed),
            indent=2,
            ensure_ascii=False,
        )
    )
