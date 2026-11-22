"""
Ijeoma AI - summarization dataset.

Builds summarization pairs from the cleaned passage table. Targets are
EXTRACTIVE: every sentence in a summary is lifted verbatim from its passage,
so each pair is faithful by construction and costs nothing to produce.

What that buys and what it does not:

  + Selection is the hard half of summarization - deciding what matters in a
    destination passage - and extractive targets teach it directly.
  + Zero hallucination in the training signal. A model cannot learn to invent
    from targets that contain no invention.
  + Free, reproducible, and auditable: you can point at the source sentence
    for every word of every target.
  - It will not teach paraphrase or compression within a sentence. For that
    you need the teacher pass (ijeoma_teacher.py), which writes abstractive
    digests.

The intended path is to train on this now and merge abstractive digests in
later, so the model learns selection first and rewording second.

Sentence selection is a LexRank-style hybrid: centrality (how much a sentence
shares vocabulary with the rest of the passage), position, whether it defines
the subject, and information density, minus penalties for list-like or
oversized sentences.

Usage:
    python3 ijeoma_summarize.py --passages out/passages.jsonl --out out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter

import pandas as pd

TAG = "[SUMMARIZE]"

STOPWORDS = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "there",
        "here",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "then",
        "too",
        "very",
        "can",
        "will",
        "just",
        "also",
        "into",
        "over",
        "under",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "s",
        "t",
        "don",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
    ]
)

RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
RE_WORD = re.compile(r"[A-Za-zÀ-ɏ'’\-]+")
RE_NUM = re.compile(r"\d[\d,.]*")
RE_PROPER = re.compile(r"\b[A-Z][a-zÀ-ɏ'’\-]{2,}\b")
RE_DEFINITION = re.compile(
    r"\b(?:is|was|are|were)\s+(?:a|an|the|one of)\b"
    r"|\b(?:is|was)\s+(?:located|situated|found)\b",
    re.I,
)
RE_NOT_A_DEFINITION = re.compile(
    r"(may refer to|is also (the|an?) |abbreviation for|disambiguation|"
    r"for other uses|not to be confused|see also)",
    re.I,
)
RE_NOTABLE = re.compile(
    r"\b(unesco|world heritage|oldest|largest|biggest|first|only|best known|"
    r"known for|famous for|renowned|notable|significant|designated|inscribed|"
    r"declared|protected|rare|endangered|unique)\b",
    re.I,
)
# A sentence that is mostly a comma-separated roll-call of names summarises
# nothing; it just moves a list into the target.
RE_LISTY = re.compile(r"(,[^,]{1,24}){4,}")

# Extractive summarisation's classic failure: a sentence whose subject is a
# pronoun pointing at something that did not make it into the summary. Lifted
# out of context, "It is a Kwa language spoken by 800,000 people" attaches
# itself to whatever came before it and quietly asserts something false.
RE_DANGLING = re.compile(
    r"^\s*(?:it|he|she|they|this|that|these|those|its|his|her|their|such|"
    r"there|here|both|neither|the former|the latter|one of them)\b",
    re.I,
)

# Words in a place name that are too generic to identify it on their own,
# but which do make a fine referent inside its own passage ("the fort").
GENERIC_PLACE_WORDS = {
    "the",
    "of",
    "and",
    "national",
    "park",
    "reserve",
    "sanctuary",
    "museum",
    "gallery",
    "fort",
    "castle",
    "palace",
    "beach",
    "falls",
    "waterfall",
    "waterfalls",
    "river",
    "island",
    "islands",
    "mount",
    "mountain",
    "lake",
    "city",
    "town",
    "village",
    "sacred",
    "grove",
    "centre",
    "center",
    "state",
    "region",
    "district",
    "area",
    "monument",
    "memorial",
    "site",
    "shrine",
    "mosque",
    "church",
    "cathedral",
    "market",
    "square",
    "garden",
    "zoo",
    "forest",
    "hills",
    "rock",
    "cave",
    "valley",
    "bay",
    "delta",
    "gate",
}


def name_tokens(name: str) -> tuple[set[str], set[str]]:
    """Split a place name into distinctive and generic tokens."""
    toks = {w.lower() for w in RE_WORD.findall(str(name)) if len(w) > 2}
    generic = toks & GENERIC_PLACE_WORDS
    return toks - generic, generic


def is_about_subject(sentence: str, name: str) -> bool:
    """True when the sentence plainly concerns the passage's subject.

    Guards the short variants, where a single wrongly-chosen sentence is the
    entire output: "The ruins of the ancient ksour of Oudane is a UNESCO
    World Heritage Site" is a fine sentence and a false summary of a passage
    about Saharan Mauritania.
    """
    distinctive, generic = name_tokens(name)
    low = sentence.lower()
    if any(t in low for t in distinctive):
        return True
    return bool(generic) and any(re.search(rf"\b(?:the|this)\s+{t}\b", low) for t in generic)


def stable_choice(seq, *parts):
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return seq[int(h[:8], 16) % len(seq)]


def split_sentences(text: str) -> list[str]:
    """Split into sentences, re-joining splits made inside brackets.

    Source passages carry parentheticals with full stops in them
    ("(French: Réserve spéciale ... )"), and a naive split leaves a fragment
    that opens on a stray closing bracket.
    """
    raw = [s.strip() for s in RE_SENT_SPLIT.split(text.strip()) if s.strip()]
    out: list[str] = []
    for s in raw:
        if out and (
            out[-1].count("(") > out[-1].count(")") or out[-1].count("“") > out[-1].count("”")
        ):
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    # An orphan closing bracket means the sentence starts mid-parenthetical.
    return [s for s in out if s.count(")") <= s.count("(")]


def content_words(s: str) -> list[str]:
    return [w.lower() for w in RE_WORD.findall(s) if w.lower() not in STOPWORDS and len(w) > 2]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_sentences(sents: list[str], name: str) -> list[float]:
    """Score each sentence for how well it would represent the passage."""
    toks = [content_words(s) for s in sents]
    df = Counter()
    for t in toks:
        df.update(set(t))
    n_docs = max(1, len(sents))
    key = str(name).strip().lower()

    scores = []
    for i, (s, t) in enumerate(zip(sents, toks, strict=True)):
        words = len(s.split())
        if not t:
            scores.append(-10.0)
            continue

        # Centrality: mean inverse-document-frequency-weighted overlap with
        # the rest of the passage. Sentences sharing the passage's dominant
        # vocabulary are about the passage's actual subject.
        central = sum(
            (df[w] - 1) / n_docs * math.log(1 + n_docs / (1 + df[w])) for w in set(t)
        ) / math.sqrt(len(set(t)))

        # Position: early sentences matter, with a gentle decay rather than a
        # cliff, so this does not collapse into "take the first three".
        position = 1.0 / (1.0 + 0.45 * i)

        defines = 0.0
        if RE_DEFINITION.search(s) and not RE_NOT_A_DEFINITION.search(s):
            defines = 1.0
            if key and key in s.lower():
                defines = 1.6

        density = min(len(RE_PROPER.findall(s)), 4) * 0.12 + min(len(RE_NUM.findall(s)), 3) * 0.10
        notable = 0.45 if RE_NOTABLE.search(s) else 0.0

        penalty = 0.0
        if words < 8:
            penalty += 1.2
        if words > 48:
            penalty += 0.8
        if RE_LISTY.search(s):
            penalty += 1.0
        if RE_NOT_A_DEFINITION.search(s):
            penalty += 2.5
        if s.lstrip().startswith(('"', "'", "“")):
            penalty += 0.8

        scores.append(
            1.35 * central + 1.25 * position + 1.15 * defines + density + notable - penalty
        )
    return scores


def pick_sentences(sents, scores, k, name, max_words):
    """Choose k sentences: the subject definition first, then by score.

    Returned in original document order - a summary that reorders the
    passage's sentences reads as a jumble.
    """
    if not sents:
        return []
    order = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    key = str(name).strip().lower()

    chosen: list[int] = []
    # A summary that never says what the place IS has failed, whatever else
    # it contains. Seed with the best definitional sentence.
    for i in order:
        s = sents[i]
        if (
            RE_DEFINITION.search(s)
            and not RE_NOT_A_DEFINITION.search(s)
            and (not key or key in s.lower())
        ):
            chosen.append(i)
            break
    if not chosen:
        for i in order:
            if RE_DEFINITION.search(sents[i]) and not RE_NOT_A_DEFINITION.search(sents[i]):
                chosen.append(i)
                break
    if not chosen:
        chosen.append(order[0])

    used_words = len(sents[chosen[0]].split())
    for i in order:
        if len(chosen) >= k:
            break
        if i in chosen:
            continue
        # A pronoun subject is only safe when the sentence it refers back to
        # is itself in the summary, immediately before it.
        if RE_DANGLING.match(sents[i]) and (i - 1) not in chosen:
            continue
        # Guard against topic drift: a high-centrality sentence can still be
        # about a different place the passage merely mentions. Admit it only
        # if it names the subject, or continues a sentence already chosen.
        if not is_about_subject(sents[i], name) and (i - 1) not in chosen and (i + 1) not in chosen:
            continue
        w = len(sents[i].split())
        if used_words + w > max_words:
            continue
        # Skip anything that mostly repeats a sentence already chosen.
        cand = set(content_words(sents[i]))
        if cand and any(len(cand & set(content_words(sents[j]))) / len(cand) > 0.6 for j in chosen):
            continue
        chosen.append(i)
        used_words += w

    return sorted(chosen)


def summarise(passage, name, k, max_words):
    sents = split_sentences(passage)
    if len(sents) < 3:
        return None, 0
    scores = score_sentences(sents, name)
    idx = pick_sentences(sents, scores, k, name, max_words)
    if not idx:
        return None, 0
    # The summary cannot open on a dangling pronoun - there is nothing before
    # it to resolve against.
    while idx and RE_DANGLING.match(sents[idx[0]]):
        idx = idx[1:]
    if not idx:
        return None, 0
    return " ".join(sents[i] for i in idx), len(sents)


# --------------------------------------------------------------------------
# Task variants
# --------------------------------------------------------------------------

INSTR_SUMMARY = [
    "Summarise this destination passage in two or three sentences. Use only "
    "sentences supported by the passage.",
    "Give a short summary of the passage below, two to three sentences, "
    "grounded entirely in what it says.",
    "Condense this passage into a brief summary a traveller could read at a "
    "glance. Stay within what the passage states.",
    "Write a concise summary of the destination described below, no more than three sentences.",
]

INSTR_ONE_LINE = [
    "Summarise this passage in a single sentence.",
    "In one sentence, what does this passage say about the place?",
    "Give a one-sentence summary of the destination below.",
]

INSTR_NOTABLE = [
    "What does the passage say makes this place significant? Answer in one or "
    "two sentences, using only the passage.",
    "Summarise why this destination is notable, according to the passage.",
    "According to the passage, what is this place best known for? Keep it to one or two sentences.",
]


def build_pair(row, variant, text, instr_pool):
    instr = stable_choice(instr_pool, row["passage_id"], variant)
    return {
        "input": f"{TAG} {instr}\n\nPASSAGE:\n{row['passage']}",
        "output": text,
        "task": "summarize",  # scored as generation, own length bounds
        "subtype": variant,
        "passage_id": row["passage_id"],
        "country": row["country"],
        "name": row["name"],
        "theme": row["theme"],
    }


def compression(summary, passage):
    return len(summary.split()) / max(1, len(passage.split()))


def cap_country_share(pairs, cap, seed):
    """Drop pairs so no country exceeds `cap` of the set.

    Drops whole passages rather than individual pairs, so a passage never
    ends up with its summary in and its one-liner out.
    """
    import random as _r

    rng = _r.Random(seed)

    pairs_of: dict[str, list] = {}
    country_of: dict[str, str] = {}
    for p in pairs:
        pairs_of.setdefault(p["passage_id"], []).append(p)
        country_of[p["passage_id"]] = str(p["country"])

    counts = Counter(str(p["country"]) for p in pairs)
    available: dict[str, list[str]] = {}
    for pid, c in country_of.items():
        available.setdefault(c, []).append(pid)
    for c in available:
        available[c].sort()
        rng.shuffle(available[c])

    dropped: set[str] = set()
    # Drop a whole passage at a time from whichever country is most
    # over-represented, until the largest share is under the cap. Dropping
    # passages rather than pairs keeps a passage's summary and one-liner
    # together on the same side.
    while True:
        total = sum(counts.values())
        if total == 0:
            break
        country, n = counts.most_common(1)[0]
        if n <= cap * total or not available[country]:
            break
        pid = available[country].pop()
        counts[country] -= len(pairs_of[pid])
        dropped.add(pid)

    return [p for p in pairs if p["passage_id"] not in dropped]


def run(passages_path, out_dir, max_per_passage, seed):
    df = pd.read_json(passages_path, lines=True)
    os.makedirs(out_dir, exist_ok=True)

    pairs, skipped = [], Counter()
    for _, row in df.iterrows():
        made = 0
        sents = split_sentences(row["passage"])
        if len(sents) < 4:
            skipped["too_few_sentences"] += 1
            continue

        # ---- main summary, length scaled to the passage
        k = 2 if len(sents) < 8 else 3
        text, _ = summarise(row["passage"], row["name"], k, max_words=95)
        if text and 25 <= len(text.split()) <= 100 and compression(text, row["passage"]) <= 0.55:
            pairs.append(build_pair(row, "summary", text, INSTR_SUMMARY))
            made += 1
        else:
            skipped["summary_bounds"] += 1

        # ---- second pair: alternate between the short variants so neither
        # starves. Without this, "notable" never gets a slot.
        prefer_notable = stable_choice([True, False], row["passage_id"], "variant")
        plan = ["notable", "one_sentence"] if prefer_notable else ["one_sentence", "notable"]

        for variant in plan:
            if made >= max_per_passage:
                break

            if variant == "one_sentence":
                one, _ = summarise(row["passage"], row["name"], 1, max_words=45)
                if one and 8 <= len(one.split()) <= 45 and is_about_subject(one, row["name"]):
                    pairs.append(build_pair(row, "one_sentence", one, INSTR_ONE_LINE))
                    made += 1
                else:
                    skipped["one_sentence_off_subject"] += 1

            else:
                # Every sentence must be about this place and must stand on
                # its own, because the whole output is one or two sentences.
                cands = [
                    s
                    for s in sents
                    if RE_NOTABLE.search(s)
                    and 8 <= len(s.split()) <= 48
                    and not RE_DANGLING.match(s)
                    and is_about_subject(s, row["name"])
                ]
                if cands:
                    sc = score_sentences(cands, row["name"])
                    best = sorted(range(len(cands)), key=lambda i: sc[i], reverse=True)[:2]
                    txt = " ".join(cands[i] for i in sorted(best))
                    if 10 <= len(txt.split()) <= 85:
                        pairs.append(build_pair(row, "notable", txt, INSTR_NOTABLE))
                        made += 1
                else:
                    skipped["no_notable_sentence"] += 1

    # Nigeria is 46% of the corpus. Left alone it becomes 45% of this set and
    # every summarisation metric becomes a Nigeria metric.
    pairs = cap_country_share(pairs, cap=0.22, seed=seed)

    path = os.path.join(out_dir, "pairs_summarization.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    sdf = pd.DataFrame(pairs)
    comp = [compression(p["output"], p["input"].split("PASSAGE:\n", 1)[-1]) for p in pairs]
    report = {
        "passages_in": len(df),
        "pairs": len(pairs),
        "by_variant": sdf["subtype"].value_counts().to_dict(),
        "skipped": dict(skipped),
        "unique_passages": int(sdf["passage_id"].nunique()),
        "summary_words_median": int(sdf["output"].str.split().str.len().median()),
        "compression_median": round(float(pd.Series(comp).median()), 3),
        "by_country_top": sdf["country"].value_counts().head(10).to_dict(),
        "by_theme": sdf["theme"].value_counts().to_dict(),
    }
    with open(os.path.join(out_dir, "summarization_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="out/passages.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-per-passage", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args()
    print(
        json.dumps(run(a.passages, a.out, a.max_per_passage, a.seed), indent=2, ensure_ascii=False)
    )
