"""
Ijeoma AI - Stage 1: cleaning, filtering and label normalisation.

Reads the raw West Africa scrape and produces a clean passage table that
Stage 2 (pair generation) consumes. Nothing here calls a model; every
decision is deterministic and auditable.

Usage:
    python3 ijeoma_clean.py --in <raw.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata

import pandas as pd

# --------------------------------------------------------------------------
# 1. Non-place detection
# --------------------------------------------------------------------------
# The scrape walked Wikipedia category members, so some rows are biographies,
# festivals or organisations wearing an attraction_type label. Training on
# them teaches the model to describe people as places.

PERSON_ROLE = (
    r"politician|writer|author|poet|novelist|academic|academician|professor|"
    r"librarian|businesswoman|businessman|singer|musician|footballer|actor|"
    r"actress|activist|lawyer|physician|journalist|historian|economist|"
    r"diplomat|soldier|archbishop|bishop|imam|scholar|teacher|engineer|"
    r"entrepreneur|trader|merchant|signare|nurse|artist|painter|sculptor|"
    r"photographer|filmmaker|dancer|designer|pastor|priest|curator|"
    r"gallerist|art collector|statesman|monarch|chieftain"
)

# "<Subject> is/was a ... <role>" inside the FIRST sentence only.
RE_PERSON_DEF = re.compile(
    rf"^[^.]{{0,160}}?\b(?:is|was|were)\s+(?:a|an|the)\s+[^.]{{0,60}}?\b(?:{PERSON_ROLE})\b",
    re.I,
)
# Birth/death parenthetical right after the subject: "(1787-1872)", "(born 1954)".
RE_LIFESPAN = re.compile(
    r"^[^.]{0,80}\((?:born\s+)?(?:\d{1,2}\s+\w+\s+)?\d{4}\s*(?:[–—-]\s*(?:\d{1,2}\s+\w+\s+)?\d{4})?\)",
)
RE_FESTIVAL = re.compile(
    r"^[^.]{0,160}?\b(?:is|was)\s+(?:a|an|the)\s+[^.]{0,60}?"
    r"\b(?:festival|conference|award|prize|competition|championship|"
    r"ceremony|carnival|pageant|summit|exhibition held)\b",
    re.I,
)
# A veto only counts when the sentence DEFINES the subject as a place.
# "Bayo Adebowale ... was born in Adeyipo village" must not veto, so a bare
# place noun anywhere in the sentence is not enough.
PLACE_NOUN = (
    r"palace|park|beach|museum|gallery|river|forest|island|falls?|mountain|"
    r"hill|village|town|city|reserve|fort|fortress|castle|shrine|grove|"
    r"monument|memorial|site|lake|market|mosque|church|cathedral|tomb|gate|"
    r"square|garden|zoo|resort|lagoon|rock|cave|valley|plateau|crater|"
    r"sanctuary|settlement|district|region|capital|state|area|centre|center|"
    r"building|house|library|archive|school|university|hotel|neighbourhood|"
    r"neighborhood|quarter|street|road|bridge|port|harbour|harbor|lighthouse|"
    r"waterfall|spring|dune|desert|wetland|estuary|peninsula|bay"
)
RE_PLACE_HEAD = re.compile(
    rf"^[^.]{{0,160}}?\b(?:is|was|are|were)\s+(?:a|an|the|one of)\s+"
    rf"[^.]{{0,70}}?\b(?:{PLACE_NOUN})\b"
    rf"|^[^.]{{0,120}}?\b(?:is|was)\s+(?:located|situated|found|sited)\b",
    re.I,
)
# Several third-person-singular personal clauses = a biography.
RE_PRONOUN_RUN = re.compile(
    r"\b(?:He|She)\s+(?:was|is|has|had|became|served|died|studied|worked|"
    r"attended|received|joined|founded|published|won|holds|obtained|graduated)\b"
)


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


def classify_non_place(text: str, raw_type: str = "") -> str | None:
    """Return a reason string if the passage is not about a place, else None.

    Deliberately conservative. Rejecting a real destination (Ibadan, Ile-Ife)
    costs far more than letting a stray biography through, so anything the
    rules are unsure about is kept and flagged for review instead.
    """
    if "destination guide" in str(raw_type).lower():
        return None          # a town or area guide is a place by construction

    head = first_sentences(text, 2)
    if not head:
        return "empty"

    person = bool(RE_PERSON_DEF.search(head) or RE_LIFESPAN.search(head))
    event = bool(RE_FESTIVAL.search(head))
    place = bool(RE_PLACE_HEAD.search(head))

    # A place definition only clears a passage that has no person signal;
    # when both fire, whichever defines the subject first wins.
    if person and place:
        p_at = min((m.start() for m in (RE_PERSON_DEF.search(head),
                                        RE_LIFESPAN.search(head)) if m), default=10**6)
        pl_at = RE_PLACE_HEAD.search(head).start()
        return None if pl_at < p_at else "biography"
    if person:
        return "biography"
    if event and not place:
        return "event"
    return None


def needs_review(text: str, raw_type: str = "") -> bool:
    """Kept, but a human should glance at it.

    Catches the residue the rules cannot settle: monuments and museums named
    after a person whose biography the scrape pulled in instead of a
    description of the site.
    """
    if classify_non_place(text, raw_type) is not None:
        return False                       # already rejected
    head = text[:300]
    if RE_PERSON_DEF.search(head) or RE_LIFESPAN.search(head):
        return True
    if len(RE_PRONOUN_RUN.findall(text[:600])) >= 2 \
            and "destination guide" not in str(raw_type).lower():
        return True
    return False


# --------------------------------------------------------------------------
# 2. Content cleaning
# --------------------------------------------------------------------------

RE_ADDL_SOURCE = re.compile(r"\[Additional source:[^\]]*\]")
RE_REF = re.compile(r"\[\d+\]")
RE_WIKI_MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}")
RE_UPDATED = re.compile(r"\(updated [^)]{1,40}\)", re.I)
RE_PHONE = re.compile(r"☏|\+\d{2,3}[\s\-]?\d[\d\s\-]{5,}")
RE_EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
RE_URL = re.compile(r"https?://\S+")
RE_HOURS = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:[\s–—\-]+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun))?"
    r"[\s,]*\d{1,2}(?::\d{2})?\s*(?:AM|PM)",
    re.I,
)
RE_PRICE = re.compile(
    r"(?:₦|US\$|\$|€|£|CFA|GH₵|XOF|NGN|GHS)\s?\d[\d,.]*",
    re.I,
)
RE_HEADING = re.compile(r"^\s*(=+)\s*(.+?)\s*=+\s*$")
# Bare opening hours with no weekday in front: "9AM-5PM", "10 PM".
RE_BARE_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b", re.I)

# Wikivoyage sections that are pure commercial listings - dropped wholesale.
LISTING_SECTIONS = {
    "eat", "drink", "sleep", "buy", "connect", "stay safe", "stay healthy",
    "cope", "work", "get in", "get around", "respect",
}
# Sections worth keeping from a destination guide.
KEEP_SECTIONS = {
    "understand", "see", "do", "go next", "history", "climate",
    "landscape", "talk", "regions", "cities", "other destinations",
}


def clean_html_entities(text: str) -> str:
    text = re.sub(r"<[^>]{1,80}>", "", text)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                 ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        text = text.replace(a, b)
    return text


def strip_control_chars(text: str) -> str:
    out = []
    for ch in text:
        if ch in "\n\t":
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "N", "P", "Z", "S")) and cat != "Cc":
            # Drop emoji / pictographs but keep ordinary symbols like degrees.
            if cat == "So" and ord(ch) > 0x2100:
                continue
            out.append(ch)
    return "".join(out)


def line_is_listing(line: str) -> bool:
    if RE_PHONE.search(line) or RE_EMAIL.search(line):
        return True
    if RE_HOURS.search(line) or RE_BARE_TIME.search(line):
        return True
    if RE_UPDATED.search(line) and len(line) < 400:
        return True
    return False


def clean_content(text: str, source: str) -> str:
    """Remove scrape artefacts, commercial listings and live logistics."""
    text = clean_html_entities(str(text))
    text = RE_ADDL_SOURCE.sub(" ", text)
    text = RE_WIKI_MARKUP.sub(" ", text)
    text = RE_REF.sub("", text)
    text = RE_URL.sub("", text)

    kept: list[str] = []
    current_section = ""
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        m = RE_HEADING.match(line)
        if m:
            # Only a top-level "== Section ==" changes the section. A nested
            # "=== By bus ===" stays inside its parent, so the whole of
            # "Get in" is dropped rather than just its first paragraph.
            if len(m.group(1)) <= 2:
                current_section = m.group(2).strip().lower()
            continue  # headings themselves are dropped from the passage
        if not line.strip():
            continue
        if source == "wikivoyage" and current_section in LISTING_SECTIONS:
            continue
        if line_is_listing(line):
            continue
        line = RE_PRICE.sub("", line)
        line = RE_UPDATED.sub("", line)
        line = RE_PHONE.sub("", line)
        # Drop bullet leaders and stray numbering from Wikivoyage listings.
        line = re.sub(r"^\s*[•\-\*]\s*", "", line)
        line = re.sub(r"^\s*\d{1,2}\s+(?=[A-Z])", "", line)
        line = re.sub(r"\s{2,}", " ", line).strip(" ,;.–-")
        if len(line) < 25:          # fragments left over from listings
            continue
        # Dropping headings can butt two sentences together; keep the boundary.
        if line and line[-1] not in ".!?:;…":
            line += "."
        kept.append(line)

    out = "\n".join(kept)
    out = strip_control_chars(out)
    out = re.sub(r"\n{2,}", "\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def trim_passage(text: str, max_chars: int = 2400) -> str:
    """Trim on a sentence boundary so no claim is cut mid-way."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for boundary in (". ", ".\n", "! ", "? "):
        idx = cut.rfind(boundary)
        if idx > max_chars * 0.5:
            return cut[: idx + 1].strip()
    return cut.rsplit(" ", 1)[0].strip()


# --------------------------------------------------------------------------
# 3. Label normalisation
# --------------------------------------------------------------------------

THEME_OF_TYPE = {
    "destination guide": "Town or Area Guide",
    "protected area": "Nature & Wildlife",
    "national park": "Nature & Wildlife",
    "nature reserve": "Nature & Wildlife",
    "wood": "Nature & Wildlife",
    "zoo": "Family & Leisure",
    "theme park": "Family & Leisure",
    "beach": "Coast & Beach",
    "waterfall": "Landscape & Scenery",
    "viewpoint": "Landscape & Scenery",
    "museum": "Museum & Gallery",
    "gallery": "Museum & Gallery",
    "archaeological site": "Archaeological Site",
    "ruins": "Archaeological Site",
    "tomb": "Archaeological Site",
    "monument": "Monument & Memorial",
    "city gate": "Monument & Memorial",
    "castle": "Fort & Castle",
    "fort": "Fort & Castle",
    "palace": "Palace & Royal Site",
    "historic site": "Historic Site",
    "tourist attraction": None,   # unspecified - infer or fall back
    "attraction": None,
    "world heritage site": None,  # orthogonal flag, not a theme
}

# Most specific first: when a row carries several types, the earliest match wins.
THEME_PRIORITY = [
    "Sacred & Ceremonial",
    "Palace & Royal Site",
    "Fort & Castle",
    "Archaeological Site",
    "Museum & Gallery",
    "Monument & Memorial",
    "Coast & Beach",
    "Landscape & Scenery",
    "Nature & Wildlife",
    "Family & Leisure",
    "Historic Site",
    "Town or Area Guide",
]

# Checked in order of specificity, most specific first: a passage about a
# palace in the mountains is a palace, not scenery. Within that order the
# category with the most mentions wins, so one stray word cannot decide it.
THEME_KEYWORDS = [
    ("Sacred & Ceremonial", r"\b(sacred|shrine|grove|deity|deities|orisha|oracle|"
                            r"mosque|church|cathedral|basilica|pilgrimage|ritual|"
                            r"spiritual|place of worship|juju|ancestral)\b"),
    ("Palace & Royal Site", r"\b(palace|royal court|royal enclosure|king'?s court|emir'?s? palace|oba'?s palace)\b"),
    ("Fort & Castle", r"\b(fort|fortress|castle|citadel|slave castle)\b"),
    ("Museum & Gallery", r"\b(museum|art gallery|exhibition hall)\b"),
    ("Archaeological Site", r"\b(archaeolog|excavat|ruins|ancient settlement|rock art|petroglyph|burial|iron age|neolithic|stone age|relics)\b"),
    ("Monument & Memorial", r"\b(monument|memorial|statue|obelisk|cenotaph|mausoleum)\b"),
    ("Coast & Beach", r"\b(beach|seafront|shoreline|lagoon|coastal resort)\b"),
    ("Nature & Wildlife", r"\b(national park|forest reserve|wildlife|game reserve|sanctuary|biosphere|wetland|conservation)\b"),
    ("Landscape & Scenery", r"\b(waterfall|falls|escarpment|plateau|crater|hot spring|cave|viewpoint|mountain|peak)\b"),
    ("Historic Site", r"\b(historic|heritage building|colonial building|slave trade|mission house|cultural landscape)\b"),
]

TERRAIN_KEYWORDS = [
    ("Coastal", r"\b(beach|coast|coastal|shoreline|seafront|atlantic ocean|bay|lagoon|peninsula)\b"),
    ("Island", r"\b(island|islet|archipelago|isle)\b"),
    ("Riverine", r"\b(river|riverbank|riverine|delta|creek|estuary|tributary)\b"),
    ("Lake or wetland", r"\b(lake|wetland|marsh|swamp|reservoir|floodplain)\b"),
    ("Forest", r"\b(forest|rainforest|woodland|mangrove|grove)\b"),
    ("Mountain or highland", r"\b(mountain|hill|highland|plateau|escarpment|massif|peak|range)\b"),
    ("Desert or Sahel", r"\b(desert|sahara|sahel|dune|arid|semi-arid)\b"),
    ("Savannah", r"\b(savannah|savanna|grassland|steppe)\b"),
    ("Urban", r"\b(city centre|city center|downtown|metropolis|urban|neighbourhood|neighborhood|district of)\b"),
]

PROFILE_OF_THEME = {
    "Sacred & Ceremonial": "Culture-focused, slow travel",
    "Museum & Gallery": "Culture-focused, indoor",
    "Archaeological Site": "History and heritage, slow travel",
    "Fort & Castle": "History and heritage",
    "Palace & Royal Site": "Culture-focused, heritage",
    "Monument & Memorial": "Short visit, city sightseeing",
    "Historic Site": "History and heritage",
    "Nature & Wildlife": "Nature and wildlife, outdoor",
    "Coast & Beach": "Beach and relaxation",
    "Landscape & Scenery": "Scenic, outdoor, active",
    "Family & Leisure": "Family-friendly, leisure",
    "Town or Area Guide": "General, base for exploring",
    "General Attraction": "General sightseeing",
}


def split_types(raw: str) -> list[str]:
    return [p.strip().lower() for p in str(raw).split(";") if p.strip()]


def normalise_theme(raw_type: str, content: str) -> tuple[str, bool, str]:
    """Return (theme, is_world_heritage, how_it_was_decided)."""
    types = split_types(raw_type)
    wh = "world heritage site" in types

    candidates = {THEME_OF_TYPE.get(t) for t in types}
    candidates.discard(None)
    for theme in THEME_PRIORITY:
        if theme in candidates:
            return theme, wh, "type"

    head = content[:1200]
    scores = [(len(re.findall(pat, head, re.I)), -i, theme)
              for i, (theme, pat) in enumerate(THEME_KEYWORDS)]
    scores = [s for s in scores if s[0] > 0]
    if scores:
        scores.sort(reverse=True)       # most mentions, then most specific
        return scores[0][2], wh, "keyword"

    return "General Attraction", wh, "fallback"


def infer_terrain(content: str) -> str | None:
    """Return a terrain only when the passage clearly supports one.

    A single incidental mention ('a bay 40km away') is not evidence about
    this place, so a category must either be introduced early or be repeated.
    Returning None is the correct, honest answer for most passages and the
    model is trained to produce it.
    """
    text = content[:1800]
    scored: list[tuple[int, int, str]] = []
    for idx, (name, pat) in enumerate(TERRAIN_KEYWORDS):
        hits = list(re.finditer(pat, text, re.I))
        if not hits:
            continue
        strong = hits[0].start() < 400 or len(hits) >= 2
        if strong:
            scored.append((-len(hits), idx, name))
    if not scored:
        return None
    scored.sort()                      # most mentions, then most distinctive
    return scored[0][2]


# --------------------------------------------------------------------------
# 4. Pipeline
# --------------------------------------------------------------------------

# Some OSM rows were enriched from the venue's own website, so the passage
# opens with navigation chrome rather than a description.
RE_CHROME = re.compile(
    r"(top of page|skip to (?:main )?content|main menu|click here|read more|"
    r"learn more|subscribe|newsletter|all rights reserved|\u00a9\s*\d{4}|"
    r"follow us|share this|cookie polic|privacy polic|terms of use)", re.I)


def is_web_boilerplate(text: str) -> bool:
    """True when the passage is page furniture rather than prose."""
    if RE_CHROME.search(text[:300]):
        return True
    return len(set(m.group(0).lower() for m in RE_CHROME.finditer(text))) >= 2


def normalise_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", n.lower()).strip()


def run(in_path: str, out_dir: str, min_chars: int = 300, max_chars: int = 2400):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_path)
    report: dict[str, object] = {"rows_in": len(df)}

    df["content"] = df["content"].fillna("")
    df["clean"] = [clean_content(c, s) for c, s in zip(df["content"], df["source"])]

    df["reject"] = ""
    df.loc[df["clean"].str.len() < min_chars, "reject"] = "too_short_after_cleaning"
    df.loc[(df["reject"] == "") & df["clean"].map(is_web_boilerplate),
           "reject"] = "web_boilerplate"

    reasons = [
        classify_non_place(c, t) if rej == "" else None
        for c, t, rej in zip(df["clean"], df["attraction_type"], df["reject"])
    ]
    df["reject"] = [
        f"not_a_place:{r}" if r else rej
        for r, rej in zip(reasons, df["reject"])
    ]

    # Duplicate names: keep the longest cleaned passage.
    df["name_key"] = df["name"].map(normalise_name)
    keep_idx = (
        df[df["reject"] == ""]
        .assign(_len=lambda d: d["clean"].str.len())
        .sort_values("_len", ascending=False)
        .drop_duplicates("name_key")
        .index
    )
    dup_mask = (df["reject"] == "") & (~df.index.isin(keep_idx))
    df.loc[dup_mask, "reject"] = "duplicate_name"

    kept = df[df["reject"] == ""].copy()
    kept["passage"] = kept["clean"].map(lambda t: trim_passage(t, max_chars))

    themes = [normalise_theme(t, c) for t, c in zip(kept["attraction_type"], kept["passage"])]
    kept["theme"] = [t[0] for t in themes]
    kept["world_heritage"] = [t[1] for t in themes]
    kept["theme_source"] = [t[2] for t in themes]
    kept["terrain"] = kept["passage"].map(infer_terrain)
    kept["traveler_profile"] = kept["theme"].map(PROFILE_OF_THEME)
    kept["needs_review"] = [
        needs_review(p, t) for p, t in zip(kept["passage"], kept["attraction_type"])
    ]
    # Content-derived so that a later cleaning change does not renumber every
    # passage and silently invalidate an existing train/test split.
    kept["passage_id"] = [
        "p" + hashlib.sha1(
            f"{n}|{c}|{u}".encode("utf-8")
        ).hexdigest()[:10]
        for n, c, u in zip(kept["name"], kept["country"], kept["url"])
    ]
    kept[kept["needs_review"]][["passage_id", "name", "country",
                                "attraction_type", "theme"]] \
        .to_csv(os.path.join(out_dir, "review.csv"), index=False)

    cols = ["passage_id", "name", "country", "theme", "world_heritage",
            "theme_source", "terrain", "traveler_profile", "needs_review", "attraction_type",
            "source", "url", "latitude", "longitude", "passage"]
    out = kept[cols]
    out.to_json(os.path.join(out_dir, "passages.jsonl"),
                orient="records", lines=True, force_ascii=False)

    report.update({
        "rows_kept": int(len(out)),
        "rejected": df[df["reject"] != ""]["reject"]
                      .str.split(":").str[0].value_counts().to_dict(),
        "chars_before_mean": float(df["content"].str.len().mean()),
        "chars_after_mean": float(out["passage"].str.len().mean()),
        "theme_counts": out["theme"].value_counts().to_dict(),
        "theme_source": out["theme_source"].value_counts().to_dict(),
        "terrain_counts": out["terrain"].fillna("(unassigned)").value_counts().to_dict(),
        "world_heritage": int(out["world_heritage"].sum()),
        "country_counts": out["country"].value_counts().to_dict(),
    })
    df[df["reject"] != ""][["name", "country", "attraction_type", "source", "reject"]] \
        .to_csv(os.path.join(out_dir, "rejected.csv"), index=False)
    with open(os.path.join(out_dir, "clean_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_dir", default="out")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=2400)
    a = ap.parse_args()
    rep = run(a.in_path, a.out_dir, a.min_chars, a.max_chars)
    print(json.dumps(rep, indent=2, ensure_ascii=False)[:4000])
