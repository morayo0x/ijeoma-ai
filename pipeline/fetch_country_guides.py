"""Fetch Wikivoyage country guides for the 16 countries in the corpus.

For each country, three passages are written to ``out/country_guides.jsonl``:

* ``overview`` - the guide's introduction and the opening of "Understand";
* ``language`` - the "Talk" section;
* ``climate`` - the "Climate" subsection of "Understand".

These passages are used by the chat interface to answer country-level questions
about languages and climate, and "tell me about <country>". They are not part
of the training or test data.

Run:  python fetch_country_guides.py --out out
Source text: Wikivoyage, CC BY-SA 4.0; each passage records its URL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikivoyage.org/w/api.php"
USER_AGENT = "ijeoma-ai/0.3 (research project; https://github.com/morayo0x/ijeoma-ai)"
MAX_CHARS = 3500

# Corpus country name -> Wikivoyage article title.
COUNTRIES = {
    "Benin": "Benin",
    "Burkina Faso": "Burkina Faso",
    "Cape Verde": "Cape Verde",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Ghana": "Ghana",
    "Guinea": "Guinea",
    "Guinea-Bissau": "Guinea-Bissau",
    "Liberia": "Liberia",
    "Mali": "Mali",
    "Mauritania": "Mauritania",
    "Niger": "Niger",
    "Nigeria": "Nigeria",
    "Senegal": "Senegal",
    "Sierra Leone": "Sierra Leone",
    "The Gambia": "The Gambia",
    "Togo": "Togo",
}

HEADING = re.compile(r"^(=+) ?(.+?) ?=+\s*$", re.M)


def fetch_extract(title: str) -> tuple[str, str]:
    """Return (resolved title, plain-text article with == headings ==)."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "titles": title,
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                page = json.load(response)["query"]["pages"][0]
            return page["title"], page.get("extract", "")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            wait = int(exc.headers.get("Retry-After") or 0) or 5 * 2**attempt
            print(f"  rate limited; waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def split_sections(text: str) -> list[tuple[int, str, str]]:
    """[(level, heading, body)], with the lead as (1, "", body)."""
    sections, last, level, heading = [], 0, 1, ""
    for match in HEADING.finditer(text):
        sections.append((level, heading, text[last : match.start()]))
        level, heading, last = len(match.group(1)), match.group(2).strip(), match.end()
    sections.append((level, heading, text[last:]))
    return sections


def clean(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text[:MAX_CHARS].rsplit(" ", 1)[0] if len(text) > MAX_CHARS else text


def guide_passages(country: str, title: str, text: str) -> list[dict]:
    sections = split_sections(text)
    lead = next((body for lvl, head, body in sections if head == ""), "")
    understand = next((body for lvl, head, body in sections if head == "Understand"), "")
    talk = next((body for lvl, head, body in sections if head == "Talk"), "")
    climate = next(
        (body for lvl, head, body in sections if head.lower() in ("climate", "weather")), ""
    )
    if not climate:
        # Some guides discuss climate inside "Understand" rather than in its own section.
        climate = "\n".join(
            para
            for lvl, head, body in sections
            for para in body.split("\n")
            if re.search(r"\b(climate|rainy season|dry season|harmattan|rainfall)\b", para, re.I)
        )
    url = f"https://en.wikivoyage.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    retrieved = dt.date.today().isoformat()
    ascii_name = unicodedata.normalize("NFKD", country).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z]+", "-", ascii_name.lower()).strip("-")
    out = []
    for facet, body in (
        ("overview", f"{lead}\n{understand}"),
        ("language", talk),
        ("climate", climate),
    ):
        body = clean(body)
        if len(body.split()) < 25:
            continue
        out.append(
            {
                "passage_id": f"country-{slug}-{facet}",
                "name": country,
                "country": country,
                "facet": facet,
                "source": "wikivoyage",
                "url": url,
                "retrieved": retrieved,
                "passage": body,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    rows, missing = [], []
    for country, title in COUNTRIES.items():
        resolved, text = fetch_extract(title)
        passages = guide_passages(country, resolved, text)
        found = {p["facet"] for p in passages}
        missing += [f"{country}:{f}" for f in ("overview", "language", "climate") if f not in found]
        rows += passages
        print(f"{country:14} {resolved:16} {', '.join(sorted(found))}")
        time.sleep(3)  # stay well within the API's rate limits

    path = args.out / "country_guides.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} passages written to {path}")
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
