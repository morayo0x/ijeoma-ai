"""
Hand-written reference digests.

Four digests written by hand from the cleaned passages, in exactly the format
the teacher pass is asked to produce. Two uses:

1. Plumbing - they exercise the teacher -> validate -> merge -> split path
   before a single API call is spent.
2. Quality yardstick - read these next to the teacher's first 20 outputs. If
   the teacher's are vaguer, longer, or reach for facts the passage does not
   contain, fix the prompt before running the full pass.

They are also good few-shot examples to paste into DIGEST_PROMPT if the
teacher's zero-shot output drifts.

Run:  python3 -I make_reference_digests.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REFERENCE = {
    "Royal Palaces of Abomey": (
        "Ten palaces spread across 47 hectares at the heart of Abomey, once "
        "capital of the Kingdom of Dahomey. The kingdom was founded in 1625 by "
        "the Fon people and grew into a military and commercial power that "
        "dominated trade with European slave traders on the Slave Coast until "
        "the late nineteenth century. Twelve kings ruled in succession to 1900. "
        "At their peak the palaces held up to 8000 people and included a "
        "two-storey cowrie house, or akuehue. UNESCO inscribed the site, later "
        "adding it to the List of World Heritage in Danger after a tornado "
        "struck in 1984."
    ),
    "Kakum National Park": (
        "A tropical forest park of 375 square kilometres in the coastal Central "
        "Region of Ghana, reserved in 1931 and gazetted as a national park in "
        "1992. Unusually, it was established on the initiative of local people "
        "rather than the state wildlife department. Its best known feature is a "
        "canopy walkway 350 metres long, linking seven tree tops, one of only "
        "three in Africa. Endangered wildlife includes Diana monkey, giant "
        "bongo antelope, yellow-backed duiker and African elephant. The park is "
        "an Important Bird Area, with 266 bird species confirmed, eight of them "
        "of global conservation concern."
    ),
    "Sukur": (
        "A UNESCO World Heritage Site on a hill above Sukur village in Adamawa "
        "State, Nigeria, set in the Mandara Mountains close to the Cameroon "
        "border. It was the first cultural landscape in Africa inscribed on the "
        "World Heritage List, recognised for its cultural heritage, material "
        "culture and naturally terraced fields. Iron Age furnaces, ore and "
        "grindstones found at the site predate the Sukur people, alongside "
        "finds from the Neolithic period. The Dur dynasty established the "
        "region in the seventeenth century as a major supplier of raw material "
        "for iron manufacture to northeastern Nigeria."
    ),
    "Bandiagara Escarpment": (
        "A sandstone cliff in the Dogon country of Mali, rising about 500 "
        "metres above the sandy flats to the south and running roughly 150 "
        "kilometres. The Dogon people live along it today; before them came the "
        "Tellem and Toloy, and many Tellem structures survive, including caves "
        "carved into the cliff face. The chain extends over 200 kilometres to "
        "the Grandamia massif, ending at Hombori Tondo, the highest peak in "
        "Mali at 1,155 metres. UNESCO listed the escarpment in 1989 for its "
        "archaeological, ethnological and geological characteristics."
    ),
}


def main(passages="out/passages.jsonl", out="out"):
    df = pd.read_json(passages, lines=True)
    written, missing = 0, []
    path = os.path.join(out, "teacher_cache.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        for name, digest in REFERENCE.items():
            hit = df[df["name"].astype(str).str.fullmatch(name, case=False)]
            if hit.empty:
                hit = df[df["name"].astype(str).str.contains(name, case=False, na=False)]
            if hit.empty:
                missing.append(name)
                continue
            row = hit.iloc[0]
            fh.write(json.dumps({
                "passage_id": row["passage_id"],
                "task": "digest",
                "raw": json.dumps({"digest": digest}, ensure_ascii=False),
                "_source": "hand-written reference",
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} reference digests to {path}")
    if missing:
        print("not found in passages.jsonl:", ", ".join(missing))


if __name__ == "__main__":
    main()
