"""Self-test for the Stage 3 validator.

A validator that has never rejected anything has not been tested. These
cases are the failure modes the teacher pass is expected to produce.

Run:  python3 -I test_validator.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ijeoma_build import validate  # noqa: E402

PASSAGE = (
    "PASSAGE:\nKakum National Park is a rainforest reserve in the Central "
    "Region of Ghana, established in 1931 and covering 375 square kilometres."
)


def pair(task, output):
    tag = "[CLASSIFY]" if task == "classify" else "[ANSWER]"
    return {"task": task, "input": f"{tag} question\n\n{PASSAGE}", "output": output}


CASES = [
    ("clean answer", "answer",
     "Kakum National Park is a rainforest reserve in the Central Region of Ghana.",
     []),
    ("invented number", "answer",
     "Kakum National Park covers 980 square kilometres of rainforest in Ghana.",
     ["number_not_in_passage"]),
    ("invented entities", "answer",
     "Kakum sits beside Lake Bosumtwi near Kumasi township in Ashanti.",
     ["entity_not_in_passage"]),
    ("phone number leak", "digest",
     "A rainforest reserve in the Central Region of Ghana, established in 1931 "
     "and covering 375 square kilometres of protected forest, best known for "
     "its canopy walkway strung high above the forest floor and reachable on a "
     "day trip from the coast. Call +233 24 123 4567 to arrange entry first.",
     ["contact_details"]),
    ("price and hours leak", "digest",
     "A rainforest reserve in the Central Region of Ghana covering 375 square "
     "kilometres, established in 1931, best known for its canopy walkway strung "
     "high above the forest floor and reachable on a day trip from the coast "
     "for walkers of most ages. Entry costs $15 per adult and gates open 8 AM.",
     ["live_logistics"]),
    ("answer far too long", "answer", "Ghana " * 120, ["length"]),
    ("malformed json", "classify", '{country: Ghana, theme: Nature}',
     ["json_parse"]),
    ("missing json keys", "classify",
     '{"country": "Ghana", "theme": "Nature & Wildlife"}', ["json_keys"]),
    ("empty output", "answer", "   ", ["empty_output"]),
]


def main():
    failures = 0
    for label, task, output, expected in CASES:
        got = validate(pair(task, output))
        ok = all(e in got for e in expected) and (bool(got) == bool(expected))
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'}  {label:22s} expected={expected or ['pass']}  got={got or ['pass']}")
    print()
    print("all validator cases passed" if not failures
          else f"{failures} validator case(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
