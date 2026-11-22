"""LLM-as-judge evaluation for Ijeoma AI predictions.

Reads an evaluation CSV written by the training notebook (one row per test item,
with the prompt in ``input``, the model output in ``pred`` and the reference in
``ref``). For every answerable grounded item -- tasks ``answer``, ``digest`` and
``summarize`` whose reference is not a refusal -- a judge model:

* scores the output 1-5 for factual accuracy, cultural nuance and helpfulness,
  judged against the source passage only; and
* lists each atomic factual claim in the output and marks whether the passage
  supports it.

The headline metric, ``claims_supported_rate``, is supported claims divided by
all claims across judged items. It is the project's success criterion
(target >= 0.90).

Example::

    export OPENAI_API_KEY=...
    python evaluation/run_judge.py results/run2_full/eval_tuned_with_input.csv

Writes ``<input>_judged.csv`` (per-item scores) and prints a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from ijeoma_build import RE_REFUSAL, extract_passage

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.4-mini"
GROUNDED_TASKS = ("answer", "digest", "summarize")
MAX_PASSAGE_CHARS = 6000
RUBRIC = ("factual_accuracy", "cultural_nuance", "helpfulness")

JUDGE_PROMPT = """You are grading a travel assistant's answer about a West African destination.
Score it against the SOURCE PASSAGE only, from 1 (poor) to 5 (excellent), on:
- factual_accuracy: every claim is supported by the passage; nothing invented
- cultural_nuance: respectful, specific, free of stereotypes and Western travel cliches
- helpfulness: actually answers what was asked, clearly and concisely
Also list every atomic factual claim in the answer (one fact each: a date, a
builder, an event and a location are separate claims) and mark whether the
passage directly supports it. A refusal or a sentence with no factual content
contributes no claims.
Return JSON only: {{"factual_accuracy": n, "cultural_nuance": n, "helpfulness": n,
                    "claims": [{{"claim": "...", "supported": true}}]}}

QUESTION / TASK:
{task}

SOURCE PASSAGE:
{passage}

ASSISTANT ANSWER:
{answer}"""


def call_judge(
    task: str, passage: str, answer: str, model: str, api_key: str, retries: int = 4
) -> dict | None:
    """Send one item to the judge; return its parsed JSON verdict, or None."""
    body = json.dumps(
        {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        task=task, passage=passage[:MAX_PASSAGE_CHARS], answer=answer
                    ),
                }
            ],
        }
    ).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(API_URL, data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            return json.loads(payload["choices"][0]["message"]["content"])
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                print(f"judge call failed after {retries} attempts: {exc}", file=sys.stderr)
                return None
            time.sleep(2**attempt)
    return None


def judge_item(row: pd.Series, model: str, api_key: str) -> dict | None:
    """Judge one evaluation row and flatten the verdict into a result record."""
    prompt = row["input"]
    task_text = prompt[: prompt.find("PASSAGE:")].strip()
    verdict = call_judge(task_text, extract_passage(prompt), str(row["pred"]), model, api_key)
    if verdict is None:
        return None
    claims = verdict.get("claims") or []
    return {
        "row": row.name,
        "task": row["task"],
        **{k: verdict.get(k) for k in RUBRIC},
        "claims_total": len(claims),
        "claims_supported": sum(bool(c.get("supported")) for c in claims),
        "unsupported": json.dumps(
            [c.get("claim") for c in claims if not c.get("supported")], ensure_ascii=False
        ),
    }


def summarise(results: pd.DataFrame, n_eligible: int, model: str) -> dict:
    total, supported = results.claims_total.sum(), results.claims_supported.sum()
    return {
        "judge_model": model,
        "judged": len(results),
        "eligible": n_eligible,
        **{k: round(float(results[k].mean()), 2) for k in RUBRIC},
        "claims_total": int(total),
        "claims_supported": int(supported),
        "claims_supported_rate": round(float(supported / total), 3) if total else None,
        "claims_supported_rate_by_task": {
            task: round(float(g.claims_supported.sum() / max(1, g.claims_total.sum())), 3)
            for task, g in results.groupby("task")
        },
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Score Ijeoma AI predictions with an LLM judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Requires OPENAI_API_KEY in the environment.",
    )
    parser.add_argument("eval_csv", type=Path, help="evaluation CSV with an 'input' column")
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--workers", type=int, default=8, help="parallel judge requests")
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY is not set")

    df = pd.read_csv(args.eval_csv)
    if "input" not in df.columns:
        parser.error(f"{args.eval_csv} has no 'input' column; re-export it from the notebook")

    is_refusal = df["ref"].astype(str).map(lambda s: bool(RE_REFUSAL.search(s)))
    eligible = df[df["task"].isin(GROUNDED_TASKS) & ~is_refusal]

    with ThreadPoolExecutor(args.workers) as pool:
        records = pool.map(
            lambda r: judge_item(r, args.model, api_key), [r for _, r in eligible.iterrows()]
        )
        results = pd.DataFrame([r for r in records if r])

    out_path = args.eval_csv.with_name(f"{args.eval_csv.stem}_judged.csv")
    results.to_csv(out_path, index=False)

    summary = summarise(results, len(eligible), args.model)
    print(json.dumps(summary, indent=2))
    print(f"per-item scores: {out_path}", file=sys.stderr)
    return summary


if __name__ == "__main__":
    main()
