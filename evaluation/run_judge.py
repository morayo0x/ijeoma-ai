"""Offline LLM judge for Ijeoma AI eval CSVs (see docs/DECISIONS.md D-16, D-20).

Scores each answerable grounded item (answer / digest / summarize) 1-5 on
factual accuracy, cultural nuance and helpfulness, and lists every atomic
claim with whether the source passage supports it.

    OPENAI_API_KEY=... python3 evaluation/run_judge.py results/run2_full/eval_tuned_with_input.csv
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
GROUNDED = ("answer", "digest", "summarize")
RE_REFUSAL = re.compile(
    r"((?:does|do)(?:\s+not|n't)\s+"
    r"(?:say|mention|cover|provide|describe|give|list|include|state|have)"
    r"|i (?:cannot|can't|am not able|do not have|don't have)"
    r"|cannot advise|can't advise|unable to|no information"
    r"|not covered|not mentioned|not stated"
    r"|source does not|passage does not)", re.I)

PROMPT = """You are grading a travel assistant's answer about a West African destination.
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


def judge(task, passage, answer):
    body = json.dumps({"model": MODEL, "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": PROMPT.format(
                           task=task, passage=passage[:6000], answer=answer)}]}).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(json.load(r)["choices"][0]["message"]["content"])
        except Exception as exc:
            if attempt == 3:
                print("judge failed:", exc, file=sys.stderr)
            time.sleep(2 ** attempt)
    return None


def main(path):
    df = pd.read_csv(path)
    items = df[df.task.isin(GROUNDED) & ~df.ref.astype(str).str.contains(RE_REFUSAL)]

    def work(row):
        inp = row["input"]
        cut = inp.find("PASSAGE:")
        s = judge(inp[:cut].strip(), inp[cut + 8:].strip(), str(row["pred"]))
        if not s:
            return None
        claims = s.get("claims") or []
        return {"row": row.name, "task": row["task"],
                "factual_accuracy": s.get("factual_accuracy"),
                "cultural_nuance": s.get("cultural_nuance"),
                "helpfulness": s.get("helpfulness"),
                "claims_total": len(claims),
                "claims_supported": sum(bool(c.get("supported")) for c in claims),
                "unsupported": json.dumps([c.get("claim") for c in claims
                                           if not c.get("supported")], ensure_ascii=False)}

    with ThreadPoolExecutor(8) as ex:
        out = [r for r in ex.map(work, [r for _, r in items.iterrows()]) if r]
    res = pd.DataFrame(out)
    res.to_csv(path.replace(".csv", "_judged.csv"), index=False)
    tot, sup = res.claims_total.sum(), res.claims_supported.sum()
    summary = {"judged": len(res), "of_eligible": len(items),
               "factual_accuracy": round(res.factual_accuracy.mean(), 2),
               "cultural_nuance": round(res.cultural_nuance.mean(), 2),
               "helpfulness": round(res.helpfulness.mean(), 2),
               "claims_total": int(tot), "claims_supported": int(sup),
               "claims_supported_rate": round(sup / tot, 3) if tot else None,
               "by_task_rate": {t: round(g.claims_supported.sum() / max(1, g.claims_total.sum()), 3)
                                for t, g in res.groupby("task")}}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(sys.argv[1])
