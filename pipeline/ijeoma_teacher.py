"""
Ijeoma AI - Stage 2b: teacher-generated abstractive pairs.

Produces the two slices that cannot be derived mechanically: the 60-100 word
traveller digest, and abstractive question-answer pairs (including unanswerable
ones). Everything else comes from ijeoma_pairs.py and costs nothing.

Designed for Colab: the API key is read from Colab secrets when available and
from the environment otherwise. Results are cached per passage, so an
interrupted run resumes without re-spending, and no passage is ever sent to
the teacher twice - which is the commitment made in the project's
environmental-impact section.

    pip install google-genai
    python3 ijeoma_teacher.py --passages out/passages.jsonl --out out \
        --n-digest 1000 --n-qa 700

Smoke-test first; it costs almost nothing and catches a bad key or a bad
model name before you spend the full run:

    python3 ijeoma_teacher.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ijeoma_build import validate
from ijeoma_pairs import TASK_TAGS, country_quota_sample, qa_prompt

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

# --------------------------------------------------------------------------
# Prompts - the three hard constraints live here
# --------------------------------------------------------------------------

RULES = """Rules you must follow exactly:
1. Use ONLY the passage. Add nothing from your own knowledge, however certain you are.
2. Never write a phone number, email, website, price, entry fee, opening hour,
   or a date stamp such as "updated Nov 2025".
3. Never use marketing language ("must-see", "hidden gem", "breathtaking").
4. If the passage is too thin to do the task properly, return {"skip": true}.
Return JSON only, with no code fence."""

DIGEST_PROMPT = """You are preparing training data for a travel discovery assistant
covering West Africa.

Write a traveller's digest of the destination below in 60 to 100 words. Say what
the place is, what a visitor would actually see or do there, and anything the
passage gives about its significance. Plain, factual, specific.

{rules}

Return: {{"digest": "..."}}

DESTINATION: {name} ({country})
PASSAGE:
{passage}"""

QA_PROMPT = """You are preparing training data for a travel discovery assistant
covering West Africa.

From the passage below produce:
- 2 questions a traveller might ask that the passage DOES answer, each with a
  1-3 sentence answer drawn strictly from the passage.
- 1 question a traveller might reasonably ask that the passage does NOT answer,
  with an answer that says plainly the source does not cover it and names what
  the source does cover instead.

Vary the question wording; do not start every question the same way.

{rules}

Return: {{"pairs": [{{"question": "...", "answer": "...", "answerable": true}},
                    {{"question": "...", "answer": "...", "answerable": true}},
                    {{"question": "...", "answer": "...", "answerable": false}}]}}

DESTINATION: {name} ({country})
PASSAGE:
{passage}"""

DIALOGUE_PROMPT = """You are preparing training data for a travel discovery assistant
covering West Africa.

Write a realistic conversation of exactly 3 exchanges between a curious
Traveller and a knowledgeable Guide about the destination below. Each
Traveller turn should follow naturally from what the Guide just said. Each
Guide reply is 1-3 sentences drawn strictly from the passage. In the third
exchange the Traveller asks something practical the passage does NOT cover
(cost, opening hours, transport, safety); the Guide says plainly the source
does not cover it and points back to what it does cover.

{rules}

Return: {{"turns": [{{"traveller": "...", "guide": "..."}},
                    {{"traveller": "...", "guide": "..."}},
                    {{"traveller": "...", "guide": "...", "answerable": false}}]}}

DESTINATION: {name} ({country})
PASSAGE:
{passage}"""

DIALOGUE_INSTRUCTION = (
    "Continue the conversation as the guide. Answer the traveller's latest "
    "question using only the passage below. If the passage does not contain "
    "the answer, say so plainly."
)

DIGEST_INSTRUCTIONS = [
    "Write a short traveller's digest of this destination, 60-100 words, using only the passage.",
    "Summarise this destination for a traveller in 60-100 words. Use only what the passage states.",
    "Give me a traveller-facing summary of this place in about 80 words, grounded in the passage below.",
]


# --------------------------------------------------------------------------
# Model client
# --------------------------------------------------------------------------


def get_api_key(names=("GOOGLE_API_KEY", "GEMINI_API_KEY")) -> str:
    try:  # Colab secrets
        from google.colab import userdata  # type: ignore

        for k in names:
            try:
                v = userdata.get(k)
                if v:
                    return v
            except Exception:
                pass
    except Exception:
        pass
    for k in names:
        if os.environ.get(k):
            return os.environ[k]
    raise SystemExit(
        f"No API key found. In Colab add {names[0]} to Secrets and enable "
        f"notebook access, or export {names[0]} in your shell."
    )


class OpenAITeacher:
    """Same interface as Teacher, backed by the OpenAI Chat Completions API.

    Plain HTTP, so it needs no extra package. JSON mode is switched on: every
    prompt in this file asks for a JSON object back.
    """

    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str):
        self.key = get_api_key(("OPENAI_API_KEY",))
        self.model = model
        self.lock = threading.Lock()
        self.calls = 0

    def generate(self, prompt: str, retries: int = 4) -> str | None:
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode()
        delay = 2.0
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    self.URL,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {self.key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.load(r)
                with self.lock:
                    self.calls += 1
                return (data["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:  # rate limit, transient 5xx
                if attempt == retries - 1:
                    print(f"  giving up after {retries}: {exc}", file=sys.stderr)
                    return None
                time.sleep(delay)
                delay *= 2
        return None


class Teacher:
    def __init__(self, model: str):
        try:
            from google import genai
        except ImportError:
            raise SystemExit("pip install google-genai") from None
        self.client = genai.Client(api_key=get_api_key())
        self.model = model
        self.lock = threading.Lock()
        self.calls = 0

    def generate(self, prompt: str, retries: int = 4) -> str | None:
        delay = 2.0
        for attempt in range(retries):
            try:
                r = self.client.models.generate_content(model=self.model, contents=prompt)
                with self.lock:
                    self.calls += 1
                return (r.text or "").strip()
            except Exception as exc:  # rate limit, transient 5xx
                if attempt == retries - 1:
                    print(f"  giving up after {retries}: {exc}", file=sys.stderr)
                    return None
                time.sleep(delay)
                delay *= 2
        return None


RE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_json(text: str | None):
    if not text:
        return None
    try:
        return json.loads(RE_FENCE.sub("", text).strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class Cache:
    """Append-only JSONL cache keyed by (passage_id, task)."""

    def __init__(self, path: str):
        self.path = path
        self.seen: set[tuple[str, str]] = set()
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.rows.append(r)
                    self.seen.add((r["passage_id"], r["task"]))
        # Kept open for the whole run: every result is flushed as it arrives.
        self.fh = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def has(self, pid: str, task: str) -> bool:
        return (pid, task) in self.seen

    def put(self, pid: str, task: str, raw):
        with self.lock:
            rec = {"passage_id": pid, "task": task, "raw": raw}
            self.rows.append(rec)
            self.seen.add((pid, task))
            self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.fh.flush()


# --------------------------------------------------------------------------
# Pair construction from raw teacher output
# --------------------------------------------------------------------------


def stable_choice(seq, *parts):
    import hashlib

    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return seq[int(h[:8], 16) % len(seq)]


def digest_pair(row, raw) -> dict | None:
    obj = parse_json(raw)
    if not obj or obj.get("skip") or not obj.get("digest"):
        return None
    instr = stable_choice(DIGEST_INSTRUCTIONS, row["passage_id"], "digest")
    return {
        "input": f"{TASK_TAGS['digest']} {instr}\n\nPASSAGE:\n{row['passage']}",
        "output": str(obj["digest"]).strip(),
        "task": "digest",
        "subtype": "teacher_digest",
        "passage_id": row["passage_id"],
        "country": row["country"],
        "name": row["name"],
    }


def qa_pairs(row, raw) -> list[dict]:
    obj = parse_json(raw)
    if not obj or obj.get("skip"):
        return []
    out = []
    for p in obj.get("pairs", []) or []:
        q, a = str(p.get("question", "")).strip(), str(p.get("answer", "")).strip()
        if not q or not a:
            continue
        answerable = bool(p.get("answerable", True))
        out.append(
            {
                "input": qa_prompt(q, row["passage"]),
                "output": a,
                "task": "answer" if answerable else "refuse",
                "subtype": "teacher_qa" if answerable else "teacher_abstain",
                "passage_id": row["passage_id"],
                "country": row["country"],
                "name": row["name"],
            }
        )
    return out


def dialogue_pairs(row, raw) -> list[dict]:
    """One multi-turn conversation becomes one pair per guide turn.

    Each pair's input carries the conversation so far, so the model learns to
    answer follow-ups in context. The final, unanswerable turn is a refusal.
    """
    obj = parse_json(raw)
    if not obj or obj.get("skip"):
        return []
    turns = [
        t
        for t in (obj.get("turns") or [])
        if str(t.get("traveller", "")).strip() and str(t.get("guide", "")).strip()
    ]
    out, history = [], []
    for i, t in enumerate(turns):
        q, a = str(t["traveller"]).strip(), str(t["guide"]).strip()
        answerable = bool(t.get("answerable", True))
        convo = "\n".join(history) if history else "(start of conversation)"
        out.append(
            {
                "input": (
                    f"{TASK_TAGS['answer']} {DIALOGUE_INSTRUCTION}\n\n"
                    f"CONVERSATION SO FAR:\n{convo}\n\n"
                    f"QUESTION: {q}\n\nPASSAGE:\n{row['passage']}"
                ),
                "output": a,
                "task": "answer" if answerable else "refuse",
                "subtype": f"dialogue_turn{i + 1}" if answerable else "dialogue_abstain",
                "passage_id": row["passage_id"],
                "country": row["country"],
                "name": row["name"],
            }
        )
        history += [f"Traveller: {q}", f"Guide: {a}"]
    return out


PROMPTS = {"digest": DIGEST_PROMPT, "qa": QA_PROMPT, "dialogue": DIALOGUE_PROMPT}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def run(args):
    df = pd.read_json(args.passages, lines=True)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cache = Cache(os.path.join(args.out, "teacher_cache.jsonl"))

    # Over-generate: a quarter of the output will fail validation.
    n_digest = int(args.n_digest * 1.25)
    n_qa = int(args.n_qa * 1.25)
    n_dialogue = int(args.n_dialogue * 1.25)

    digest_rows = country_quota_sample(df, min(n_digest, len(df)), 0.22, rng)
    rich = df[df["passage"].str.len() >= 900]
    qa_rows = country_quota_sample(rich, min(n_qa, len(rich)), 0.22, rng)
    dialogue_rows = country_quota_sample(rich, min(n_dialogue, len(rich)), 0.22, rng)

    if args.limit:
        digest_rows = digest_rows.head(args.limit)
        qa_rows = qa_rows.head(args.limit)
        dialogue_rows = dialogue_rows.head(args.limit)

    jobs = (
        [("digest", r) for _, r in digest_rows.iterrows()]
        + [("qa", r) for _, r in qa_rows.iterrows()]
        + [("dialogue", r) for _, r in dialogue_rows.iterrows()]
    )
    jobs = [(t, r) for t, r in jobs if not cache.has(r["passage_id"], t)]
    print(f"{len(jobs)} teacher calls to make ({len(cache.seen)} already cached)")

    if args.dry_run:
        if jobs:
            t, r = jobs[0]
            tmpl = PROMPTS[t]
            print("\n--- example prompt ---\n")
            print(
                tmpl.format(
                    rules=RULES, name=r["name"], country=r["country"], passage=r["passage"][:600]
                )
            )
        return {"dry_run": True, "pending_calls": len(jobs)}

    teacher = (OpenAITeacher if args.provider == "openai" else Teacher)(args.model)
    done = [0]

    def work(job):
        task, row = job
        tmpl = PROMPTS[task]
        prompt = tmpl.format(
            rules=RULES, name=row["name"], country=row["country"], passage=row["passage"]
        )
        raw = teacher.generate(prompt)
        cache.put(row["passage_id"], task, raw)
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"  {done[0]}/{len(jobs)}")

    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, jobs))

    # ---- assemble and validate
    by_id = {r["passage_id"]: r for _, r in df.iterrows()}
    pairs, skipped = [], 0
    for rec in cache.rows:
        row = by_id.get(rec["passage_id"])
        if row is None:
            continue
        if rec["task"] == "digest":
            p = digest_pair(row, rec["raw"])
            if p:
                pairs.append(p)
            else:
                skipped += 1
        else:
            fn = dialogue_pairs if rec["task"] == "dialogue" else qa_pairs
            got = fn(row, rec["raw"])
            pairs.extend(got)
            if not got:
                skipped += 1

    kept, failures = [], {}
    for p in pairs:
        fails = validate(p)
        if fails:
            for f in fails:
                failures[f] = failures.get(f, 0) + 1
        else:
            kept.append(p)

    path = os.path.join(args.out, "pairs_teacher.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for p in kept:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    report = {
        "model": args.model,
        "api_calls_this_run": teacher.calls,
        "cached_total": len(cache.rows),
        "pairs_built": len(pairs),
        "unparseable_or_skipped": skipped,
        "validation_failures": failures,
        "pairs_kept": len(kept),
        "rejection_rate": round(1 - len(kept) / max(1, len(pairs)), 3),
        "by_task": pd.Series([p["task"] for p in kept]).value_counts().to_dict(),
    }
    with open(os.path.join(args.out, "teacher_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="out/passages.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    ap.add_argument(
        "--model",
        default=None,
        help=f"default: {DEFAULT_MODEL} (gemini) / {DEFAULT_OPENAI_MODEL} (openai)",
    )
    ap.add_argument("--n-digest", type=int, default=1000)
    ap.add_argument("--n-qa", type=int, default=700)
    ap.add_argument("--n-dialogue", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N of each")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args()
    a.model = a.model or (DEFAULT_OPENAI_MODEL if a.provider == "openai" else DEFAULT_MODEL)
    print(json.dumps(run(a), indent=2, ensure_ascii=False))
