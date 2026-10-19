"""Run the agent on the ground-truth questions and score it.

    python -m recon.evaluate                # real model, one request at a time
    python -m recon.evaluate --mock         # offline mock model
    python -m recon.evaluate --version v1   # reproduce the v1 baseline
    python -m recon.evaluate --limit 10 --workers 2

Writes eval/results/<model>_<prompt>_<timestamp>.json (every question, its answer and
trace id) and prints a summary table.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from recon.agent import Agent
from recon.config import settings
from recon.llm import make_model
from recon.tracing import TraceStore

TOL = 0.01


def load_questions(limit: int | None = None) -> list[dict]:
    with open(settings.questions_path, encoding="utf-8") as f:
        qs = [json.loads(line) for line in f]
    return qs[:limit] if limit else qs


def tools_called(record: dict) -> list[str]:
    return [s["tool"] for s in record["steps"] if s["kind"] == "tool" and not s.get("error")]


def classify(expected: dict, record: dict) -> str | None:
    """None if the answer is correct, otherwise the first failure category that applies."""
    if record["outcome"] in ("error", "timeout", "step_limit"):
        return f"run_{record['outcome']}"
    a = record["answer"]
    if a["status"] == "insufficient_evidence":
        return "refused"
    if expected["status"] == "not_found":
        return None if a["status"] == "not_found" else "false_match"       # answered about the wrong invoice
    if a["status"] == "not_found":
        return "missed_invoice"                                             # retrieval failed
    if a["invoice_number"] != expected["invoice_number"]:
        return "wrong_invoice"                                              # retrieved a different invoice
    if "query_ledger" not in tools_called(record):
        return "skipped_verification"
    if a["paid_amount"] is None or abs(a["paid_amount"] - expected["paid_amount"]) > TOL:
        return "overcounted_payment" if (a["paid_amount"] or 0) > expected["paid_amount"] else "missed_payment"
    if a["status"] != expected["status"]:
        return "wrong_status"
    return None


def score(q: dict, record: dict) -> dict:
    exp = q["expected"]
    needed = {"search_invoices"} | ({"query_ledger"} if exp["status"] != "not_found" else set())
    verify = [s for s in record["steps"] if s["kind"] == "verify"]
    failure = classify(exp, record)
    return {
        "qid": q["qid"], "category": q["category"], "traps": q["traps"], "question": q["question"],
        "expected": exp, "answer": record["answer"], "outcome": record["outcome"],
        "correct": failure is None, "failure": failure,
        "right_tools": needed <= set(tools_called(record)),
        "llm_calls": record["llm_calls"], "tool_calls": record["tool_calls"],
        "verifier_rejections": sum(1 for v in verify if not v["ok"]),
        "verifier_problems": [p for v in verify for p in v["problems"]],
        "latency_ms": record["latency_ms"], "cost_usd": record["cost_usd"],
        "tokens": record["prompt_tokens"] + record["completion_tokens"],
        "request_id": record["request_id"],
    }


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    lat = [r["latency_ms"] for r in rows]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["correct"])
    return {
        "questions": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "right_tools": sum(r["right_tools"] for r in rows) / n,
        "mean_llm_calls": statistics.mean(r["llm_calls"] for r in rows),
        "answers_rejected_by_verifier": sum(1 for r in rows if r["verifier_rejections"]),
        "latency_p50_ms": pct(lat, 0.5), "latency_p95_ms": pct(lat, 0.95),
        "cost_per_1k_requests_usd": 1000 * sum(r["cost_usd"] for r in rows) / n,
        "mean_tokens": statistics.mean(r["tokens"] for r in rows),
        "failures": dict(Counter(r["failure"] for r in rows if r["failure"]).most_common()),
        "accuracy_by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
    }


def print_summary(s: dict, header: str) -> None:
    print(f"\n{header}")
    print(f"  accuracy          {s['accuracy']:.1%} of {s['questions']}")
    print(f"  right tools       {s['right_tools']:.1%}")
    print(f"  model calls/q     {s['mean_llm_calls']:.2f}")
    print(f"  verifier caught   {s['answers_rejected_by_verifier']} answer(s) on first submission")
    print(f"  latency p50/p95   {s['latency_p50_ms']:.0f} / {s['latency_p95_ms']:.0f} ms")
    print(f"  cost / 1k req     ${s['cost_per_1k_requests_usd']:.3f}")
    print(f"  failures          {s['failures'] or 'none'}")
    print("  by category       " + ", ".join(f"{c} {v}" for c, v in s["accuracy_by_category"].items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--version", default=settings.agent_version, help="agent version (v1 to v4)")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1, help="concurrent requests (keep low on a free tier)")
    args = p.parse_args()

    model = make_model("mock" if args.mock else "mistral")
    agent = Agent(model=model, store=TraceStore(), version=args.version)
    questions = load_questions(args.limit)

    def one(q):
        row = score(q, agent.ask(q["question"]))
        mark = "ok " if row["correct"] else "BAD"
        print(f"{mark} {q['qid']} {q['category']:<14} {row['failure'] or '':<20} {row['latency_ms']:>7.0f} ms  {q['question']}",
              flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, questions))

    summary = summarize(rows)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = settings.eval_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{settings.dataset}_{model.name}_{agent.version}_{stamp}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": settings.dataset, "model": model.name, "agent_version": agent.version, "workers": args.workers,
                   "summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)
    print_summary(summary, f"{settings.dataset} set / {model.name} / agent {agent.version}  ->  {out.relative_to(settings.eval_dir.parent)}")


if __name__ == "__main__":
    main()
