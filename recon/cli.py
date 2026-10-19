"""Ask the agent one question from the terminal and print the answer and its trace.

    python -m recon.cli "Has supplier Dupont's March invoice been paid, and does the bank amount match?"
    python -m recon.cli --mock "Is invoice DUP-2026-0003 from Dupont fully paid?"
"""

import argparse
import json

from recon.agent import Agent
from recon.llm import make_model
from recon.tracing import TraceStore


def print_trace(record: dict) -> None:
    for s in record["steps"]:
        if s["kind"] == "llm":
            line = f"llm     {s['latency_ms']:>7.0f} ms  {s['prompt_tokens']:>5}+{s['completion_tokens']:<4} tok  -> {', '.join(s['tool_calls']) or '(no tool call)'}"
            if s.get("error"):
                line += f"  ERROR {s['error']}"
        elif s["kind"] == "tool":
            line = f"tool    {s['latency_ms']:>7.1f} ms  {s['tool']}({json.dumps(s['args'], ensure_ascii=False)})\n" \
                   f"                    {s['error'] or s['summary']}"
        elif s["kind"] == "verify":
            line = "verify  " + ("ok" if s["ok"] else "REJECTED: " + " | ".join(s["problems"]))
        else:
            line = f"{s['kind']:<8}{s.get('error', '')}"
        print(f"  [{s['i']}] {line}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--mock", action="store_true", help="use the offline mock model")
    args = p.parse_args()

    agent = Agent(model=make_model("mock" if args.mock else "mistral"), store=TraceStore())
    record = agent.ask(args.question)
    print(f"request {record['request_id']}  outcome={record['outcome']}  {record['latency_ms']:.0f} ms  "
          f"{record['prompt_tokens'] + record['completion_tokens']} tokens  ${record['cost_usd']:.6f}")
    print_trace(record)
    print(json.dumps(record["answer"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
