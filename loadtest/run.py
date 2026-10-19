"""Start the API, load it with Locust at several concurrency levels, write a results table.

    python -m loadtest.run --llm mock                      # service overhead only
    python -m loadtest.run --llm mock --mock-latency-ms 800  # mock that waits like a model
    python -m loadtest.run --llm mistral --duration 60     # end to end, provider limits included

Results go to loadtest/results/<llm>_<timestamp>.{json,md}.
"""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthy(url: str, timeout_s: float = 60) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                return json.load(r)
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("API did not become healthy")


def run_level(url: str, users: int, duration_s: int, out_prefix: Path) -> dict:
    # Locust exits with status 1 when any request failed; failures are results, not a crash.
    subprocess.run([sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--headless",
                    "--host", url, "--users", str(users), "--spawn-rate", str(users),
                    "--run-time", f"{duration_s}s", "--csv", str(out_prefix), "--only-summary"],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(f"{out_prefix}_stats.csv", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    with open(f"{out_prefix}_failures.csv", encoding="utf-8") as f:
        failures = sorted(csv.DictReader(f), key=lambda r: -int(r["Occurrences"]))
    n, fails = int(row["Request Count"]), int(row["Failure Count"])
    return {"users": users, "requests": n, "rps": round(float(row["Requests/s"]), 2),
            "p50_ms": float(row["50%"]), "p95_ms": float(row["95%"]),
            "error_rate": round(fails / n, 4) if n else None,
            "top_failure": failures[0]["Error"] if failures else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["mock", "mistral"], default="mock")
    p.add_argument("--users", default="1,5,10,20")
    p.add_argument("--duration", type=int, default=30, help="seconds per concurrency level")
    p.add_argument("--mock-latency-ms", type=int, default=0)
    args = p.parse_args()

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "RECON_LLM": args.llm, "RECON_MOCK_LATENCY_MS": str(args.mock_latency_ms)}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "results" / f"{args.llm}_{stamp}"
    out_dir.mkdir(parents=True)

    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "recon.api:app", "--port", str(port),
                               "--log-level", "warning"], cwd=ROOT, env=env)
    try:
        health = wait_healthy(url)
        rows = []
        for users in map(int, args.users.split(",")):
            row = run_level(url, users, args.duration, out_dir / f"u{users}")
            rows.append(row)
            print(f"users {users:>3}  {row['requests']:>6} req  {row['rps']:>7.2f} req/s  "
                  f"p50 {row['p50_ms']:>7.0f} ms  p95 {row['p95_ms']:>7.0f} ms  errors {row['error_rate']:.1%}", flush=True)
    finally:
        server.terminate()
        server.wait(timeout=10)

    meta = {"llm": args.llm, "model": health["model"], "agent_version": health["agent_version"],
            "mock_latency_ms": args.mock_latency_ms, "duration_s": args.duration, "server": "uvicorn, 1 process",
            "machine": f"{os.cpu_count()} logical CPUs, {sys.platform}", "rows": rows}
    (out_dir / "results.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    lines = ["| users | requests | req/s | P50 (ms) | P95 (ms) | error rate |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['users']} | {r['requests']} | {r['rps']} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['error_rate']:.1%} |"
              for r in rows]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{meta['model']} / agent {meta['agent_version']}, {args.duration}s per level -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
