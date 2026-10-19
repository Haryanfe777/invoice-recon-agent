"""One trace record per request: every step, its latency, tokens and cost.

Records go to an append-only JSONL file (easy to grep and diff) and to SQLite
(easy to query, and what the /trace/{id} endpoint reads).
"""

import json
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from recon.config import settings


class Tracer:
    def __init__(self, question: str, model: str, prompt_version: str, request_id: str | None = None):
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.question = question
        self.model = model
        self.prompt_version = prompt_version
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._t0 = time.perf_counter()
        self.steps: list[dict] = []

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    def _add(self, kind: str, **fields) -> None:
        self.steps.append({"i": len(self.steps), "kind": kind, **fields})

    def llm(self, latency_ms: float, prompt_tokens: int, completion_tokens: int,
            tool_calls: list[str], retries: int = 0, error: str | None = None) -> None:
        self._add("llm", latency_ms=round(latency_ms, 1), prompt_tokens=prompt_tokens,
                  completion_tokens=completion_tokens,
                  cost_usd=settings.cost_usd(prompt_tokens, completion_tokens),
                  tool_calls=tool_calls, retries=retries, error=error)

    def tool(self, name: str, args: dict | str, latency_ms: float, summary: str, error: str | None = None) -> None:
        self._add("tool", tool=name, args=args, latency_ms=round(latency_ms, 1), summary=summary, error=error)

    def verify(self, ok: bool, problems: list[str]) -> None:
        self._add("verify", ok=ok, problems=problems)

    def finish(self, outcome: str, answer: dict | None) -> dict:
        llm_steps = [s for s in self.steps if s["kind"] == "llm"]
        return {
            "request_id": self.request_id,
            "started_at": self.started_at,
            "question": self.question,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "outcome": outcome,  # answered | refused | step_limit | timeout | error
            "answer": answer,
            "latency_ms": round(self.elapsed_s() * 1000, 1),
            "llm_calls": len(llm_steps),
            "tool_calls": sum(1 for s in self.steps if s["kind"] == "tool"),
            "prompt_tokens": sum(s["prompt_tokens"] for s in llm_steps),
            "completion_tokens": sum(s["completion_tokens"] for s in llm_steps),
            "cost_usd": round(sum(s["cost_usd"] for s in llm_steps), 8),
            "steps": self.steps,
        }


class TraceStore:
    def __init__(self, directory: Path = settings.trace_dir):
        directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = directory / "traces.jsonl"
        self.db_path = directory / "traces.db"
        self._lock = threading.Lock()
        with closing(self._connect()) as con, con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""CREATE TABLE IF NOT EXISTS traces (
                request_id TEXT PRIMARY KEY, started_at TEXT, question TEXT, model TEXT,
                prompt_version TEXT, outcome TEXT, status TEXT, latency_ms REAL, llm_calls INTEGER,
                prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, record TEXT)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def save(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            with closing(self._connect()) as con, con:
                con.execute(
                    "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (record["request_id"], record["started_at"], record["question"], record["model"],
                     record["prompt_version"], record["outcome"], (record["answer"] or {}).get("status"),
                     record["latency_ms"], record["llm_calls"], record["prompt_tokens"],
                     record["completion_tokens"], record["cost_usd"], line))

    def get(self, request_id: str) -> dict | None:
        with closing(self._connect()) as con, con:
            row = con.execute("SELECT record FROM traces WHERE request_id = ?", (request_id,)).fetchone()
        return json.loads(row[0]) if row else None
