"""One trace record per request: every step, its latency, tokens and cost.

Records go to an append-only JSONL file (easy to grep and diff) and to SQLite
(easy to query, and what the /trace/{id} endpoint reads).
"""

import atexit
import json
import queue
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from recon.config import settings


class Tracer:
    def __init__(self, question: str, model: str, agent_version: str, request_id: str | None = None):
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.question = question
        self.model = model
        self.agent_version = agent_version
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._t0 = time.perf_counter()
        self.steps: list[dict] = []

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    def _add(self, kind: str, **fields) -> None:
        self.steps.append({"i": len(self.steps), "kind": kind, **fields})

    def llm(self, latency_ms: float, prompt_tokens: int, completion_tokens: int,
            tool_calls: list[str], retries: int = 0, error: str | None = None, wait_ms: float = 0.0) -> None:
        # latency_ms includes wait_ms, the time queued in the client-side rate limiter
        self._add("llm", latency_ms=round(latency_ms, 1), wait_ms=round(wait_ms, 1), prompt_tokens=prompt_tokens,
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
            "agent_version": self.agent_version,
            "outcome": outcome,  # answered | refused | step_limit | timeout | error
            "answer": answer,
            "latency_ms": round(self.elapsed_s() * 1000, 1),
            "llm_calls": len(llm_steps),
            "limiter_wait_ms": round(sum(s.get("wait_ms", 0) for s in llm_steps), 1),
            "tool_calls": sum(1 for s in self.steps if s["kind"] == "tool"),
            "prompt_tokens": sum(s["prompt_tokens"] for s in llm_steps),
            "completion_tokens": sum(s["completion_tokens"] for s in llm_steps),
            "cost_usd": round(sum(s["cost_usd"] for s in llm_steps), 8),
            "steps": self.steps,
        }


class TraceStore:
    """Trace persistence, off the request path.

    save() only enqueues. One writer thread drains the queue in batches over a single
    SQLite connection (one commit per batch) and appends the same batch to the JSONL file.
    Records stay readable through get() while they wait in the queue.
    Trade-off: on a hard crash, records still queued are lost.
    """

    _STOP = object()

    def __init__(self, directory: Path = settings.trace_dir, batch_size: int = 200, max_queue: int = 10_000):
        directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = directory / "traces.jsonl"
        self.db_path = directory / "traces.db"
        self.batch_size = batch_size
        self.write_errors = 0
        with closing(self._connect()) as con, con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""CREATE TABLE IF NOT EXISTS traces (
                request_id TEXT PRIMARY KEY, started_at TEXT, question TEXT, model TEXT,
                agent_version TEXT, outcome TEXT, status TEXT, latency_ms REAL, llm_calls INTEGER,
                prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, record TEXT)""")
        # Bounded: if the writer falls behind, save() blocks (backpressure) instead of growing memory.
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._pending: dict[str, dict] = {}
        self._pending_lock = threading.Lock()
        self._writer = threading.Thread(target=self._run, name="trace-writer", daemon=True)
        self._writer.start()
        atexit.register(self.close)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def save(self, record: dict) -> None:
        with self._pending_lock:
            self._pending[record["request_id"]] = record
        self._queue.put(record)

    def get(self, request_id: str) -> dict | None:
        with self._pending_lock:
            if request_id in self._pending:
                return self._pending[request_id]
        with closing(self._connect()) as con:
            row = con.execute("SELECT record FROM traces WHERE request_id = ?", (request_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def flush(self) -> None:
        """Block until everything saved so far is on disk."""
        self._queue.join()

    def close(self) -> None:
        if self._writer.is_alive():
            self._queue.put(self._STOP)
            self._writer.join(timeout=30)

    def _run(self) -> None:
        con = self._connect()  # used only by this thread
        stop = False
        while not stop:
            batch = [self._queue.get()]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            stop = any(r is self._STOP for r in batch)
            records = [r for r in batch if r is not self._STOP]
            try:
                if records:
                    self._write(con, records)
            except Exception as e:  # keep the writer alive; a lost trace must not stop the service
                self.write_errors += 1
                print(f"trace writer: {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                with self._pending_lock:
                    for r in records:
                        self._pending.pop(r["request_id"], None)
                for _ in batch:
                    self._queue.task_done()
        con.close()

    def _write(self, con: sqlite3.Connection, records: list[dict]) -> None:
        lines = [json.dumps(r, ensure_ascii=False, default=str) for r in records]
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
        with con:
            con.executemany(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(r["request_id"], r["started_at"], r["question"], r["model"], r["agent_version"], r["outcome"],
                  (r["answer"] or {}).get("status"), r["latency_ms"], r["llm_calls"], r["prompt_tokens"],
                  r["completion_tokens"], r["cost_usd"], line) for r, line in zip(records, lines)])
