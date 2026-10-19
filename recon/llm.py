"""Chat models behind one small interface: the real Mistral API, and offline stand-ins.

MockModel lets the API and load tests measure the service's own overhead without
spending tokens; ScriptedModel replays fixed responses for unit tests.
"""

import json
import random
import re
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from recon.config import settings


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass
class ModelResponse:
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0  # provider-side retries (rate limits, 5xx) before this response
    wait_ms: float = 0.0  # time spent queued in the client-side rate limiter


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse: ...


class ModelError(Exception):
    """The provider call failed for good (after retries)."""


def _call_id() -> str:
    # Mistral tool-call ids are 9 alphanumeric characters.
    return "".join(random.choices(string.ascii_letters + string.digits, k=9))


class RateLimiter:
    """Thread-safe token bucket that queues callers instead of letting them hit a 429.

    Each acquire() reserves the next free slot under the lock, then sleeps until it comes up.
    Reservations are handed out in arrival order, so waiting callers never stampede the
    provider together. A caller whose slot is further away than its timeout gets its
    reservation back and an error, rather than waiting past its request deadline.
    """

    def __init__(self, rate_per_s: float, burst: float = 1.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self.rate, self.capacity = rate_per_s, max(1.0, burst)
        self.clock, self.sleep = clock, sleep
        self.tokens, self.updated = self.capacity, clock()
        self._lock = threading.Lock()

    def acquire(self, timeout_s: float | None = None) -> float:
        """Wait for a slot; return the seconds waited. Raises TimeoutError past timeout_s."""
        with self._lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            self.tokens -= 1  # may go negative: that is a reservation for a future slot
            wait = max(0.0, -self.tokens / self.rate)
            if timeout_s is not None and wait > timeout_s:
                self.tokens += 1
                raise TimeoutError(f"rate limiter queue is {wait:.1f}s long, over the {timeout_s:.1f}s budget")
        if wait:
            self.sleep(wait)
        return wait


_limiters: dict[float, RateLimiter] = {}
_limiters_lock = threading.Lock()


def shared_limiter(rate_per_s: float) -> RateLimiter:
    """One limiter per rate for the whole process: the provider quota is per API key, not per thread."""
    with _limiters_lock:
        if rate_per_s not in _limiters:
            _limiters[rate_per_s] = RateLimiter(rate_per_s)
        return _limiters[rate_per_s]


class MistralModel:
    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(self, model: str = settings.model, api_key: str = settings.api_key,
                 timeout_s: float = settings.llm_timeout_s, max_retries: int = 3,
                 limiter: RateLimiter | None = None):
        from mistralai.client import Mistral  # imported lazily so offline runs need no key

        if not api_key:
            raise ModelError("MISTRAL_API_KEY is not set (copy .env.example to .env)")
        self.name = model
        self.client = Mistral(api_key=api_key, timeout_ms=int(timeout_s * 1000))
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.limiter = limiter or (shared_limiter(settings.llm_rps) if settings.llm_rps > 0 else None)

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        retries, waited = 0, 0.0
        while True:
            if self.limiter:  # every attempt, retries included, spends quota
                try:
                    waited += self.limiter.acquire(timeout_s=self.timeout_s)
                except TimeoutError as e:
                    raise ModelError(f"rate limited locally: {e}") from e
            try:
                res = self.client.chat.complete(
                    model=self.name, messages=messages, tools=tools,
                    tool_choice="any",  # every turn must be a tool call; the answer is a tool too
                    parallel_tool_calls=False, temperature=0.0)
                break
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status in self.RETRYABLE and retries < self.max_retries:
                    retries += 1
                    # Full jitter: spread retries over the whole window so threads do not retry in lockstep.
                    time.sleep(random.uniform(0, min(8.0, 0.5 * 2 ** retries)))
                    continue
                raise ModelError(f"{type(e).__name__} (status {status}): {e}") from e

        msg = res.choices[0].message
        calls = [ToolCallRequest(id=tc.id or _call_id(), name=tc.function.name,
                                 arguments=tc.function.arguments if isinstance(tc.function.arguments, str)
                                 else json.dumps(tc.function.arguments))
                 for tc in (msg.tool_calls or [])]
        content = msg.content if isinstance(msg.content, str) else ""
        return ModelResponse(content=content or "", tool_calls=calls,
                             prompt_tokens=res.usage.prompt_tokens or 0,
                             completion_tokens=res.usage.completion_tokens or 0, retries=retries,
                             wait_ms=waited * 1000)


def _approx_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) + len(json.dumps(m.get("tool_calls", []))) for m in messages) // 4


class MockModel:
    """A deterministic, rule-based stand-in that drives the real tools.

    It follows the happy path (search, check ledger, answer) and makes no attempt to
    handle the traps. Its job is to generate realistic traffic, not to be accurate.
    """

    name = "mock"
    MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june",
                                          "july", "august", "september", "october", "november", "december"], 1)}

    def __init__(self, latency_ms: int = settings.mock_latency_ms):
        self.latency_ms = latency_ms

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        question = next(m["content"] for m in messages if m["role"] == "user")
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        name, args = self._next_call(question, results)
        return ModelResponse(content="", tool_calls=[ToolCallRequest(_call_id(), name, json.dumps(args))],
                             prompt_tokens=_approx_tokens(messages), completion_tokens=40)

    def _next_call(self, question: str, results: list[dict]) -> tuple[str, dict]:
        if not results:
            args = {"query": question}
            number = re.search(r"[A-Z]{3}-\d{4}-\d{4}", question)
            month = next((i for m, i in self.MONTHS.items() if m in question.lower()), None)
            if number:
                args["query"] = number.group()
            elif month:
                args["issue_month"] = f"2026-{month:02d}"
            return "search_invoices", args

        invoices = results[0].get("results", [])
        if not invoices:
            return "final_answer", {"status": "not_found", "explanation": "No matching invoice found."}
        inv = invoices[0]
        if len(results) == 1:
            return "query_ledger", {"counterparty": inv["supplier"], "date_from": inv["issue_date"]}

        # Naive matching: a payment counts if it cites the invoice number or equals the TTC total.
        txns = [t for t in results[-1].get("transactions", [])
                if inv["invoice_number"] in t["label"] or abs(t["amount"] - inv["amount_ttc"]) < 0.01]
        paid = round(sum(t["amount"] for t in txns), 2)
        status = "unpaid" if not txns else ("paid" if abs(paid - inv["amount_ttc"]) < 0.01 else "partially_paid")
        return "final_answer", {
            "status": status, "invoice_number": inv["invoice_number"], "invoice_amount_ttc": inv["amount_ttc"],
            "paid_amount": paid, "amount_matches": status == "paid",
            "evidence_transaction_ids": [t["txn_id"] for t in txns],
            "explanation": f"{len(txns)} matching payment(s), total {paid} EUR.",
        }


class ScriptedModel:
    """Replays a fixed list of responses; for tests."""

    name = "scripted"

    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


def make_model(kind: str = settings.llm) -> ChatModel:
    if kind == "mock":
        return MockModel()
    if kind == "mistral":
        return MistralModel()
    raise ValueError(f"unknown RECON_LLM '{kind}' (expected 'mistral' or 'mock')")
