"""Chat models behind one small interface: the real Mistral API, and offline stand-ins.

MockModel lets the API and load tests measure the service's own overhead without
spending tokens; ScriptedModel replays fixed responses for unit tests.
"""

import json
import random
import re
import string
import time
from dataclasses import dataclass, field
from typing import Protocol

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


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse: ...


class ModelError(Exception):
    """The provider call failed for good (after retries)."""


def _call_id() -> str:
    # Mistral tool-call ids are 9 alphanumeric characters.
    return "".join(random.choices(string.ascii_letters + string.digits, k=9))


class MistralModel:
    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(self, model: str = settings.model, api_key: str = settings.api_key,
                 timeout_s: float = settings.llm_timeout_s, max_retries: int = 3):
        from mistralai.client import Mistral  # imported lazily so offline runs need no key

        if not api_key:
            raise ModelError("MISTRAL_API_KEY is not set (copy .env.example to .env)")
        self.name = model
        self.client = Mistral(api_key=api_key, timeout_ms=int(timeout_s * 1000))
        self.max_retries = max_retries

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        retries = 0
        while True:
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
                    time.sleep(min(8.0, 0.5 * 2 ** retries) + random.random() * 0.25)
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
                             completion_tokens=res.usage.completion_tokens or 0, retries=retries)


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
            return "query_ledger", {"label_contains": inv["invoice_number"]}

        txns = results[-1].get("transactions", [])
        paid = round(sum(t["amount"] for t in txns), 2)
        status = "unpaid" if not txns else ("paid" if abs(paid - inv["amount_ttc"]) < 0.01 else "partially_paid")
        return "final_answer", {
            "status": status, "invoice_number": inv["invoice_number"], "invoice_amount_ttc": inv["amount_ttc"],
            "paid_amount": paid, "amount_matches": status == "paid",
            "evidence_transaction_ids": [t["txn_id"] for t in txns],
            "explanation": f"{len(txns)} payment(s) referencing {inv['invoice_number']}, total {paid} EUR.",
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
