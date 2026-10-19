import json

import pytest

from recon.config import settings
from recon.data_gen import ensure_data
from recon.llm import ModelResponse, ToolCallRequest


@pytest.fixture(scope="session", autouse=True)
def dataset():
    ensure_data()
    with open(settings.data_dir / "truth.json", encoding="utf-8") as f:
        return json.load(f)


def first_invoice(truth, scenario):
    return next(t for t in truth if t["scenario"] == scenario and not t["traps"][1:])


_n = 0


def call(name: str, args) -> ModelResponse:
    """A model turn that makes one tool call; args may be a dict or a raw (possibly broken) string."""
    global _n
    _n += 1
    raw = args if isinstance(args, str) else json.dumps(args)
    return ModelResponse(content="", tool_calls=[ToolCallRequest(f"call{_n:05d}", name, raw)],
                         prompt_tokens=100, completion_tokens=20)


def answer_for(t: dict, **overrides) -> dict:
    """A correct final_answer for a truth record, with optional wrong fields."""
    inv = t["invoice"]
    a = {"status": t["status"], "invoice_number": inv["invoice_number"], "invoice_amount_ttc": inv["amount_ttc"],
         "paid_amount": t["paid_amount"], "amount_matches": t["status"] == "paid",
         "evidence_transaction_ids": [p["txn_id"] for p in t["payments"]], "explanation": "test"}
    return {**a, **overrides}
