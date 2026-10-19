"""The agent's control flow, driven by scripted model turns (no API calls)."""

from conftest import answer_for, call, first_invoice

from recon.agent import Agent
from recon.llm import MockModel, ModelResponse, ScriptedModel
from recon.tracing import TraceStore


def run(turns, **kw):
    model = ScriptedModel(turns)
    return Agent(model=model, **kw).ask("question"), model


def happy_path(t, final=None):
    num = t["invoice"]["invoice_number"]
    return [call("search_invoices", {"query": num}),
            call("query_ledger", {"label_contains": num}),
            call("final_answer", final or answer_for(t))]


def kinds(record):
    return [s["kind"] for s in record["steps"]]


def test_happy_path(dataset):
    t = first_invoice(dataset, "split_payment")
    rec, _ = run(happy_path(t))
    assert rec["outcome"] == "answered"
    assert rec["answer"]["status"] == "paid"
    assert kinds(rec) == ["llm", "tool", "llm", "tool", "llm", "verify"]
    assert rec["prompt_tokens"] == 300 and rec["cost_usd"] > 0


def test_recovers_from_malformed_tool_call(dataset):
    t = first_invoice(dataset, "paid_exact")
    bad = call("search_invoices", '{"query": "Dupont", "issue_month": "March"}')
    rec, model = run([bad] + happy_path(t))
    assert rec["outcome"] == "answered"
    assert "issue_month" in rec["steps"][1]["error"]
    # the model was shown the validation error before its retry
    assert "issue_month" in model.calls[1][-1]["content"]


def test_two_malformed_calls_in_a_row_abort(dataset):
    bad = call("search_invoices", "not json")
    rec, _ = run([bad, call("search_invoices", "{}")])
    assert rec["outcome"] == "error" and rec["answer"] is None


def test_prose_reply_gets_one_nudge(dataset):
    t = first_invoice(dataset, "paid_exact")
    rec, model = run([ModelResponse(content="It was paid.")] + happy_path(t))
    assert rec["outcome"] == "answered"
    assert "tool call only" in model.calls[1][-1]["content"]


def test_made_up_amount_is_rejected_then_fixed(dataset):
    t = first_invoice(dataset, "paid_exact")
    wrong = answer_for(t, invoice_amount_ttc=1234.56)
    rec, _ = run(happy_path(t, final=wrong) + [call("final_answer", answer_for(t))])
    verify = [s for s in rec["steps"] if s["kind"] == "verify"]
    assert [v["ok"] for v in verify] == [False, True]
    assert "differs from the document" in verify[0]["problems"][0]
    assert rec["outcome"] == "answered"


def test_repeated_bad_answer_becomes_a_refusal(dataset):
    t = first_invoice(dataset, "paid_exact")
    wrong = answer_for(t, evidence_transaction_ids=["TXN-99999"])
    rec, _ = run(happy_path(t, final=wrong) + [call("final_answer", wrong)])
    assert rec["outcome"] == "refused"
    assert rec["answer"]["status"] == "insufficient_evidence"


def test_answer_without_checking_ledger_is_rejected(dataset):
    t = first_invoice(dataset, "paid_exact")
    num = t["invoice"]["invoice_number"]
    rec, _ = run([call("search_invoices", {"query": num}), call("final_answer", answer_for(t)),
                  call("final_answer", answer_for(t))])
    problems = next(s for s in rec["steps"] if s["kind"] == "verify")["problems"]
    assert any("without calling query_ledger" in p for p in problems)
    assert rec["outcome"] == "refused"


def test_vat_only_payment_cannot_be_called_paid(dataset):
    t = first_invoice(dataset, "vat_only")
    claim = answer_for(t, status="paid", amount_matches=True)
    rec, _ = run(happy_path(t, final=claim) + [call("final_answer", answer_for(t))])
    first = next(s for s in rec["steps"] if s["kind"] == "verify")
    assert any("expected partially_paid" in p for p in first["problems"])
    assert rec["answer"]["status"] == "partially_paid"


def test_step_limit(dataset):
    turns = [call("query_ledger", {"counterparty": "DUPONT"}) for _ in range(5)]
    rec, _ = run(turns, max_steps=3)
    assert rec["outcome"] == "step_limit" and rec["llm_calls"] == 3


def test_traces_are_stored_and_retrievable(tmp_path):
    store = TraceStore(tmp_path)
    rec = Agent(model=MockModel(), store=store).ask("Is invoice DUP-2026-0001 from Dupont fully paid?")
    assert store.get(rec["request_id"])["steps"] == rec["steps"]
    assert (tmp_path / "traces.jsonl").read_text(encoding="utf-8").count("\n") == 1
