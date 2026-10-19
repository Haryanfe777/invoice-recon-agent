"""The agent's control flow, driven by scripted model turns (no API calls)."""

from conftest import answer_for, call, first_invoice

from recon.agent import Agent
from recon.llm import MockModel, ModelResponse, ScriptedModel
from recon.tools import QueryLedgerInput, query_ledger
from recon.tracing import TraceStore


def question_for(t) -> str:
    inv = t["invoice"]
    return f"Is invoice {inv['invoice_number']} from {inv['supplier']} fully paid?"


def run(turns, t=None, question=None, **kw):
    model = ScriptedModel(turns)
    q = question or (question_for(t) if t else "Is anything owed?")
    return Agent(model=model, **kw).ask(q), model


def happy_path(t, final=None):
    inv = t["invoice"]
    return [call("search_invoices", {"query": inv["invoice_number"]}),
            call("query_ledger", {"counterparty": inv["supplier"], "date_from": inv["issue_date"]}),
            call("final_answer", final or answer_for(t))]


def label_only_path(t):
    num = t["invoice"]["invoice_number"]
    return [call("search_invoices", {"query": num}), call("query_ledger", {"label_contains": num})]


def kinds(record):
    return [s["kind"] for s in record["steps"]]


def verify_steps(record):
    return [s for s in record["steps"] if s["kind"] == "verify"]


def test_happy_path(dataset):
    t = first_invoice(dataset, "split_payment")
    rec, _ = run(happy_path(t), t)
    assert rec["outcome"] == "answered"
    assert rec["answer"]["status"] == "paid"
    assert kinds(rec) == ["llm", "tool", "llm", "tool", "llm", "verify"]
    assert rec["prompt_tokens"] == 300 and rec["cost_usd"] > 0


def test_recovers_from_malformed_tool_call(dataset):
    t = first_invoice(dataset, "paid_exact")
    bad = call("search_invoices", '{"query": "Dupont", "issue_month": "March"}')
    rec, model = run([bad] + happy_path(t), t)
    assert rec["outcome"] == "answered"
    assert "issue_month" in rec["steps"][1]["error"]
    # the model was shown the validation error before its retry
    assert "issue_month" in model.calls[1][-1]["content"]


def test_two_malformed_calls_in_a_row_abort():
    bad = call("search_invoices", "not json")
    rec, _ = run([bad, call("search_invoices", "{}")])
    assert rec["outcome"] == "error" and rec["answer"] is None


def test_prose_reply_gets_one_nudge(dataset):
    t = first_invoice(dataset, "paid_exact")
    rec, model = run([ModelResponse(content="It was paid.")] + happy_path(t), t)
    assert rec["outcome"] == "answered"
    assert "tool call only" in model.calls[1][-1]["content"]


def test_made_up_amount_is_rejected_then_fixed(dataset):
    t = first_invoice(dataset, "paid_exact")
    wrong = answer_for(t, invoice_amount_ttc=1234.56)
    rec, _ = run(happy_path(t, final=wrong) + [call("final_answer", answer_for(t))], t)
    verify = verify_steps(rec)
    assert [v["ok"] for v in verify] == [False, True]
    assert "differs from the document" in verify[0]["problems"][0]
    assert rec["outcome"] == "answered"


def test_repeated_bad_answer_becomes_a_refusal(dataset):
    t = first_invoice(dataset, "paid_exact")
    wrong = answer_for(t, evidence_transaction_ids=["TXN-99999"])
    rec, _ = run(happy_path(t, final=wrong) + [call("final_answer", wrong)], t)
    assert rec["outcome"] == "refused"
    assert rec["answer"]["status"] == "insufficient_evidence"


def test_answer_without_checking_ledger_is_rejected(dataset):
    t = first_invoice(dataset, "paid_exact")
    num = t["invoice"]["invoice_number"]
    rec, _ = run([call("search_invoices", {"query": num}), call("final_answer", answer_for(t)),
                  call("final_answer", answer_for(t))], t)
    assert any("without calling query_ledger" in p for p in verify_steps(rec)[0]["problems"])
    assert rec["outcome"] == "refused"


def test_step_limit():
    turns = [call("query_ledger", {"counterparty": "DUPONT"}) for _ in range(5)]
    rec, _ = run(turns, max_steps=3)
    assert rec["outcome"] == "step_limit" and rec["llm_calls"] == 3


def test_traces_are_stored_and_retrievable(tmp_path):
    store = TraceStore(tmp_path)
    rec = Agent(model=MockModel(), store=store).ask("Is invoice DUP-2026-0001 from Dupont fully paid?")
    assert store.get(rec["request_id"])["steps"] == rec["steps"]  # readable before it is written
    store.flush()
    assert (tmp_path / "traces.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert store.get(rec["request_id"])["steps"] == rec["steps"]  # and from SQLite after


def test_trace_writes_survive_concurrency(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store = TraceStore(tmp_path, batch_size=7)
    agent = Agent(model=MockModel(), store=store)
    with ThreadPoolExecutor(8) as pool:
        recs = list(pool.map(agent.ask, ["Is invoice DUP-2026-0001 from Dupont fully paid?"] * 100))
    store.close()
    assert (tmp_path / "traces.jsonl").read_text(encoding="utf-8").count("\n") == 100
    assert all(store.get(r["request_id"]) for r in recs) and store.write_errors == 0


# ---------- version-specific verifier rules ----------

def test_v1_accepts_unpaid_after_label_only_search(dataset):
    t = first_invoice(dataset, "unpaid")
    rec, _ = run(label_only_path(t) + [call("final_answer", answer_for(t))], t, version="v1")
    assert rec["outcome"] == "answered" and rec["answer"]["status"] == "unpaid"


def test_v2_rejects_unpaid_after_label_only_search(dataset):
    t = first_invoice(dataset, "unpaid")
    final = call("final_answer", answer_for(t))
    rec, _ = run(label_only_path(t) + [final, call("final_answer", answer_for(t))], t, version="v2")
    assert any("only by label" in p for p in verify_steps(rec)[0]["problems"])
    assert rec["outcome"] == "refused"


def test_v2_feedback_needs_two_rounds_for_vat_only(dataset):
    """The v2 failure seen in the eval: one problem reported per rejection, retries run out."""
    t = first_invoice(dataset, "vat_only")
    claim = answer_for(t, status="paid", amount_matches=True, paid_amount=t["invoice"]["amount_ttc"])
    half_fixed = answer_for(t, status="paid", amount_matches=True)
    rec, _ = run(happy_path(t, final=claim) + [call("final_answer", half_fixed)], t, version="v2")
    assert rec["outcome"] == "refused"


def test_v3_reports_the_whole_fix_in_one_rejection(dataset):
    t = first_invoice(dataset, "vat_only")
    claim = answer_for(t, status="paid", amount_matches=True, paid_amount=t["invoice"]["amount_ttc"])
    rec, _ = run(happy_path(t, final=claim) + [call("final_answer", answer_for(t))], t, version="v3")
    problems = verify_steps(rec)[0]["problems"]
    assert len(problems) == 1 and "status=partially_paid" in problems[0] and f"paid_amount={t['paid_amount']:.2f}" in problems[0]
    assert rec["outcome"] == "answered" and rec["answer"]["status"] == "partially_paid"


def test_v3_rejects_invoice_from_similarly_named_supplier(dataset):
    dupond = next(t for t in dataset if t["invoice"]["supplier"] == "Dupond SAS")
    question = "Has supplier Dupont's invoice been paid, and does the bank amount match?"
    rec, _ = run(happy_path(dupond) + [call("final_answer", {"status": "not_found", "explanation": "none"})],
                 question=question, version="v3")
    assert "not the supplier named in the question" in verify_steps(rec)[0]["problems"][0]
    assert rec["answer"]["status"] == "not_found"


def search_and_ledger(t, **ledger_args):
    return [call("search_invoices", {"query": t["invoice"]["invoice_number"]}), call("query_ledger", ledger_args)]


def test_v3_rejects_payment_to_another_counterparty(dataset):
    t = next(t for t in dataset if t["invoice"]["supplier"] == "Dupont SARL" and "similar_name" in t["traps"])
    ttc = t["invoice"]["amount_ttc"]
    twin = query_ledger(QueryLedgerInput(counterparty="DUPOND", min_amount=ttc, max_amount=ttc)).transactions[0]
    claim = answer_for(t, status="paid", amount_matches=True, paid_amount=ttc, evidence_transaction_ids=[twin.txn_id])
    turns = search_and_ledger(t, counterparty="DUPON") + [call("final_answer", claim), call("final_answer", claim)]
    rec, _ = run(turns, t, version="v3")
    assert any("was paid to DUPOND SAS" in p for p in verify_steps(rec)[0]["problems"])
    assert rec["outcome"] == "refused"


def test_v3_rejects_payment_labelled_for_another_invoice(dataset):
    t = first_invoice(dataset, "unpaid")
    inv = t["invoice"]
    other = next(o for o in dataset if o["invoice"]["supplier"] == inv["supplier"] and o["payments"]
                 and o["invoice"]["invoice_number"] in o["payments"][0]["label"])
    pay = other["payments"][0]
    claim = answer_for(t, status="partially_paid", paid_amount=pay["amount"], amount_matches=False,
                       evidence_transaction_ids=[pay["txn_id"]])
    turns = search_and_ledger(t, counterparty=inv["supplier"]) + [call("final_answer", claim),
                                                                 call("final_answer", answer_for(t))]
    rec, _ = run(turns, t, version="v3")
    assert any("a different invoice" in x for x in verify_steps(rec)[0]["problems"])
    assert rec["answer"]["status"] == "unpaid"


def test_v4_rejects_payment_dated_before_the_invoice(dataset):
    t = first_invoice(dataset, "unpaid")
    inv = t["invoice"]
    early = next(p for o in dataset if o["invoice"]["supplier"] == inv["supplier"] for p in o["payments"]
                 if p["date"] < inv["issue_date"] and inv["invoice_number"] not in p["label"])
    claim = answer_for(t, status="partially_paid", paid_amount=early["amount"], amount_matches=False,
                       evidence_transaction_ids=[early["txn_id"]])
    turns = search_and_ledger(t, counterparty=inv["supplier"]) + [call("final_answer", claim),
                                                                 call("final_answer", answer_for(t))]
    rec, _ = run(turns, t, version="v4")
    assert any("before the invoice was issued" in p for p in verify_steps(rec)[0]["problems"])
