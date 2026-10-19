import sqlite3

import pytest

from recon.config import settings
from recon.data_gen import generate
from recon.tools import (QueryLedgerInput, SearchInvoicesInput, ToolInputError, parse_args, query_ledger,
                         search_invoices, tool_schemas)


def test_generation_is_deterministic():
    a, b = generate(), generate()
    assert a[1] == b[1] and a[3] == b[3]


def test_search_finds_invoice_by_number(dataset):
    inv = dataset[0]["invoice"]
    out = search_invoices(SearchInvoicesInput(query=inv["invoice_number"]))
    assert out.results[0].invoice_number == inv["invoice_number"]


def test_supplier_filter_separates_similar_names():
    out = search_invoices(SearchInvoicesInput(query="facture", supplier="Dupont", top_k=10))
    assert out.results and {r.supplier for r in out.results} == {"Dupont SARL"}


def test_search_with_no_match_explains_why():
    out = search_invoices(SearchInvoicesInput(query="x", supplier="Durand Plomberie"))
    assert out.results == [] and "No invoice" in out.note


def test_ledger_finds_payments_by_reference(dataset):
    t = next(t for t in dataset if t["scenario"] == "split_payment")
    out = query_ledger(QueryLedgerInput(label_contains=t["invoice"]["invoice_number"]))
    assert out.total_amount == pytest.approx(t["invoice"]["amount_ttc"], abs=0.01)
    assert len(out.transactions) == 2


def test_ledger_treats_input_as_data_not_sql():
    out = query_ledger(QueryLedgerInput(counterparty="x' OR 1=1; DROP TABLE transactions; --"))
    assert out.transactions == []
    with sqlite3.connect(settings.ledger_path) as con:
        assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] > 0


def test_ledger_wildcards_are_escaped():
    assert query_ledger(QueryLedgerInput(counterparty="%")).transactions == []


@pytest.mark.parametrize("name, raw, fragment", [
    ("search_invoices", "not json", "not valid JSON"),
    ("search_invoices", '{"query": "x", "issue_month": "March"}', "issue_month"),
    ("search_invoices", '{"query": "x", "colour": "red"}', "colour"),
    ("query_ledger", "{}", "at least one filter"),
    ("query_ledger", '{"min_amount": 10, "max_amount": 5}', "min_amount is greater"),
    ("delete_invoice", "{}", "unknown tool"),
])
def test_bad_arguments_give_actionable_errors(name, raw, fragment):
    with pytest.raises(ToolInputError, match=fragment):
        parse_args(name, raw)


def test_schemas_are_plain_json_schema():
    text = str(tool_schemas())
    assert "anyOf" not in text and "title" not in text
