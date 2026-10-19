"""The agent's tools. Inputs and outputs are Pydantic models, so a malformed tool call
fails validation here (and is reported back to the model) instead of failing later."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Callable, Literal
from urllib.request import pathname2url

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from recon.config import settings
from recon.retrieval import BM25, tokenize


class ToolInputError(Exception):
    """The model called a tool with arguments that do not parse or validate."""


# ---------- search_invoices ----------

class SearchInvoicesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Free-text search, e.g. a supplier name, an invoice number or a description.")
    supplier: str | None = Field(None, description="Only return invoices whose supplier name contains all of these words.")
    issue_month: str | None = Field(None, pattern=r"^\d{4}-\d{2}$", description="Only return invoices issued in this month, as YYYY-MM.")
    top_k: int = Field(5, ge=1, le=10, description="Number of results to return.")


class InvoiceHit(BaseModel):
    doc_id: str
    supplier: str
    invoice_number: str
    issue_date: str
    due_date: str
    amount_ht: float
    vat_rate: float
    vat_amount: float
    amount_ttc: float
    is_copy: bool = Field(description="True for a DUPLICATA, a copy of an invoice that also appears as an original.")
    score: float


class SearchInvoicesOutput(BaseModel):
    results: list[InvoiceHit]
    note: str | None = None


@lru_cache(maxsize=1)
def _invoice_index() -> tuple[list[dict], BM25]:
    with open(settings.invoices_path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    return docs, BM25([d["text"] for d in docs])


def search_invoices(inp: SearchInvoicesInput) -> SearchInvoicesOutput:
    docs, index = _invoice_index()
    wanted = set(tokenize(inp.supplier)) if inp.supplier else set()

    def keep(i: int) -> bool:
        d = docs[i]
        if wanted and not wanted <= set(tokenize(d["supplier"])):
            return False
        return not inp.issue_month or d["issue_date"].startswith(inp.issue_month)

    hits = index.search(inp.query, inp.top_k, keep)
    results = [InvoiceHit(**{k: docs[i][k] for k in InvoiceHit.model_fields if k != "score"}, score=round(s, 3))
               for i, s in hits]
    note = None if results else "No invoice matches these filters. Check the supplier spelling or remove a filter."
    return SearchInvoicesOutput(results=results, note=note)


# ---------- query_ledger ----------

class QueryLedgerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    counterparty: str | None = Field(None, description="Counterparty name contains this text (case-insensitive).")
    label_contains: str | None = Field(None, description="Bank label contains this text, e.g. an invoice number.")
    min_amount: float | None = Field(None, ge=0, description="Minimum amount in EUR.")
    max_amount: float | None = Field(None, ge=0, description="Maximum amount in EUR.")
    date_from: date | None = Field(None, description="Earliest transaction date, YYYY-MM-DD.")
    date_to: date | None = Field(None, description="Latest transaction date, YYYY-MM-DD.")
    direction: Literal["out", "in"] | None = Field("out", description="'out' for payments we made (default), 'in' for money received.")
    limit: int = Field(20, ge=1, le=50)

    @model_validator(mode="after")
    def check(self):
        if not any([self.counterparty, self.label_contains, self.min_amount is not None,
                    self.max_amount is not None, self.date_from, self.date_to]):
            raise ValueError("give at least one filter (counterparty, label_contains, amount or date range)")
        if self.min_amount is not None and self.max_amount is not None and self.min_amount > self.max_amount:
            raise ValueError("min_amount is greater than max_amount")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        return self


class Transaction(BaseModel):
    txn_id: str
    date: str
    amount: float
    direction: str
    counterparty: str
    label: str


class QueryLedgerOutput(BaseModel):
    transactions: list[Transaction]
    total_amount: float
    truncated: bool


def _like(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def query_ledger(inp: QueryLedgerInput) -> QueryLedgerOutput:
    # Parameterised SQL on a read-only connection: the model only ever supplies values.
    where, params = [], []
    if inp.counterparty:
        where.append("counterparty LIKE ? ESCAPE '\\'"); params.append(_like(inp.counterparty))
    if inp.label_contains:
        where.append("label LIKE ? ESCAPE '\\'"); params.append(_like(inp.label_contains))
    if inp.min_amount is not None:
        where.append("amount >= ?"); params.append(inp.min_amount)
    if inp.max_amount is not None:
        where.append("amount <= ?"); params.append(inp.max_amount)
    if inp.date_from:
        where.append("date >= ?"); params.append(inp.date_from.isoformat())
    if inp.date_to:
        where.append("date <= ?"); params.append(inp.date_to.isoformat())
    if inp.direction:
        where.append("direction = ?"); params.append(inp.direction)
    sql = ("SELECT txn_id, date, amount, direction, counterparty, label FROM transactions"
           f" WHERE {' AND '.join(where)} ORDER BY date, txn_id LIMIT ?")
    params.append(inp.limit + 1)

    con = sqlite3.connect(f"file:{pathname2url(str(settings.ledger_path))}?mode=ro", uri=True)
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    truncated = len(rows) > inp.limit
    txns = [Transaction(txn_id=r[0], date=r[1], amount=r[2], direction=r[3], counterparty=r[4], label=r[5])
            for r in rows[:inp.limit]]
    return QueryLedgerOutput(transactions=txns, total_amount=round(sum(t.amount for t in txns), 2), truncated=truncated)


# ---------- final_answer ----------

class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["paid", "partially_paid", "unpaid", "not_found", "insufficient_evidence"] = Field(
        description="paid: payments equal the TTC total. partially_paid: some payment, less than TTC. "
                    "unpaid: no payment found. not_found: no such invoice. "
                    "insufficient_evidence: the evidence is contradictory or incomplete.")
    invoice_number: str | None = Field(None, description="The invoice the answer is about.")
    invoice_amount_ttc: float | None = Field(None, description="Invoice total including VAT, from the invoice document.")
    paid_amount: float | None = Field(None, description="Sum of the cited bank payments for this invoice (0 if unpaid).")
    amount_matches: bool | None = Field(None, description="True only if paid_amount equals invoice_amount_ttc.")
    evidence_transaction_ids: list[str] = Field(default_factory=list, description="txn_id of every payment counted in paid_amount.")
    explanation: str = Field(description="One or two sentences for an accountant.")


# ---------- registry ----------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[BaseModel], BaseModel] | None  # None for final_answer, which the agent handles itself


TOOLS = {t.name: t for t in [
    ToolSpec("search_invoices", "Search the supplier invoice documents. Returns structured invoice fields.",
             SearchInvoicesInput, search_invoices),
    ToolSpec("query_ledger", "Read-only search of the company bank account transactions.",
             QueryLedgerInput, query_ledger),
    ToolSpec("final_answer", "Submit the final structured answer. Call it alone, once, after checking the ledger.",
             FinalAnswer, None),
]}


def _clean_schema(node):
    """Flatten Pydantic's JSON schema into the plain subset function-calling APIs expect."""
    if isinstance(node, dict):
        node = {k: _clean_schema(v) for k, v in node.items()
                if k != "title" and not (k == "default" and v is None)}
        any_of = node.pop("anyOf", None)
        if any_of:
            non_null = [s for s in any_of if s.get("type") != "null"]
            node = {**(non_null[0] if len(non_null) == 1 else {"anyOf": non_null}), **node}
        return node
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def tool_schemas() -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": _clean_schema(t.input_model.model_json_schema())}}
            for t in TOOLS.values()]


def parse_args(name: str, raw: str | dict) -> BaseModel:
    """Validate a tool call's arguments, raising ToolInputError with a message the model can act on."""
    if name not in TOOLS:
        raise ToolInputError(f"unknown tool '{name}'. Available: {', '.join(TOOLS)}")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        raise ToolInputError(f"arguments are not valid JSON: {e}") from e
    try:
        return TOOLS[name].input_model.model_validate(args or {})
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors())
        raise ToolInputError(f"invalid arguments for {name}: {problems}") from e


def run_tool(name: str, inp: BaseModel) -> BaseModel:
    return TOOLS[name].fn(inp)
