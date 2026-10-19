# invoice-recon-agent

An agent that answers accounts-payable questions such as *"Has supplier Dupont's March invoice
been paid, and does the bank amount match?"*. It retrieves the invoice, verifies it against the
bank ledger, and refuses to answer when the evidence does not agree.

> **The data is synthetic.** It is generated from a fixed seed (`python -m recon.data_gen`), so
> every evaluation question has a known correct answer. It includes deliberate traps: split
> payments, partial payments, payments of the pre-VAT amount only, duplicate invoice copies,
> payments without a reference, and suppliers with near-identical names (Dupont / Dupond).

**Status: work in progress.** Done: data, tools, agent loop, tracing, tests. Next: evaluation,
FastAPI + Docker, load test.

## How it works

```
START -> model --(tool call)------> tools  --> model   gather evidence
               --(final_answer)---> verify --> END     every claim checked against tool output
                                           --> model   one chance to fix a rejected answer
               --(no tool call)---> model              one retry after a malformed turn
```

- **LangGraph** state machine, **Mistral** native function calling (`tool_choice="any"`).
- **Tools** (`recon/tools.py`), all typed with Pydantic:
  - `search_invoices`: BM25 over invoice documents, with supplier and month filters.
  - `query_ledger`: parameterised, read-only SQL over bank transactions.
  - `final_answer`: a structured answer (status, amounts, cited transaction ids).
- **Verification** (`check_answer` in `recon/agent.py`): the cited invoice and transactions must
  have been returned by the tools in this run, `paid_amount` must equal the sum of the cited
  payments, and the status must be consistent with the amounts. A rejected answer gets one retry;
  after that the agent returns `insufficient_evidence`.
- **Limits**: step budget, request deadline, provider retries with backoff on 429/5xx, one retry
  after a malformed tool call.
- **Tracing** (`recon/tracing.py`): one record per request with every step's arguments, result
  summary, latency, tokens and cost, written to `traces/traces.jsonl` and `traces/traces.db`.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
cp .env.example .env            # then add your MISTRAL_API_KEY
python -m recon.data_gen
python -m recon.cli "Has supplier Dupont's March invoice been paid, and does the bank amount match?"
```

Without an API key, `--mock` uses a deterministic rule-based model that drives the same tools:

```bash
python -m recon.cli --mock "Is invoice DUP-2026-0003 from Dupont fully paid?"
```

Tests (no API calls):

```bash
pytest -q
```
