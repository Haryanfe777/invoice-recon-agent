"""The agent: a LangGraph state machine around native function calling.

    START -> model --(tool call)------> tools  --> model   gather evidence
                   --(final_answer)---> verify --> END     answer passes the evidence checks
                                               --> model   one chance to fix a rejected answer
                   --(no tool call)---> model              one retry after a malformed turn

Limits: a step budget (model calls), a request deadline, one retry after a malformed
tool call, and one retry after a rejected answer. After that the agent refuses
(status insufficient_evidence) rather than return an answer the evidence does not support.
"""

import json
import time
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from recon.config import settings
from recon.data_gen import ensure_data
from recon.llm import ChatModel, ModelError, make_model
from recon.tools import FinalAnswer, ToolInputError, parse_args, run_tool, tool_schemas
from recon.tracing import Tracer, TraceStore

PROMPT_VERSION = "v1"
SYSTEM_PROMPT = """You are an accounts-payable assistant for a small French company.
You answer questions about supplier invoices and whether the bank account shows they were paid.

Work only from tool results:
1. Find the invoice with search_invoices. Use the supplier and issue_month filters when the question gives them.
2. Check payments with query_ledger. Never conclude anything about payment from the invoice alone.
3. Submit with final_answer. paid_amount is the sum of the payments you cite in evidence_transaction_ids.
   status is "paid" if paid_amount equals the invoice TTC total, "partially_paid" if it is lower, "unpaid" if no payment exists.
   Use "not_found" if no invoice matches the question, and "insufficient_evidence" if the evidence is contradictory.
Amounts are in EUR. HT means excluding VAT, TTC means including VAT. Today is 2026-08-31."""

TOL = 0.01  # euro cent tolerance for amount comparisons


class AgentState(TypedDict):
    messages: list[dict]
    pending: list[dict]       # tool calls from the latest model turn
    llm_calls: int
    malformed_streak: int     # consecutive turns with an unusable tool call
    rejected_answers: int
    evidence: dict            # what the tools actually returned in this run
    outcome: str | None       # set when the run is over
    answer: dict | None


def check_answer(a: FinalAnswer, ev: dict) -> list[str]:
    """Deterministic checks that every claim in the answer is backed by tool output."""
    if a.status == "insufficient_evidence":
        return []
    if a.status == "not_found":
        return [] if ev["searched"] else ["status not_found but search_invoices was never called"]

    problems = []
    inv = ev["invoices"].get(a.invoice_number) if a.invoice_number else None
    if not a.invoice_number:
        problems.append(f"invoice_number is required for status {a.status}")
    elif inv is None:
        problems.append(f"invoice {a.invoice_number} was not returned by search_invoices")
    elif a.invoice_amount_ttc is None or abs(a.invoice_amount_ttc - inv["amount_ttc"]) > TOL:
        problems.append(f"invoice_amount_ttc {a.invoice_amount_ttc} differs from the document's TTC total {inv['amount_ttc']}")

    if not ev["ledger_queried"]:
        problems.append("payment status given without calling query_ledger")
    ids = a.evidence_transaction_ids
    unknown = [t for t in ids if t not in ev["transactions"]]
    if unknown:
        problems.append(f"transactions not returned by query_ledger: {', '.join(unknown)}")
    if len(set(ids)) != len(ids):
        problems.append("the same transaction is cited more than once")
    cited = round(sum(ev["transactions"][t]["amount"] for t in set(ids) if t in ev["transactions"]), 2)
    if a.paid_amount is None or abs(a.paid_amount - cited) > TOL:
        problems.append(f"paid_amount {a.paid_amount} is not the sum of the cited transactions ({cited:.2f})")

    if inv is not None and a.paid_amount is not None:
        ttc = inv["amount_ttc"]
        if a.paid_amount > ttc + TOL:
            problems.append(f"paid_amount {a.paid_amount} exceeds the TTC total {ttc}: check for payments "
                            "that belong to another invoice or another supplier")
        else:
            expected = "paid" if abs(a.paid_amount - ttc) <= TOL else ("partially_paid" if a.paid_amount > TOL else "unpaid")
            if a.status != expected:
                problems.append(f"status {a.status} is inconsistent with paid {a.paid_amount} vs TTC {ttc} (expected {expected})")
            if a.amount_matches is not (expected == "paid"):
                problems.append(f"amount_matches should be {expected == 'paid'}")
    return problems


def _summarize(name: str, out) -> str:
    if name == "search_invoices":
        hits = ", ".join(f"{h.invoice_number}{' (copy)' if h.is_copy else ''} {h.supplier} TTC {h.amount_ttc}"
                         for h in out.results)
        return f"{len(out.results)} hit(s): {hits}" if out.results else "0 hits"
    hits = ", ".join(f"{t.txn_id} {t.counterparty} {t.amount}" for t in out.transactions[:6])
    more = " ..." if len(out.transactions) > 6 or out.truncated else ""
    return f"{len(out.transactions)} txn(s), total {out.total_amount}: {hits}{more}"


def _args_for_trace(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _tool_message(call: dict, content: dict | str) -> dict:
    body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {"role": "tool", "name": call["name"], "content": body, "tool_call_id": call["id"]}


class Agent:
    def __init__(self, model: ChatModel | None = None, store: TraceStore | None = None,
                 max_steps: int = settings.max_steps, request_timeout_s: float = settings.request_timeout_s,
                 max_malformed_retries: int = 1, max_rejected_answers: int = 1):
        ensure_data()
        self.model = model or make_model()
        self.store = store
        self.max_steps = max_steps
        self.request_timeout_s = request_timeout_s
        self.max_malformed_retries = max_malformed_retries
        self.max_rejected_answers = max_rejected_answers
        self.tool_specs = tool_schemas()
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("model", self._model_node)
        g.add_node("tools", self._tools_node)
        g.add_node("verify", self._verify_node)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", self._after_model, ["tools", "verify", "model", END])
        g.add_conditional_edges("tools", lambda s: END if s["outcome"] else "model", ["model", END])
        g.add_conditional_edges("verify", lambda s: END if s["outcome"] else "model", ["model", END])
        return g.compile()

    def ask(self, question: str, request_id: str | None = None) -> dict:
        """Answer one question. Always returns a trace record, even when the run fails."""
        tracer = Tracer(question, self.model.name, PROMPT_VERSION, request_id)
        state: AgentState = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            "pending": [], "llm_calls": 0, "malformed_streak": 0, "rejected_answers": 0,
            "evidence": {"searched": False, "ledger_queried": False, "invoices": {}, "transactions": {}},
            "outcome": None, "answer": None,
        }
        try:
            final = self.graph.invoke(state, config={"configurable": {"tracer": tracer},
                                                     "recursion_limit": 4 * self.max_steps + 10})
            outcome, answer = final["outcome"] or "error", final["answer"]
        except Exception as e:  # one bad request must not take the service down; record it instead
            tracer.steps.append({"i": len(tracer.steps), "kind": "error", "error": f"{type(e).__name__}: {e}"})
            outcome, answer = "error", None
        record = tracer.finish(outcome, answer)
        if self.store:
            self.store.save(record)
        return record

    # ---------- nodes ----------

    def _model_node(self, state: AgentState, config: RunnableConfig) -> dict:
        tracer: Tracer = config["configurable"]["tracer"]
        if tracer.elapsed_s() > self.request_timeout_s:
            return {"outcome": "timeout", "pending": []}
        if state["llm_calls"] >= self.max_steps:
            return {"outcome": "step_limit", "pending": []}

        t0 = time.perf_counter()
        try:
            res = self.model.complete(state["messages"], self.tool_specs)
        except ModelError as e:
            tracer.llm((time.perf_counter() - t0) * 1000, 0, 0, [], error=str(e))
            return {"outcome": "error", "pending": [], "llm_calls": state["llm_calls"] + 1}
        tracer.llm((time.perf_counter() - t0) * 1000, res.prompt_tokens, res.completion_tokens,
                   [c.name for c in res.tool_calls], retries=res.retries)

        pending = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in res.tool_calls]
        assistant = {"role": "assistant", "content": res.content}
        if pending:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}} for c in pending]
        update = {"messages": state["messages"] + [assistant], "pending": pending, "llm_calls": state["llm_calls"] + 1}

        if not pending:  # prose instead of a tool call
            streak = state["malformed_streak"] + 1
            if streak > self.max_malformed_retries:
                return {**update, "malformed_streak": streak, "outcome": "error"}
            nudge = {"role": "user", "content": "Respond with a tool call only: search_invoices or query_ledger "
                                                "to gather evidence, or final_answer to submit."}
            return {**update, "messages": update["messages"] + [nudge], "malformed_streak": streak}
        return update

    def _after_model(self, state: AgentState) -> str:
        if state["outcome"]:
            return END
        if not state["pending"]:
            return "model"
        if any(c["name"] == "final_answer" for c in state["pending"]):
            return "verify"
        return "tools"

    def _tools_node(self, state: AgentState, config: RunnableConfig) -> dict:
        tracer: Tracer = config["configurable"]["tracer"]
        messages = list(state["messages"])
        ev = {**state["evidence"], "invoices": dict(state["evidence"]["invoices"]),
              "transactions": dict(state["evidence"]["transactions"])}
        streak = state["malformed_streak"]

        for call in state["pending"]:
            t0 = time.perf_counter()
            try:
                inp = parse_args(call["name"], call["arguments"])
                out = run_tool(call["name"], inp)
            except ToolInputError as e:
                # Tell the model exactly what was wrong; it gets one retry.
                streak += 1
                tracer.tool(call["name"], _args_for_trace(call["arguments"]), (time.perf_counter() - t0) * 1000,
                            "rejected", error=str(e))
                messages.append(_tool_message(call, {"error": str(e)}))
                continue
            except Exception as e:
                streak += 1
                tracer.tool(call["name"], _args_for_trace(call["arguments"]), (time.perf_counter() - t0) * 1000,
                            "failed", error=f"{type(e).__name__}: {e}")
                messages.append(_tool_message(call, {"error": f"tool failed ({type(e).__name__}); try different arguments"}))
                continue

            streak = 0
            if call["name"] == "search_invoices":
                ev["searched"] = True
                for h in out.results:
                    ev["invoices"][h.invoice_number] = h.model_dump()
            else:
                ev["ledger_queried"] = True
                for t in out.transactions:
                    ev["transactions"][t.txn_id] = t.model_dump()
            tracer.tool(call["name"], inp.model_dump(mode="json", exclude_none=True),
                        (time.perf_counter() - t0) * 1000, _summarize(call["name"], out))
            messages.append(_tool_message(call, out.model_dump_json()))

        update = {"messages": messages, "evidence": ev, "malformed_streak": streak, "pending": []}
        if streak > self.max_malformed_retries:
            update["outcome"] = "error"
        return update

    def _verify_node(self, state: AgentState, config: RunnableConfig) -> dict:
        tracer: Tracer = config["configurable"]["tracer"]
        messages = list(state["messages"])
        answer, problems, answer_call = None, [], None

        for call in state["pending"]:
            if call["name"] != "final_answer":
                problems.append(f"{call['name']} was called together with final_answer; call final_answer alone")
                messages.append(_tool_message(call, {"error": "not executed: final_answer must be called alone"}))
                continue
            answer_call = call
            try:
                answer = parse_args("final_answer", call["arguments"])
            except ToolInputError as e:
                problems.append(str(e))
        if answer is not None:
            problems += check_answer(answer, state["evidence"])
        tracer.verify(not problems, problems)

        if not problems:
            messages.append(_tool_message(answer_call, {"accepted": True}))
            outcome = "refused" if answer.status == "insufficient_evidence" else "answered"
            return {"messages": messages, "pending": [], "answer": answer.model_dump(), "outcome": outcome}

        rejected = state["rejected_answers"] + 1
        if rejected > self.max_rejected_answers:
            refusal = FinalAnswer(status="insufficient_evidence",
                                  explanation="No answer given: the evidence did not support it. " + "; ".join(problems))
            return {"messages": messages, "pending": [], "rejected_answers": rejected,
                    "answer": refusal.model_dump(), "outcome": "refused"}
        messages.append(_tool_message(answer_call, {
            "rejected": True, "problems": problems,
            "instruction": "Fix these problems using tool results, then call final_answer again."}))
        return {"messages": messages, "pending": [], "rejected_answers": rejected}
