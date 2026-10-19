"""HTTP API.

    uvicorn recon.api:app --port 8000
    RECON_LLM=mock uvicorn recon.api:app --port 8000     # offline, for load tests

Endpoints are plain `def`: the agent is synchronous, so FastAPI runs each request in
its worker thread pool and a slow model call never blocks the event loop.
"""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from recon.agent import Agent
from recon.tracing import TraceStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if get_agent.cache_info().currsize:  # flush queued traces on shutdown
        get_agent().store.close()


app = FastAPI(title="invoice-recon-agent", version="0.1.0", lifespan=lifespan)


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    """One agent per process, built on first request (tests override this dependency)."""
    return Agent(store=TraceStore())


AgentDep = Annotated[Agent, Depends(get_agent)]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    request_id: str
    outcome: str
    answer: dict | None
    latency_ms: float
    llm_calls: int
    cost_usd: float
    trace_url: str


@app.get("/health")
def health(agent: AgentDep) -> dict:
    return {"status": "ok", "model": agent.model.name, "agent_version": agent.version}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, agent: AgentDep) -> AskResponse:
    r = agent.ask(req.question)
    return AskResponse(request_id=r["request_id"], outcome=r["outcome"], answer=r["answer"],
                       latency_ms=r["latency_ms"], llm_calls=r["llm_calls"], cost_usd=r["cost_usd"],
                       trace_url=f"/trace/{r['request_id']}")


@app.get("/trace/{request_id}")
def trace(request_id: str, agent: AgentDep) -> dict:
    record = agent.store.get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return record
