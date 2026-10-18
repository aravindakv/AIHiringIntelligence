# 03 — Shared output contract and observability

**Goal:** one output shape for every agent, and a run store that records every LLM call, decision and timing.

**Time:** 30 minutes.

## Concepts

- **One contract, five required fields.** The brief requires `recommendation`, `evidence`, `confidence_score`, `cost_of_insight` and `alternative` on every insight. One Pydantic model enforces them for all agents, so evaluation, human review, A2A and the dashboard all read one shape.
- **Who fills what.** The LLM fills the reasoning fields. Code fills `cost_of_insight` from real token counts, so cost can never be hallucinated.
- **Validation is the first quality gate.** The `recommendation` validator rejects generic advice ("consider improving…") before it reaches anyone.
- **Observability you can query.** Every LLM call, OpenJev decision, routing choice and stage timing goes to SQLite. The Optimization Agent, the cost report and the dashboard all read those tables. JSON log lines go to stderr when `LOG_LEVEL=INFO`.

## Step 3.1 — The contract

**File:** `hiring_intel/contracts.py`
```python
"""Shared output contract. Every agent returns AgentOutput; every finding is an Insight.

The five fields required by the brief map 1:1:
    recommendation, evidence, confidence_score, cost_of_insight, alternative
cost_of_insight is filled by code from real (or estimated) token counts, never by the LLM.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Evidence(BaseModel):
    metric: str                                   # e.g. "tech_1_pass_rate"
    value: float | str
    segment: dict[str, str] = {}                  # {"role": "Data", "source": "agency"}
    source: Literal["data", "rag"] = "data"       # computed from ATS data, or retrieved context
    ref: str | None = None                        # RAG chunk id for traceability


class Cost(BaseModel):
    model: str
    tier: str = "small"
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    latency_ms: int = 0                           # LLM latency (simulated in mock mode)
    simulated: bool = False
    cached: bool = False


class Alternative(BaseModel):
    option: str                                   # "Run on small model" / "Rules only"
    trade_off: str
    est_usd: float | None = None


class Move(BaseModel):
    from_interviewer: str
    to_interviewer: str
    slots_per_week: float


class Insight(BaseModel):
    title: str
    recommendation: str = Field(min_length=30)
    evidence: list[Evidence] = Field(min_length=1)
    confidence_score: float = Field(ge=0, le=1)
    alternative: Alternative
    severity: Literal["low", "medium", "high"] = "medium"
    rebalance_plan: list[Move] = []               # Panel Load Balancer only

    @field_validator("recommendation")
    @classmethod
    def must_be_actionable(cls, v: str) -> str:
        vague = ("consider improving", "monitor closely", "it may be", "keep an eye")
        if any(p in v.lower() for p in vague):
            raise ValueError("recommendation is generic; name an owner, action or number")
        return v


class LLMInsights(BaseModel):
    """What the LLM is asked to return (a subset of AgentOutput)."""
    insights: list[Insight]


class AgentOutput(BaseModel):
    agent: str
    run_id: str
    status: Literal["ok", "degraded", "failed"] = "ok"
    insights: list[Insight] = []
    cost_of_insight: Cost | None = None
    compute_ms: int = 0                           # local work: pandas, retrieval, OpenJev
    retrieved_chunks: list[str] = []
    error: str | None = None


class Decision(BaseModel):
    """A closed-set decision made by OpenJev (or its fallback)."""
    task: str
    choice: str
    confidence: float
    probs: dict[str, float]
    backend: str                                  # openjev-local | openjev-http | keyword-fallback
    latency_ms: int
    escalated: bool = False                       # confidence too low, handed to LLM/human


class ReviewItem(BaseModel):
    item_id: str
    run_id: str
    agent: str
    insight: Insight
    reasons: list[str]                            # why HITL was triggered
    status: Literal["pending", "approved", "rejected", "edited", "auto_published"] = "pending"
    reviewer: str | None = None
    note: str | None = None
```

`Decision` (an OpenJev answer) and `ReviewItem` (a human-review queue entry) are defined here too, because several modules share them.

## Step 3.2 — The run store

Tables: `runs`, `llm_calls`, `decisions`, `timings`, `agent_outputs`, `routing`, `eval_results`, `review_queue`. A lock makes writes safe from LangGraph's parallel threads.

**File:** `hiring_intel/observability.py`
```python
"""Structured JSON logs + a SQLite run store. The dashboard and the Optimization Agent read it."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
import threading

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(message)s")
log = logging.getLogger("hiring_intel")

DB_PATH = os.getenv("HI_DB", "data/runs.db")
_lock = threading.Lock()
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
DB = sqlite3.connect(DB_PATH, check_same_thread=False)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, ts TEXT, scenario TEXT, size TEXT, variant TEXT,
    data_path TEXT, n_candidates INT, total_usd REAL, llm_calls INT, input_tokens INT, output_tokens INT,
    e2e_latency_ms INT, wall_ms INT, mean_eval REAL, hitl_pending INT, hitl_auto INT, status TEXT);
CREATE TABLE IF NOT EXISTS llm_calls(ts TEXT, run_id TEXT, agent TEXT, model TEXT, tier TEXT,
    input_tokens INT, output_tokens INT, usd REAL, latency_ms INT, cached INT, simulated INT);
CREATE TABLE IF NOT EXISTS decisions(ts TEXT, run_id TEXT, task TEXT, choice TEXT, confidence REAL,
    backend TEXT, latency_ms INT, escalated INT, n_items INT);
CREATE TABLE IF NOT EXISTS timings(run_id TEXT, stage TEXT, agent TEXT, compute_ms INT, llm_ms INT);
CREATE TABLE IF NOT EXISTS agent_outputs(run_id TEXT, agent TEXT, status TEXT, json TEXT);
CREATE TABLE IF NOT EXISTS routing(run_id TEXT, agent TEXT, tier TEXT, model TEXT, reason TEXT, decided_by TEXT);
CREATE TABLE IF NOT EXISTS eval_results(run_id TEXT, agent TEXT, overall REAL, json TEXT);
CREATE TABLE IF NOT EXISTS review_queue(item_id TEXT PRIMARY KEY, run_id TEXT, agent TEXT, status TEXT,
    reviewer TEXT, note TEXT, reasons TEXT, json TEXT, ts TEXT);
"""
with _lock:
    DB.executescript(SCHEMA)
    DB.commit()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def execute(sql: str, params=()) -> None:
    with _lock:
        DB.execute(sql, params)
        DB.commit()


def log_call(agent: str, run_id: str, cost) -> None:
    row = dict(ts=_now(), run_id=run_id, agent=agent, model=cost.model, tier=cost.tier,
               input_tokens=cost.input_tokens, output_tokens=cost.output_tokens, usd=cost.usd,
               latency_ms=cost.latency_ms, cached=int(cost.cached), simulated=int(cost.simulated))
    log.info(json.dumps({"event": "llm_call", **row}))
    execute("INSERT INTO llm_calls VALUES (:ts,:run_id,:agent,:model,:tier,:input_tokens,:output_tokens,"
            ":usd,:latency_ms,:cached,:simulated)", row)


def log_decision(run_id: str, d, n_items: int = 1) -> None:
    log.info(json.dumps({"event": "decision", "run_id": run_id, **d.model_dump(exclude={"probs"})}))
    execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
            (_now(), run_id, d.task, d.choice, d.confidence, d.backend, d.latency_ms, int(d.escalated), n_items))


def log_timing(run_id: str, stage: str, agent: str, compute_ms: int, llm_ms: int) -> None:
    execute("INSERT INTO timings VALUES (?,?,?,?,?)", (run_id, stage, agent, int(compute_ms), int(llm_ms)))


def query(sql: str, params=()):
    import pandas as pd
    with _lock:
        return pd.read_sql(sql, DB, params=params)
```

## Verify

<!-- verify:03 -->
```bash
python - <<'EOF'
from pydantic import ValidationError
from hiring_intel.contracts import Insight, Evidence, Alternative
alt = Alternative(option="rules only", trade_off="no narrative")
ok = Insight(title="t", recommendation="Recruiting lead: move 20% of agency budget to referrals this quarter.",
             evidence=[Evidence(metric="hire_rate", value=0.12)], confidence_score=0.7, alternative=alt)
print("valid insight:", ok.confidence_score)
try:
    Insight(title="t", recommendation="Consider improving the sourcing process across the board.",
            evidence=[Evidence(metric="m", value=1)], confidence_score=0.5, alternative=alt)
except ValidationError as e:
    print("rejected:", e.errors()[0]["msg"])
from hiring_intel.observability import query
print(query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").name.tolist())
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
valid insight: 0.7
rejected: Value error, recommendation is generic; name an owner, action or number
['agent_outputs', 'decisions', 'eval_results', 'llm_calls', 'review_queue', 'routing', 'runs', 'timings']
```

## Checkpoint

- [ ] a vague recommendation is rejected at validation time
- [ ] `data/runs.db` exists with eight tables
