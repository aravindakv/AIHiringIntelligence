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