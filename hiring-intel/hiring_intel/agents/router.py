"""Routing Agent: picks the model tier before an insight agent runs.

Order of decision makers, cheapest first:
  1. scored rules (free, explainable)
  2. OpenJev yes/no on the task description, for borderline scores ($0 API)
  3. small LLM, only if OpenJev's confidence is below threshold
Every decision is logged to the `routing` table with who decided it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel

from hiring_intel.llm import call_structured, model_for
from hiring_intel.observability import execute, log_decision
from hiring_intel.openjev import decider


@dataclass
class TaskProfile:
    agent: str
    context_tokens: int
    n_segments: int
    needs_causal_reasoning: bool
    stakes: Literal["low", "high"]
    last_eval_score: float | None = None

    def describe(self) -> str:
        return (f"Analytics task for the {self.agent} agent over {self.n_segments} data segments and about "
                f"{self.context_tokens} tokens of context. "
                + ("It must attribute causes, separating competing explanations. " if self.needs_causal_reasoning
                   else "It mostly reads thresholds and ranks numbers. ")
                + ("Its output affects compensation or offers." if self.stakes == "high" else ""))


class RouteDecision(BaseModel):
    tier: Literal["small", "large"]
    reason: str


def score(p: TaskProfile) -> float:
    s = 0.0
    s += 0.3 if p.context_tokens > 6000 else 0.15 if p.context_tokens > 2000 else 0
    s += 0.2 if p.n_segments > 60 else 0.1 if p.n_segments > 20 else 0
    s += 0.25 if p.needs_causal_reasoning else 0
    s += 0.15 if p.stakes == "high" else 0
    if p.last_eval_score is not None and p.last_eval_score < 0.6:
        s += 0.3
    return round(s, 2)


def route(p: TaskProfile, run_id: str, force_tier: str | None = None) -> tuple[str, str]:
    """Returns (tier, model)."""
    if force_tier:
        tier, reason, by = force_tier, "forced by caller", "config"
    else:
        s = score(p)
        if s < 0.3:
            tier, reason, by = "small", f"rule score {s}", "rules"
        elif s > 0.5:
            tier, reason, by = "large", f"rule score {s}", "rules"
        else:
            d = decider().yes_no("route_tier", p.describe(),
                                 "Does this task require multi-step causal reasoning across many segments?",
                                 yes_keywords=["attribute causes", "competing explanations", "compensation"])
            log_decision(run_id, d)
            if not d.escalated:
                tier = "large" if d.choice == "yes" else "small"
                reason, by = f"openjev p(yes)={d.probs['yes']} (score {s})", d.backend
            else:
                rd, _ = call_structured(
                    agent="router", run_id=run_id, model=model_for("small"),
                    system="Choose a model tier for an analytics sub-task. 'large' only for multi-step causal "
                           "reasoning over many segments.",
                    user=str(asdict(p)), schema=RouteDecision,
                    mock_fn=lambda: RouteDecision(tier="large" if p.needs_causal_reasoning else "small",
                                                  reason="mock router: causal reasoning flag"))
                tier, reason, by = rd.tier, f"llm: {rd.reason} (score {s})", "llm-small"
    model = model_for(tier)
    execute("INSERT INTO routing VALUES (?,?,?,?,?,?)", (run_id, p.agent, tier, model, reason, by))
    return tier, model