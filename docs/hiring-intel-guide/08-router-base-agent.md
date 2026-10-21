# 08 — Routing Agent, base agent and prompts

**Goal:** the shared pipeline every insight agent runs (features → retrieve → route → LLM → calibrate → contract), and the Routing Agent that picks the model tier.

**Time:** 45 minutes.

## Concepts

- **Routing, cheapest decision-maker first.**
  1. Scored rules over a `TaskProfile` (context size, segments, causal reasoning needed, stakes, last eval score). Free and explainable.
  2. Borderline scores go to OpenJev: "does this need multi-step causal reasoning?"
  3. Only if OpenJev is unsure does a small LLM decide.

  Paying a large model to pick a model on every call would eat the savings routing creates.
- **Escalation.** An agent whose last eval score was under 0.6 gets +0.3 on its score, which pushes it to the large tier.
- **The base agent does the plumbing once.** Each insight agent supplies only its `features()`, a RAG query, a prompt file and `analyze()`, a rule-based analyst. In mock mode that analyst writes the JSON. In real mode it is the **detector**: its findings go into the prompt as `SIGNALS`, and the LLM may report only those, explaining causes and sharpening the action. It is also the "rules only" alternative named in every insight.
- **Detect in code, explain with the LLM.** Given only the feature tables, a real model flagged noise as `high` on the `healthy` scenario. Small-sample gaps look like findings when you can't run a significance test. `guard()` then caps each LLM finding's severity at the worst detected signal, so a no-action run stays `low`. `SIGNALS=off` turns this off so you can see the difference.
- **Confidence calibration.** The LLM's self-reported confidence is capped by evidence strength: `min(0.95, 0.5 + 0.1 × evidence items + 0.1 if any RAG evidence)`. A model can't claim 0.9 on one data point.
- **Failure isolation.** Any exception inside an agent becomes a `failed` output with the error. The run continues.

## Step 8.1 — Routing Agent

**File:** `hiring_intel/agents/router.py`
```python
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
```

## Step 8.2 — Base agent

**File:** `hiring_intel/agents/base.py`
```python
"""BaseInsightAgent: features -> retrieve -> route -> LLM (or mock analyst) -> calibrate -> contract."""
from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from hiring_intel.agents.router import TaskProfile, route
from hiring_intel.contracts import AgentOutput, Evidence, Insight, LLMInsights
from hiring_intel.ingest import HiringData
from hiring_intel.llm import ROOT, call_structured
from hiring_intel.metrics import to_text
from hiring_intel.rag.retriever import retrieve

SLA = yaml.safe_load((ROOT / "config/sla.yaml").read_text())

SYSTEM_COMMON = """You are a hiring analytics expert advising an Engineering Manager.
Rules:
- Use ONLY numbers present in FEATURES or CONTEXT. Never invent figures.
- Each evidence item names the metric, its value (copied exactly) and the segment it comes from.
- Evidence from CONTEXT has source="rag" and ref=<the chunk id in brackets>.
- A recommendation names WHO does WHAT by WHEN, or a measurable target.
- confidence_score: 0.9+ only with large samples and a large effect; below 0.5 when n < 15.
- alternative: a cheaper or faster way to reach a similar decision, and its trade-off.
- SIGNALS lists the findings that pass this agent's statistical thresholds (sample size, effect size,
  policy limits), computed in code. Report ONLY those findings: you may merge or reword them, explain
  likely causes using CONTEXT, and sharpen the recommendation. Do not raise issues that are not in
  SIGNALS: a gap in a small sample or inside policy is noise, not a finding.
- If SIGNALS says no action is needed, return exactly one low-severity insight saying so, with the
  numbers that show it.
- Return 1-3 insights, most impactful first."""

SEVERITY = {"low": 0, "medium": 1, "high": 2}


def signals_text(sig: LLMInsights) -> str:
    lines = []
    for i in sig.insights:
        evs = "; ".join(f"{e.metric}={e.value} ({', '.join(f'{k}={v}' for k, v in e.segment.items())})"
                        for e in i.evidence if e.source == "data")
        lines.append(f"- [{i.severity}] {i.title} | {evs}")
    return "\n".join(lines) or "- (none)"


def guard(insights: list[Insight], sig: LLMInsights) -> list[Insight]:
    """Code has the last word on severity: an LLM finding can't be more severe than the worst detected
    signal, and a no-action run stays low. Stops a real model raising alarms on a healthy pipeline."""
    top = max((SEVERITY[i.severity] for i in sig.insights), default=0)
    name = {v: k for k, v in SEVERITY.items()}
    for ins in insights:
        if SEVERITY[ins.severity] > top:
            ins.severity = name[top]
    return insights[:max(1, len(sig.insights))]


class BaseInsightAgent:
    name: str = "base"
    rag_query: str = ""
    causal: bool = False
    stakes: str = "low"
    schema = LLMInsights

    # ---- per-agent hooks -------------------------------------------------
    def features(self, data: HiringData, compress: bool = False):
        raise NotImplementedError

    def role_filter(self, data: HiringData) -> str | None:
        return None

    def analyze(self, feats, chunks: tuple) -> LLMInsights:
        """Rule-based analyst. Powers the offline mock backend and the 'rules only' alternative."""
        raise NotImplementedError

    def prompt(self) -> str:
        return (Path(__file__).resolve().parent.parent / "prompts" / f"{self.name}.md").read_text()

    # ---- shared pipeline -------------------------------------------------
    def run(self, data: HiringData, run_id: str, *, force_tier: str | None = None, compress: bool = False,
            use_cache: bool = False, last_eval: float | None = None, rag_k: int | None = None) -> AgentOutput:
        t0 = time.perf_counter()
        llm_ms = 0
        try:
            feats = self.features(data, compress=compress)
            feats_txt = to_text(feats)
            k = rag_k or (2 if compress else 4)
            chunks = retrieve(self.rag_query, self.name, self.role_filter(data), k)
            ctx = "\n".join(f"[{cid}] {txt}" for cid, txt in chunks) or "(no relevant context found)"
            # detect in code, explain with the LLM: the rule-based analyst is the detector
            # (SIGNALS=off sends the raw tables only, to show what a model does without it)
            use_signals = os.getenv("SIGNALS", "on") != "off"
            sig = self.analyze(feats, chunks)
            sig_block = f"\n\nSIGNALS (passed thresholds in code):\n{signals_text(sig)}" if use_signals else ""
            profile = TaskProfile(self.name, context_tokens=len(feats_txt + ctx) // 4,
                                  n_segments=feats_txt.count("\n"), needs_causal_reasoning=self.causal,
                                  stakes=self.stakes, last_eval_score=last_eval)
            tier, model = route(profile, run_id, force_tier)
            parsed, cost = call_structured(
                agent=self.name, run_id=run_id, model=model,
                system=SYSTEM_COMMON + "\n\n" + self.prompt(),
                user=f"FEATURES (computed from ATS data):\n{feats_txt}{sig_block}\n\nCONTEXT (retrieved):\n{ctx}",
                schema=self.schema, mock_fn=lambda: sig, use_cache=use_cache)
            llm_ms = cost.latency_ms
            insights = [self._calibrate(i) for i in parsed.insights]
            if use_signals:
                insights = guard(insights, sig)
            return AgentOutput(agent=self.name, run_id=run_id, insights=insights, cost_of_insight=cost,
                               retrieved_chunks=[c for c, _ in chunks],
                               compute_ms=int((time.perf_counter() - t0) * 1000) - (0 if cost.simulated else llm_ms),
                               status="ok" if chunks else "degraded")
        except Exception as e:                        # one agent failing never kills the run
            return AgentOutput(agent=self.name, run_id=run_id, status="failed",
                               compute_ms=int((time.perf_counter() - t0) * 1000),
                               error=f"{type(e).__name__}: {e}")

    @staticmethod
    def _calibrate(ins: Insight) -> Insight:
        """Cap self-reported confidence by evidence strength (no 0.9 on one data point)."""
        n_ev = len(ins.evidence)
        has_rag = any(e.source == "rag" for e in ins.evidence)
        cap = min(0.95, 0.5 + 0.1 * n_ev + (0.1 if has_rag else 0))
        ins.confidence_score = round(min(ins.confidence_score, cap), 2)
        return ins


# helpers for the rule-based analysts
def ev(metric: str, value, **segment) -> Evidence:
    return Evidence(metric=metric, value=value, segment={k: str(v) for k, v in segment.items()})


def rag_ev(chunks: tuple, prefer: str | None = None) -> list[Evidence]:
    if not chunks:
        return []
    cid, txt = next(((c, t) for c, t in chunks if prefer and prefer in c), chunks[0])
    body = " ".join(l for l in txt.splitlines() if not l.startswith("#")).strip()
    return [Evidence(metric="benchmark", value=body[:160], source="rag", ref=cid)]


def conf_from(n: int, effect: float) -> float:
    """Simple confidence: more data and bigger effects -> higher, capped later by _calibrate."""
    base = 0.45 + min(0.3, n / 1000) + min(0.25, effect / 4)
    return round(min(0.95, base), 2)
```

## Step 8.3 — Prompts

One file per agent, versioned in git. The common rules (use only given numbers, cite chunk ids, name who/what/when) live in `SYSTEM_COMMON` in `base.py`.

**File:** `hiring_intel/prompts/sourcing.md`
```markdown
TASK: Which channels give the best candidates for which roles, and which should be deprioritised?
- Compare hire_rate across sources within each role; ignore cells with fewer than 20 applicants.
- Flag a channel only when the gap is large AND the sample supports it (say so in confidence).
- Check whether one channel dominates applicant volume while converting poorly.
- Compare against the sourcing benchmarks in CONTEXT when present (cite them as rag evidence).
- Recommendations move recruiter hours or budget between named channels with a re-check point.
```

**File:** `hiring_intel/prompts/rejection.md`
```markdown
TASK: Find where and why candidates are rejected, by role and stage, and whether the JD or the interviewing is the problem.
- High rejection at `screen` with reason "JD mismatch" or "experience gap" -> JD or sourcing problem.
- High rejection at technical stages concentrated in one source (see tech1_by_source) -> sourcing quality problem.
- High rejection at technical stages across all sources -> bar or interviewer calibration problem.
- Compare with prior rejection history in CONTEXT when available.
Start each title with the root cause in brackets: [jd], [sourcing], [interview_calibration] or [unclear].
```

**File:** `hiring_intel/prompts/panel_load.md`
```markdown
TASK: Who is overloaded with interviews, who is under-used, and how should load be rebalanced?
- Overloaded = per_week above the policy limit (4) AND above 2x the panel median.
- Under-used = per_week below 0.4x the median.
- Fill rebalance_plan with concrete moves (from_interviewer, to_interviewer, slots_per_week), same team/level when possible.
- Cite the load policy from CONTEXT.
```

**File:** `hiring_intel/prompts/offers.md`
```markdown
TASK: Why are offers declined: compensation, timing or competing offers?
- Compare decline_rate across ctc ratio bands (<0.9, 0.9-1.0, >1.0). A large gap at <0.9 points to compensation.
- Use decline_reasons (some reasons were classified from free-text notes by OpenJev) to confirm.
- Quantify the threshold and propose a floor or approval-speed change with an owner.
- These recommendations change compensation: be conservative with confidence.
```

**File:** `hiring_intel/prompts/pipeline.md`
```markdown
TASK: How fast is the funnel moving, where are SLA breaches, and which requisitions have been open too long?
- Report the worst stage x role breaches (breach_rate = share of candidates over the stage SLA).
- Report aging requisitions by role (open beyond the open-days limit).
- One fix per breach with an owner and a date.
```

## Verify

<!-- verify:08 -->
```bash
python - <<'EOF'
from hiring_intel.agents.router import TaskProfile, route, score
cases = [TaskProfile("panel_load", 500, 16, False, "low"),
         TaskProfile("offers", 1100, 12, True, "high"),
         TaskProfile("rejection", 1500, 40, True, "low", last_eval_score=0.5)]
for p in cases:
    print(p.agent, "score", score(p), "->", route(p, "ch08"))
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
panel_load score 0.0 -> ('small', 'gpt-4o-mini')
offers score 0.4 -> ('large', 'gpt-4o')
rejection score 0.65 -> ('large', 'gpt-4o')
```

## Checkpoint

- [ ] a low-stakes threshold task routes small
- [ ] a borderline causal task is decided by OpenJev (see the `routing` table's `decided_by`)
- [ ] a poorly scoring agent escalates to large
