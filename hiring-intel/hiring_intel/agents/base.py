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