"""Optimization Agent: reads this run's cost/latency, proposes cheaper or faster options."""
from __future__ import annotations

import pandas as pd

from hiring_intel.agents.base import ev
from hiring_intel.contracts import Alternative, Insight, LLMInsights
from hiring_intel.llm import call_structured, model_for
from hiring_intel.observability import query


def run_stats(run_id: str) -> pd.DataFrame:
    df = query("SELECT * FROM llm_calls WHERE run_id = ?", (run_id,))
    if df.empty:
        return df
    df["agent"] = df.agent.str.split(":").str[0]
    return (df.groupby(["agent", "tier"]).agg(calls=("usd", "size"), usd=("usd", "sum"),
                                              in_tok=("input_tokens", "sum"), out_tok=("output_tokens", "sum"),
                                              p50_ms=("latency_ms", "median"), cached=("cached", "sum"))
            .reset_index().round({"usd": 6}))


def rule_hits(stats: pd.DataFrame, evals: dict) -> list[dict]:
    hits = []
    for r in stats.itertuples():
        e = evals.get(r.agent, {})
        overall = e.get("overall") if isinstance(e, dict) else None
        if r.tier == "large" and overall is not None and overall >= 0.85:
            hits.append({"kind": "downgrade", "agent": r.agent, "usd": r.usd, "overall": overall, "in_tok": r.in_tok})
        if r.in_tok > 2500:
            hits.append({"kind": "compress", "agent": r.agent, "usd": r.usd, "in_tok": r.in_tok})
        if r.agent == "enrich" and r.calls > 5:
            hits.append({"kind": "openjev", "agent": r.agent, "usd": r.usd, "calls": r.calls})
        if r.calls > 1 and r.cached == 0 and r.agent not in ("enrich", "evaluator", "router"):
            hits.append({"kind": "cache", "agent": r.agent, "usd": r.usd, "calls": r.calls})
    return hits


def _analyst(hits: list[dict]) -> LLMInsights:
    out = []
    for h in sorted(hits, key=lambda h: -h["usd"])[:3]:
        if h["kind"] == "downgrade":
            out.append(Insight(title=f"Trial the small model for {h['agent']}",
                               recommendation=(f"Pipeline owner: run {h['agent']} on the small tier for the next 3 runs "
                                               f"and keep it there if its eval score stays at or above 0.8 "
                                               f"(now {h['overall']})."),
                               evidence=[ev("usd", h["usd"], agent=h["agent"]), ev("in_tok", h["in_tok"], agent=h["agent"])],
                               confidence_score=0.7, severity="medium",
                               alternative=Alternative(option="Keep large tier", trade_off="Higher cost, marginal quality gain")))
        elif h["kind"] == "compress":
            out.append(Insight(title=f"Compress {h['agent']} context",
                               recommendation=(f"Pipeline owner: enable compressed features for {h['agent']} (top rows, "
                                               f"pruned columns, top_k 2); it sends {h['in_tok']} input tokens per run."),
                               evidence=[ev("in_tok", h["in_tok"], agent=h["agent"]), ev("usd", h["usd"], agent=h["agent"])],
                               confidence_score=0.75, severity="medium",
                               alternative=Alternative(option="Keep full tables", trade_off="More context, more tokens")))
        elif h["kind"] == "openjev":
            out.append(Insight(title="Move note classification to OpenJev",
                               recommendation=(f"Pipeline owner: switch note classification from {h['calls']} small-LLM "
                                               f"calls to OpenJev (local, $0 API) and keep the LLM for low-confidence notes."),
                               evidence=[ev("calls", h["calls"], agent="enrich"), ev("usd", h["usd"], agent="enrich")],
                               confidence_score=0.8, severity="high",
                               alternative=Alternative(option="Keep LLM labelling", trade_off="Simpler stack, pays per note")))
        else:
            out.append(Insight(title=f"Cache {h['agent']} responses",
                               recommendation=f"Pipeline owner: turn on the response cache for {h['agent']}; it made "
                                              f"{h['calls']} calls with no cache hits this run.",
                               evidence=[ev("calls", h["calls"], agent=h["agent"])], confidence_score=0.6, severity="low",
                               alternative=Alternative(option="No cache", trade_off="Always fresh, always paid")))
    if not out:
        out.append(Insight(title="No optimization needed", severity="low", confidence_score=0.6,
                           recommendation="No change this run: every agent is on the cheapest tier that passes eval "
                                          "and context sizes are under 2,500 tokens.",
                           evidence=[ev("agents_checked", 0)],
                           alternative=Alternative(option="n/a", trade_off="n/a")))
    return LLMInsights(insights=out)


def optimize(run_id: str, evals: dict) -> list[Insight]:
    stats = run_stats(run_id)
    if stats.empty:
        return []
    hits = rule_hits(stats, evals)
    parsed, _ = call_structured(
        agent="optimizer", run_id=run_id, model=model_for("small"),
        system="You are a cost engineer for an LLM pipeline. Turn the stats and rule hits into 1-3 concrete "
               "optimizations. Evidence = numbers from STATS. Alternative = the option you rejected.",
        user=f"STATS:\n{stats.to_csv(index=False)}\n\nRULE HITS:\n{hits}", schema=LLMInsights,
        mock_fn=lambda: _analyst(hits))
    return parsed.insights