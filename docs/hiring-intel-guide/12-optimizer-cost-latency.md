# 12 — Optimization Agent, cost and latency

**Goal:** define cost and latency precisely, compute them per agent and per run, and have an agent that proposes cheaper or faster options from the numbers.

**Time:** 40 minutes.

## Concepts

**Cost**

```latex
\text{USD}_{call} = \frac{T_{in}\cdot P_{in} + T_{out}\cdot P_{out}}{10^{6}}
```

Agent cost is the sum of its calls, including repairs and retries. Run cost is the sum of every call. OpenJev decisions and cache hits cost $0 API.

**Latency**

- **Call latency:** request to parsed response (simulated per tier in mock mode).
- **Agent latency:** local compute + LLM latency.
- **End-to-end (E2E):** the graph's critical path, `ingest + enrich + max(agents) + evaluate + max(retries) + review + optimize`. Parallel stages count only their slowest branch.
- **Wall time:** what the process measured. In mock mode it excludes simulated LLM waits.

**Optimization Agent:** reads this run's `llm_calls`, applies rules (a large-tier agent that scores ≥ 0.85 → try small; > 2,500 input tokens → compress; many LLM labelling calls → OpenJev; repeated calls with no cache hits → cache), then has the small model phrase 1–3 suggestions in the shared contract.

## Step 12.1 — Optimization Agent

**File:** `hiring_intel/agents/optimizer.py`
```python
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
```

## Step 12.2 — Cost and latency accounting

**File:** `hiring_intel/report.py`
```python
"""Cost and latency accounting. Definitions live in docs/COST_AND_LATENCY.md; this file implements them.

  call cost (USD)      = input_tokens  x input_price_per_1M / 1e6 + output_tokens x output_price_per_1M / 1e6
  agent cost           = sum of its call costs, including JSON-repair retries and router calls it triggered
  run cost             = sum of all call costs in the run (cache hits count as $0)
  OpenJev decisions    = $0 API cost; their compute time counts toward latency

  call latency         = request -> parsed response (simulated from config in mock mode)
  agent latency        = local compute (features, retrieval, OpenJev) + its LLM latency
  end-to-end latency   = critical path of the graph:
                         ingest + enrich + max(agent latencies) + evaluate + max(retries) + review
                         + optimize   (parallel stages contribute their slowest branch only)
  wall time            = measured clock time of this process (mock mode: excludes simulated LLM waits)
"""
from __future__ import annotations

import time

from hiring_intel.ingest import load_all
from hiring_intel.observability import execute, query


def e2e_latency_ms(run_id: str) -> dict:
    t = query("SELECT stage, agent, compute_ms, llm_ms FROM timings WHERE run_id=?", (run_id,))
    calls = query("SELECT agent, latency_ms FROM llm_calls WHERE run_id=?", (run_id,))
    calls["agent"] = calls.agent.str.split(":").str[0]
    by_stage = {}
    for stage in ["ingest", "enrich", "evaluate", "human_review", "optimize"]:
        by_stage[stage] = int(t[t.stage == stage].compute_ms.sum())
    # LLM work attributed to sequential stages
    by_stage["enrich"] += int(calls[calls.agent == "enrich"].latency_ms.sum() / 8)   # 8 concurrent requests
    by_stage["evaluate"] += int(calls[calls.agent == "evaluator"].latency_ms.sum())
    by_stage["optimize"] += int(calls[calls.agent == "optimizer"].latency_ms.sum())
    router = int(calls[calls.agent == "router"].latency_ms.max()) if (calls.agent == "router").any() else 0
    ag = t[t.stage == "agents"]
    by_stage["agents"] = int((ag.compute_ms + ag.llm_ms).max()) + router if len(ag) else 0
    rt = t[t.stage == "retry"]
    by_stage["retry"] = int((rt.compute_ms + rt.llm_ms).max()) if len(rt) else 0
    per_agent = {r.agent: int(r.compute_ms + r.llm_ms) for r in ag.itertuples()}
    return {"e2e_ms": sum(by_stage.values()), "by_stage": by_stage, "per_agent": per_agent}


def summarize_run(run_id: str, s: dict) -> dict:
    cfg = s["cfg"]
    calls = query("SELECT * FROM llm_calls WHERE run_id=?", (run_id,))
    lat = e2e_latency_ms(run_id)
    evals = s.get("evals", {})
    review = s.get("review", [])
    data = load_all(cfg["data_path"])
    row = dict(run_id=run_id, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), scenario=cfg.get("scenario", ""),
               size=cfg.get("size", ""), variant=cfg.get("variant", ""), data_path=cfg["data_path"],
               n_candidates=len(data.candidates), total_usd=round(float(calls.usd.sum()), 6),
               llm_calls=int(len(calls)), input_tokens=int(calls.input_tokens.sum()),
               output_tokens=int(calls.output_tokens.sum()), e2e_latency_ms=lat["e2e_ms"],
               wall_ms=int((time.perf_counter() - cfg.get("started", time.perf_counter())) * 1000),
               mean_eval=round(sum(e["overall"] for e in evals.values()) / max(1, len(evals)), 3),
               hitl_pending=sum(1 for r in review if r["status"] == "pending"),
               hitl_auto=sum(1 for r in review if r["status"] == "auto_published"),
               status="ok" if not s.get("errors") else "partial")
    execute("INSERT OR REPLACE INTO runs VALUES (:run_id,:ts,:scenario,:size,:variant,:data_path,:n_candidates,"
            ":total_usd,:llm_calls,:input_tokens,:output_tokens,:e2e_latency_ms,:wall_ms,:mean_eval,"
            ":hitl_pending,:hitl_auto,:status)", row)
    return row
```

## Step 12.3 — Write the definitions down

The capstone grades whether cost is tracked per agent; reviewers will ask how. This doc is the answer.

**File:** `docs/COST_AND_LATENCY.md`
````markdown
# Cost and latency: definitions

Implemented in `hiring_intel/llm.py` (per call) and `hiring_intel/report.py` (per agent, per run). Every number on the dashboard follows these definitions.

## Cost

| Term | Definition |
| --- | --- |
| **Call cost (USD)** | `input_tokens × input_price_per_1M / 1,000,000 + output_tokens × output_price_per_1M / 1,000,000` |
| Prices | `config/models.yaml` per tier (falls back to LiteLLM's price table for unknown models). Examples only: update from your provider before quoting USD. |
| Tokens | Real runs: the provider's `usage` counts. Mock runs: estimated as `characters / 4`. |
| **Agent cost** | Sum of the agent's call costs, including JSON-repair retries and its retry on the large tier. |
| **Run cost** | Sum of every call in the run: insight agents, router LLM fallbacks, note labelling (if done by LLM), judge, optimizer. |
| Cache hit | $0 and ~2 ms; still logged so you can count hits. |
| **OpenJev decision** | $0 API cost. It uses local CPU/GPU time, which counts toward latency. Hardware cost is outside the USD figure; note it if you self-host on paid GPUs. |
| **Cost of insight** | The `cost_of_insight` field on each `AgentOutput`: model, tier, tokens, USD, latency for the call that produced it. |
| **LLM-equivalent cost** | For OpenJev labelling, what the same job would cost on the small LLM (`enrich.llm_equivalent_usd`), so the saving is visible. |

```latex
\text{USD}_{call} = \frac{T_{in}\cdot P_{in} + T_{out}\cdot P_{out}}{10^{6}}
```

## Latency

| Term | Definition |
| --- | --- |
| **Call latency** | Request sent to parsed, validated response. Real runs: wall clock. Mock runs: **simulated** as `base_ms + ms_per_out_token × out + ms_per_1k_in × in/1000`, ×0.85–1.25 deterministic jitter, per tier. |
| **Compute latency** | Local work: pandas features, retrieval, OpenJev, validation. Always measured. |
| **Agent latency** | compute + its LLM latency. |
| **End-to-end (E2E) latency** | Critical path through the graph: `ingest + enrich + max(agent) + evaluate + max(retry) + human_review + optimize`. Parallel stages count only their slowest branch. LLM note labelling assumes 8 concurrent requests. |
| **Wall time** | Measured clock time of the process. In mock mode it excludes simulated LLM waits, so it shows pure compute. |
| p50 / p95 | Per agent across runs, from the `llm_calls` table. |
| Human review time | Not counted in E2E. With `hitl_mode=queue` the run completes and review is async; with `interrupt` the run pauses until resumed. |

Simulated latency parameters (from `config/models.yaml`):

| Tier | base | per output token | per 1k input tokens |
| --- | --- | --- | --- |
| small | 350 ms | 9 ms | 40 ms |
| large | 700 ms | 22 ms | 90 ms |

## Ask (per question)

An Ask question is one OpenJev topic decision plus one LLM call: about $0.0004–0.001 on the small tier and $0.007–0.012 on the large tier at the example prices, with 1.3–1.9k input tokens. Questions about a person, out of scope or invalid cost $0 (no model call). Each question logs a `questions` row with USD and latency; its LLM call is in `llm_calls` under the `ask_id`. See [ASK.md](ASK.md).

## Where to see them

- Dashboard → **Cost & latency**: USD per agent and tier, critical-path latency by stage, routing decisions, OpenJev decisions.
- Dashboard → **Compare inputs**: run cost and E2E latency per input size and variant.
- Dashboard → **Ask**: cost, latency and grounding per question; "Recent questions" history.
- `data/runs.db` tables: `llm_calls`, `timings`, `decisions`, `routing`, `runs`, `questions`.
- `docs/OPTIMIZATION_REPORT.md` for the before/after numbers.
````

## Verify

The optimizer needs a full run's data, so its verification is in chapter 13.

## Checkpoint

- [ ] you can state the E2E formula from memory
- [ ] you know which table each number comes from (`llm_calls`, `timings`, `decisions`)
