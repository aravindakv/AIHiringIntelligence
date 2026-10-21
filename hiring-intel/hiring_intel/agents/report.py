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
    row = {"run_id": run_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenario": cfg.get("scenario", ""),
               "size": cfg.get("size", ""), "variant": cfg.get("variant", ""), "data_path": cfg["data_path"],
               "n_candidates": len(data.candidates), "total_usd": round(float(calls.usd.sum()), 6),
               "llm_calls": len(calls), "input_tokens": int(calls.input_tokens.sum()),
               "output_tokens": int(calls.output_tokens.sum()), "e2e_latency_ms": lat["e2e_ms"],
               "wall_ms": int((time.perf_counter() - cfg.get("started", time.perf_counter())) * 1000),
               "mean_eval": round(sum(e["overall"] for e in evals.values()) / max(1, len(evals)), 3),
               "hitl_pending": sum(1 for r in review if r["status"] == "pending"),
               "hitl_auto": sum(1 for r in review if r["status"] == "auto_published"),
               "status": "ok" if not s.get("errors") else "partial"}
    execute("INSERT OR REPLACE INTO runs VALUES (:run_id,:ts,:scenario,:size,:variant,:data_path,:n_candidates,"
            ":total_usd,:llm_calls,:input_tokens,:output_tokens,:e2e_latency_ms,:wall_ms,:mean_eval,"
            ":hitl_pending,:hitl_auto,:status)", row)
    return row