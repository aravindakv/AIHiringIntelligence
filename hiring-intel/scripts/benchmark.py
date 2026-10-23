"""Run the pipeline across different INPUTS (scenario x size) and optimization VARIANTS.

    python scripts/benchmark.py                 # full matrix -> data/eval/benchmark.csv
    python scripts/benchmark.py --quick         # small + medium only

Variants (each adds one lever to the previous):
  all_large           every agent on the large model, full context, notes labelled by a small LLM
  routed              + lever A: Routing Agent picks small/large per agent
  routed_compressed   + lever B: compressed features, top_k 2
  optimized           + lever C: OpenJev labels notes + decides borderline routing ($0 API)
  optimized_cached    + lever D: same input re-run with the response cache warm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VARIANTS = {
    "all_large": {"force_tier": "large", "compress": False, "classify": "llm", "use_cache": False},
    "routed": {"force_tier": None, "compress": False, "classify": "llm", "use_cache": False},
    "routed_compressed": {"force_tier": None, "compress": True, "classify": "llm", "use_cache": False},
    "optimized": {"force_tier": None, "compress": True, "classify": "openjev", "use_cache": True},
    "optimized_cached": {"force_tier": None, "compress": True, "classify": "openjev", "use_cache": True, "repeat": True},
}
SCENARIOS = ["baseline", "healthy", "comp_crunch", "interviewer_burnout", "sourcing_shift"]


def save_state(s: dict) -> None:
    """Keep the full run state (outputs, evals, HITL queue, suggestions, latency by stage) for the dashboard."""
    import json

    from hiring_intel.ingest import load_all
    from hiring_intel.metrics import STAGES
    from hiring_intel.observability import query
    from hiring_intel.report import e2e_latency_ms
    d = load_all(s["cfg"]["data_path"])
    funnel = d.stage_events.groupby("stage").candidate_id.nunique().reindex(STAGES).fillna(0).astype(int).to_dict()
    routing = query("SELECT agent, tier, model, reason, decided_by FROM routing WHERE run_id=?", (s["run_id"],))
    dec = query("SELECT task, backend, COUNT(*) n, SUM(n_items) items, AVG(confidence) conf, SUM(escalated) esc "
                "FROM decisions WHERE run_id=? GROUP BY task, backend", (s["run_id"],))
    keep = {k: s.get(k) for k in ["run_id", "cfg", "enrich", "outputs", "evals", "retried", "review", "suggestions",
                                  "errors"]}
    keep.update(funnel=funnel, latency=e2e_latency_ms(s["run_id"]), routing=routing.to_dict("records"),
                decisions=dec.to_dict("records"))
    out = Path("data/eval/runs")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{s['run_id']}.json").write_text(json.dumps(keep, default=str))


def main(quick: bool = False) -> pd.DataFrame:
    from hiring_intel.graph import run
    from hiring_intel.llm import clear_cache
    from hiring_intel.observability import query
    sizes = ["small", "medium"] if quick else ["small", "medium", "large"]
    inputs = [(s, z) for s in SCENARIOS for z in sizes] + ([] if quick else [("baseline", "xlarge")])
    run_ids = []
    for scenario, size in inputs:
        path = f"data/raw/{scenario}_{size}"
        for name, cfg in VARIANTS.items():
            kw = {k: v for k, v in cfg.items() if k != "repeat"}
            if not cfg.get("repeat"):
                clear_cache()
            t0 = time.perf_counter()
            s = run(path, scenario=scenario, size=size, variant=name, hitl_mode="queue",
                    run_id=f"{name}__{scenario}__{size}", **kw)
            run_ids.append(s["run_id"])
            save_state(s)
            print(f"{scenario:20} {size:7} {name:18} {time.perf_counter() - t0:5.1f}s", flush=True)
    df = query("SELECT * FROM runs WHERE run_id IN ({})".format(",".join("?" * len(run_ids))), tuple(run_ids))
    Path("data/eval").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/eval/benchmark.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    df = main(ap.parse_args().quick)
    piv = df.pivot_table(index=["scenario", "size"], columns="variant", values="total_usd")
    print(piv.round(4).to_string())