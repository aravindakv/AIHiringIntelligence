"""CLI run.

    python scripts/run_pipeline.py --data data/raw/baseline_medium
    python scripts/run_pipeline.py --data data/raw/comp_crunch_small --hitl interrupt   # review in the terminal
    python scripts/run_pipeline.py --transport a2a      # agents over A2A (start the A2A server first)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hiring_intel.graph import resume, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/raw/baseline_medium")
ap.add_argument("--variant", default="optimized")
ap.add_argument("--hitl", default="queue", choices=["queue", "interrupt", "simulate"])
ap.add_argument("--transport", default="local", choices=["local", "a2a"])
a = ap.parse_args()

from scripts.benchmark import VARIANTS  # noqa: E402

kw = {k: v for k, v in VARIANTS[a.variant].items() if k != "repeat"}
name = Path(a.data).name
scenario, size = name.rsplit("_", 1) if "_" in name else (name, "")
s = run(a.data, scenario=scenario, size=size, variant=a.variant, hitl_mode=a.hitl, transport=a.transport, **kw)
run_id = s["run_id"]

while "__interrupt__" in s:                     # human in the loop, terminal edition
    pending = s["__interrupt__"][0].value["pending"]
    decisions = {}
    for p in pending:
        ins = p["insight"]
        print(f"\n[{ins['severity']}] {ins['title']}  ({p['agent']}, conf {ins['confidence_score']})")
        print("  ", ins["recommendation"])
        print("   why review:", "; ".join(p["reasons"]))
        ans = input("   approve / reject / edit? [a/r/e] ").strip().lower()[:1] or "a"
        d = {"action": {"a": "approved", "r": "rejected", "e": "edited"}[ans], "reviewer": "cli"}
        if ans == "e":
            d["edited_recommendation"] = input("   new recommendation: ")
        decisions[p["item_id"]] = d
    s = resume(run_id, decisions)

print(f"\nrun {run_id} done; errors: {s.get('errors')}")
for agent, o in s["outputs"].items():
    for i in o["insights"]:
        print(f"  {agent:11} [{i['severity']:6}] {i['title']}")