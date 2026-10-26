"""Run the golden set as a LangSmith experiment.

    # online: needs LANGSMITH_API_KEY; creates dataset 'hiring-intel-golden' once, one experiment per call.
    # Real LLM calls cost money: prints an estimate, refuses above --max-usd (default 1.0), and checks the
    # provider with one tiny call before creating the experiment.

    # offline: same target and evaluators, scored locally, nothing uploaded
    python scripts/langsmith_eval.py --offline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hiring_intel.evals.langsmith_eval import estimate_usd, run_experiment, run_offline  # noqa: E402
from hiring_intel.llm import LLMUnavailable, backend  # noqa: E402
from hiring_intel.openjev import OpenJevUnavailable  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="medium")
ap.add_argument("--variant", default="optimized")
ap.add_argument("--offline", action="store_true")
ap.add_argument("--max-usd", type=float, default=1.0, help="refuse to start if the estimated spend is higher")
a = ap.parse_args()

est = estimate_usd(a.size, a.variant)
print(f"estimated LLM spend: {'$' + str(est) if est is not None else 'unknown (run scripts/benchmark.py)'} "
      f"for {a.variant}/{a.size} ({'mock: $0 actually spent' if backend() == 'mock' else 'real calls'})")

if a.offline:
    rows = run_offline(a.size, a.variant)
else:
    try:
        res = run_experiment(a.size, a.variant, max_usd=a.max_usd)
    except (LLMUnavailable, OpenJevUnavailable) as e:     # setup problem: say so in one line, create nothing
        sys.exit(f"cannot start the experiment: {e}")
    rows = []
    for r in res:
        row = {"case_id": r["example"].inputs["case_id"]}
        if getattr(r["run"], "error", None):
            row["error"] = str(r["run"].error).strip().splitlines()[-1][:200]
        else:
            row.update({e.key: e.score for e in r["evaluation_results"]["results"]})
        rows.append(row)
    print("experiment:", getattr(res, "experiment_name", "see the LangSmith UI"))

print(f"{'case':10} {'pass':>4} {'grounded':>8} {'eval':>5}")
for r in rows:
    if "error" in r:
        print(f"{r['case_id']:10} {'ERR':>4} {'-':>8} {'-':>5}")
    else:
        print(f"{r['case_id']:10} {r.get('golden_pass'):>4} {r.get('numbers_grounded'):>8} {r.get('eval_agent_overall'):>5}")
scored = [r for r in rows if "error" not in r]
errs = [r for r in rows if "error" in r]
print(f"\ngolden pass rate: {sum(1 for r in scored if r.get('golden_pass'))}/{len(scored)} scored")
if errs:
    print(f"{len(errs)} case(s) errored and were not scored. First error: {errs[0]['error']}")
    sys.exit(1)