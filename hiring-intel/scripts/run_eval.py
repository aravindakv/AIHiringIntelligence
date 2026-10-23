"""Golden-set evaluation. Runs the pipeline once per scenario (default: medium size) and checks each case.

    python scripts/run_eval.py --size medium --variant optimized
Writes data/eval/golden_<variant>_<size>.csv and prints the pass rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hiring_intel.graph import run  # noqa: E402

from scripts.benchmark import VARIANTS  # noqa: E402

SEV = {"low": 0, "medium": 1, "high": 2}


def check(insights: list[dict], expect: dict) -> tuple[bool, str]:
    titles = [i["title"] for i in insights]
    sev = [SEV[i["severity"]] for i in insights] or [0]
    if "title_any" in expect and not any(any(t_ in t for t_ in expect["title_any"]) for t in titles):
        return False, f"no title matches {expect['title_any']}"
    if "min_severity" in expect and max(sev) < SEV[expect["min_severity"]]:
        return False, f"severity below {expect['min_severity']}"
    if "max_severity" in expect and max(sev) > SEV[expect["max_severity"]]:
        return False, f"severity above {expect['max_severity']}"
    if any(any(b in t for b in expect.get("must_not_any", [])) for t in titles):
        return False, f"forbidden title {expect['must_not_any']}"
    return True, "ok"


def golden_score(size: str = "medium", variant: str = "optimized", run_kwargs: dict | None = None) -> pd.DataFrame:
    cases = [json.loads(l) for l in Path("data/golden/golden_set.jsonl").read_text().splitlines()]
    kw = dict(VARIANTS[variant]) if run_kwargs is None else run_kwargs
    kw.pop("repeat", None)
    rows, cache = [], {}
    for c in cases:
        if c["scenario"] not in cache:
            cache[c["scenario"]] = run(f"data/raw/{c['scenario']}_{size}", scenario=c["scenario"], size=size,
                                       variant=f"golden-{variant}", hitl_mode="queue", **kw)
        s = cache[c["scenario"]]
        out = s["outputs"].get(c["agent"], {})
        ok, why = check(out.get("insights", []), c["expect"])
        rows.append({"id": c["id"], "scenario": c["scenario"], "agent": c["agent"], "pass": ok, "why": why,
                     "eval_overall": s["evals"].get(c["agent"], {}).get("overall"),
                     "top_title": (out.get("insights") or [{}])[0].get("title")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium")
    ap.add_argument("--variant", default="optimized", choices=list(VARIANTS))
    a = ap.parse_args()
    df = golden_score(a.size, a.variant)
    Path("data/eval").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"data/eval/golden_{a.variant}_{a.size}.csv", index=False)
    print(df.to_string(index=False))
    print(f"\npass rate: {df['pass'].mean():.0%} ({df['pass'].sum()}/{len(df)})")