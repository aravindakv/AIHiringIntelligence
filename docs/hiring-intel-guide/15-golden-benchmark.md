# 15 — Golden set and benchmark across inputs

**Goal:** 18 pre-validated cases that define "correct", and an 80-run benchmark over inputs × optimization variants that shows a measurable before/after for each lever.

**Time:** 1 hour (the benchmark itself runs in about 1–2 minutes).

## Concepts

- **Golden cases** name a scenario, an agent and an expectation: a title must mention X, a minimum or maximum severity, or titles that must not appear. The healthy control must raise nothing above low. Trap cases forbid the wrong root cause.
- **Variants each add one lever:**

| Variant | Lever |
| --- | --- |
| `all_large` | baseline: every agent on the large model, full context, LLM labels notes |
| `routed` | A. Routing Agent |
| `routed_compressed` | B. compressed features, top_k 2 |
| `optimized` | C. OpenJev labels notes and breaks routing ties |
| `optimized_cached` | D. same input re-run with the cache warm |

- **Report quality next to cost.** A saving that breaks the golden set isn't a saving. This is how the compression bug was found: the panel median was computed on the trimmed table, and Panel Load stopped flagging overload. A second one showed up when this guide's chapter outputs were compared: the trimmed table kept only the top 8 interviewers, so the title said "8 over policy" where the full table said 10. Both fixes are in `metrics.py` (chapter 05).

## Step 15.1 — Golden set

**File:** `scripts/make_golden_set.py`
```python
"""Write data/golden/golden_set.jsonl: 18 pre-validated scenarios with expected outputs.

Each case runs the pipeline on a scenario dataset and checks the named agent's insights.
Checks:  title_any       at least one insight title contains one of these strings
         min_severity    the top insight is at least this severe
         max_severity    no insight is above this severity (healthy control)
         must_not_any    no insight title contains these strings
"""
import json
from pathlib import Path

CASES = [
    # baseline: all planted patterns
    ("b-src", "baseline", "sourcing", {"title_any": ["Backend: referral"], "min_severity": "medium"}),
    ("b-rej", "baseline", "rejection", {"title_any": ["[jd] Mobile screen"], "min_severity": "high"}),
    ("b-rej-data", "baseline", "rejection", {"title_any": ["agency candidates fail tech_1"]}),
    ("b-pan", "baseline", "panel_load", {"title_any": ["over load policy"]}),
    ("b-off", "baseline", "offers", {"title_any": ["below 90%"]}),
    ("b-pip", "baseline", "pipeline", {"title_any": ["SRE onsite", "SRE requisitions"]}),
    # healthy control: nothing should be raised above low
    ("h-src", "healthy", "sourcing", {"max_severity": "low"}),
    ("h-rej", "healthy", "rejection", {"max_severity": "low", "must_not_any": ["[jd]"]}),
    ("h-pan", "healthy", "panel_load", {"max_severity": "low"}),
    ("h-off", "healthy", "offers", {"max_severity": "low", "must_not_any": ["below 90%"]}),
    ("h-pip", "healthy", "pipeline", {"max_severity": "low"}),
    # comp crunch
    ("c-off", "comp_crunch", "offers", {"title_any": ["below 90%"], "min_severity": "high"}),
    ("c-rej", "comp_crunch", "rejection", {"must_not_any": ["[jd] Mobile"]}),
    # interviewer burnout
    ("i-pan", "interviewer_burnout", "panel_load", {"title_any": ["over load policy"], "min_severity": "high"}),
    ("i-pip", "interviewer_burnout", "pipeline", {"title_any": ["breaches SLA"]}),
    # sourcing shift
    ("s-src", "sourcing_shift", "sourcing", {"title_any": ["agency"], "min_severity": "high"}),
    ("s-rej", "sourcing_shift", "rejection", {"title_any": ["[sourcing] agency"]}),
    ("s-pan", "sourcing_shift", "panel_load", {"title_any": ["over load policy"]}),
]

if __name__ == "__main__":
    out = Path("data/golden/golden_set.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for cid, scenario, agent, expect in CASES:
            f.write(json.dumps({"id": cid, "scenario": scenario, "agent": agent, "expect": expect}) + "\n")
    print(f"wrote {len(CASES)} cases to {out}")
```

## Step 15.2 — Golden evaluation

**File:** `scripts/run_eval.py`
```python
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
from scripts.benchmark import VARIANTS  # noqa: E402
from hiring_intel.graph import run  # noqa: E402

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
```

## Step 15.3 — Benchmark

**File:** `scripts/benchmark.py`
```python
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
    "all_large": dict(force_tier="large", compress=False, classify="llm", use_cache=False),
    "routed": dict(force_tier=None, compress=False, classify="llm", use_cache=False),
    "routed_compressed": dict(force_tier=None, compress=True, classify="llm", use_cache=False),
    "optimized": dict(force_tier=None, compress=True, classify="openjev", use_cache=True),
    "optimized_cached": dict(force_tier=None, compress=True, classify="openjev", use_cache=True, repeat=True),
}
SCENARIOS = ["baseline", "healthy", "comp_crunch", "interviewer_burnout", "sourcing_shift"]


def save_state(s: dict) -> None:
    """Keep the full run state (outputs, evals, HITL queue, suggestions, latency by stage) for the dashboard."""
    import json
    from hiring_intel.metrics import STAGES
    from hiring_intel.ingest import load_all
    from hiring_intel.report import e2e_latency_ms
    from hiring_intel.observability import query
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
    df = query("SELECT * FROM runs WHERE run_id IN (%s)" % ",".join("?" * len(run_ids)), tuple(run_ids))
    Path("data/eval").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/eval/benchmark.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    df = main(ap.parse_args().quick)
    piv = df.pivot_table(index=["scenario", "size"], columns="variant", values="total_usd")
    print(piv.round(4).to_string())
```

## Verify

<!-- verify:15 -->
```bash
python scripts/make_golden_set.py
python scripts/benchmark.py > /tmp/bench.txt 2>&1; tail -18 /tmp/bench.txt
for z in small medium; do python scripts/run_eval.py --size $z --variant optimized | tail -1; done
```

Expected output (captured on a fresh extract, mock backend):

```text
wrote 18 cases to data/golden/golden_set.jsonl
variant                     all_large  optimized  optimized_cached  routed  routed_compressed
scenario            size                                                                     
baseline            large      0.1705     0.0172            0.0003  0.1377             0.1289
                    medium     0.0797     0.0171            0.0003  0.0531             0.0449
                    small      0.0532     0.0169            0.0003  0.0273             0.0224
                    xlarge     0.4479     0.0173            0.0003  0.3969             0.3869
comp_crunch         large      0.1649     0.0142            0.0003  0.1351             0.1263
                    medium     0.0773     0.0144            0.0003  0.0507             0.0425
                    small      0.0520     0.0143            0.0003  0.0260             0.0204
healthy             large      0.1515     0.0110            0.0003  0.1297             0.1209
                    medium     0.0619     0.0110            0.0003  0.0464             0.0382
                    small      0.0381     0.0118            0.0003  0.0227             0.0177
interviewer_burnout large      0.1631     0.0144            0.0003  0.1339             0.1254
                    medium     0.0786     0.0145            0.0003  0.0503             0.0421
                    small      0.0527     0.0144            0.0003  0.0259             0.0203
sourcing_shift      large      0.1762     0.0181            0.0003  0.1407             0.1321
                    medium     0.0851     0.0180            0.0003  0.0558             0.0478
                    small      0.0567     0.0199            0.0003  0.0308             0.0264
pass rate: 83% (15/18)
pass rate: 100% (18/18)
```

Read the table row by row: with an LLM labelling notes, cost grows with input size; with OpenJev it stays flat. The small inputs fail 3 cases because noise in ~50-applicant segments creates false positives. That is a documented limitation, not a bug to hide.

## Checkpoint

- [ ] `data/eval/benchmark.csv` has 80 runs
- [ ] medium and large inputs pass 18/18
- [ ] you can explain each lever's saving and its effect on quality
