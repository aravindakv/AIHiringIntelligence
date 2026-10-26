# 16 — External validation with LangSmith

**Goal:** trace every run, host the golden set as a LangSmith dataset, and record each variant as an experiment you can compare side by side.

**Time:** 45 minutes (plus account setup).

## Concepts

- **Why external.** The Evaluation Agent is your own code grading your own agents. LangSmith adds:
  - traces you can open call by call;
  - versioned experiments over a fixed dataset;
  - a comparison view that shows which cases changed between two prompt or model versions.
- **Tracing.** With `LANGSMITH_TRACING=true`, LangGraph nodes are traced automatically. `call_structured` is decorated with `@traceable` (chapter 04) and tags each span with agent, tier, tokens, USD and latency, so you can filter traces by cost.
- **Experiments.** `evaluate(target, data=dataset, evaluators=[...])` runs the pipeline for each case and records one experiment. Evaluators:

| Evaluator | Score |
| --- | --- |
| `golden_pass` | 1 if the case's expectation holds |
| `numbers_grounded` | share of evidence numbers found in the features |
| `eval_agent_overall` | the in-repo evaluator's score, to check it agrees with the golden check |
| `run_cost_usd`, `run_e2e_seconds` | cost and latency of the run |
| `golden_pass_rate` (summary) | share of cases passed |

- **Offline mode** runs the same target and evaluators with no LangSmith client, so nothing leaves your machine. CI uses it as a gate.

## Step 16.1 — Account and keys

1. Sign up at smith.langchain.com; create an API key under Settings → API Keys.
2. In `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=hiring-intel
# EU accounts: LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

## Step 16.2 — The LangSmith module

**File:** `hiring_intel/evals/langsmith_eval.py`
```python
"""LangSmith: tracing, a hosted golden dataset, and experiments you can compare in the UI.

Tracing   set LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT. LangGraph nodes are traced
          automatically; `llm.call_structured` is wrapped with @traceable, so each LLM call shows its
          model, tokens, cost and latency inside the run tree.
Dataset   push_golden_dataset() uploads data/golden/golden_set.jsonl (18 cases).
Experiment run_experiment() runs the pipeline per case and scores it with the evaluators below.
          Each variant x size becomes one experiment; compare them side by side in LangSmith.
Offline   run_offline() scores the same cases with the same evaluators locally, no account needed.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from hiring_intel.llm import ROOT

GOLDEN = ROOT / "data/golden/golden_set.jsonl"
DATASET = os.getenv("LANGSMITH_DATASET", "hiring-intel-golden")
SEV = {"low": 0, "medium": 1, "high": 2}


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def _cases() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def push_golden_dataset(name: str = DATASET):
    """Create (or reuse) the LangSmith dataset holding the golden cases."""
    from langsmith import Client
    client = Client()
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    ds = client.create_dataset(name, description="Hiring Intelligence golden set: scenario, agent, expectation")
    client.create_examples(dataset_id=ds.id, examples=[
        {"inputs": {"case_id": c["id"], "scenario": c["scenario"], "agent": c["agent"]},
         "outputs": {"expect": c["expect"]}, "metadata": {"scenario": c["scenario"]}} for c in _cases()])
    return ds


# ---------------------------------------------------------------- target
_failed: dict[tuple, Exception] = {}


def _pipeline(scenario: str, size: str, variant: str) -> str:
    """One pipeline run per scenario, shared by its cases. A failure is remembered too, so the other
    cases of that scenario report the same error instead of re-running (and re-paying for) the pipeline."""
    key = (scenario, size, variant)
    if key in _failed:
        raise _failed[key]
    try:
        return _pipeline_ok(scenario, size, variant)
    except Exception as e:
        _failed[key] = e
        raise


@lru_cache(maxsize=64)
def _pipeline_ok(scenario: str, size: str, variant: str) -> str:
    from hiring_intel.graph import run
    from scripts.benchmark import VARIANTS
    kw = {k: v for k, v in VARIANTS[variant].items() if k != "repeat"}
    s = run(f"data/raw/{scenario}_{size}", scenario=scenario, size=size, variant=f"ls-{variant}", **kw)
    from hiring_intel.observability import query
    usd = float(query("SELECT COALESCE(SUM(usd),0) u FROM llm_calls WHERE run_id=?", (s["run_id"],)).u.iloc[0])
    from hiring_intel.report import e2e_latency_ms
    return json.dumps({"outputs": s["outputs"], "evals": s["evals"], "review": s.get("review", []),
                       "usd": usd, "e2e_ms": e2e_latency_ms(s["run_id"])["e2e_ms"]}, default=str)


def make_target(size: str, variant: str):
    def target(inputs: dict) -> dict:
        st = json.loads(_pipeline(inputs["scenario"], size, variant))
        out = st["outputs"].get(inputs["agent"], {})
        ev = st["evals"].get(inputs["agent"], {})
        pend = [r for r in st["review"] if r["agent"] == inputs["agent"] and r["status"] == "pending"]
        return {"insights": [{k: i[k] for k in ("title", "severity", "confidence_score", "recommendation")}
                             for i in out.get("insights", [])],
                "eval_overall": ev.get("overall"), "grounded": ev.get("numbers_grounded"),
                "held_for_review": len(pend), "run_usd": st["usd"], "run_e2e_ms": st["e2e_ms"]}
    return target


# ---------------------------------------------------------------- evaluators
def golden_check(outputs: dict, reference_outputs: dict) -> dict:
    exp, ins = reference_outputs["expect"], outputs["insights"]
    titles = [i["title"] for i in ins]
    sev = max([SEV[i["severity"]] for i in ins] or [0])
    ok, why = True, "ok"
    if "title_any" in exp and not any(any(t in x for t in exp["title_any"]) for x in titles):
        ok, why = False, f"no title matches {exp['title_any']}"
    elif "min_severity" in exp and sev < SEV[exp["min_severity"]]:
        ok, why = False, f"severity below {exp['min_severity']}"
    elif "max_severity" in exp and sev > SEV[exp["max_severity"]]:
        ok, why = False, f"severity above {exp['max_severity']}"
    elif any(any(b in x for b in exp.get("must_not_any", [])) for x in titles):
        ok, why = False, "forbidden title present"
    return {"key": "golden_pass", "score": int(ok), "comment": why}


def grounding(outputs: dict) -> dict:
    return {"key": "numbers_grounded", "score": outputs.get("grounded") or 0.0}


def eval_agent_score(outputs: dict) -> dict:
    return {"key": "eval_agent_overall", "score": outputs.get("eval_overall") or 0.0}


def cost_usd(outputs: dict) -> dict:
    return {"key": "run_cost_usd", "score": outputs.get("run_usd", 0.0)}


def latency_s(outputs: dict) -> dict:
    return {"key": "run_e2e_seconds", "score": round(outputs.get("run_e2e_ms", 0) / 1000, 2)}


def pass_rate(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    """Summary evaluator: share of cases that passed."""
    passed = sum(golden_check(o, r)["score"] for o, r in zip(outputs, reference_outputs))
    return {"key": "golden_pass_rate", "score": passed / max(1, len(outputs))}


EVALUATORS = [golden_check, grounding, eval_agent_score, cost_usd, latency_s]


# ---------------------------------------------------------------- before spending money
def estimate_usd(size: str, variant: str) -> float | None:
    """Expected LLM spend for one experiment: one pipeline run per scenario in the golden set, priced from
    the last benchmark (config prices, mock token estimates; real token counts usually run higher)."""
    import pandas as pd
    f = ROOT / "data/eval/benchmark.csv"
    if not f.exists():
        return None
    b = pd.read_csv(f)
    b = b[(b.variant == variant) & (b["size"] == size)]
    scen = {c["scenario"] for c in _cases()}
    per = b[b.scenario.isin(scen)].groupby("scenario").total_usd.mean()
    return round(float(per.sum()), 4) if len(per) == len(scen) else None


def preflight(size: str, variant: str, max_usd: float | None = None) -> float | None:
    """Fail in about a second, before an experiment exists, if the provider can't serve the run or the
    estimate is over budget."""
    from hiring_intel.llm import backend, preflight as llm_preflight
    from hiring_intel.openjev import get_backend
    get_backend()                       # OPENJEV_BACKEND=http with the service down fails here, not mid-run
    est = estimate_usd(size, variant)
    if backend() != "mock":
        if max_usd is not None and est is not None and est > max_usd:
            raise SystemExit(f"estimated spend ${est} for {variant}/{size} is over --max-usd {max_usd}; "
                             "raise the limit or pick a cheaper variant/size")
        llm_preflight()
    return est


# ---------------------------------------------------------------- run
def run_offline(size: str = "medium", variant: str = "optimized") -> list[dict]:
    """Same target and evaluators as the LangSmith experiment, run locally with no LangSmith client,
    so nothing leaves the machine. Use it in CI or before you have an account."""
    target = make_target(size, variant)
    rows = []
    for c in _cases():
        try:
            out = target({"case_id": c["id"], "scenario": c["scenario"], "agent": c["agent"]})
        except Exception as e:                       # report it as an error, not as a failed case
            rows.append({"case_id": c["id"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        ref = {"expect": c["expect"]}
        scores = {r["key"]: r for r in (golden_check(out, ref), grounding(out), eval_agent_score(out),
                                        cost_usd(out), latency_s(out))}
        rows.append({"case_id": c["id"], **{k: v["score"] for k, v in scores.items()},
                     "comment": scores["golden_pass"]["comment"]})
    return rows


def run_experiment(size: str = "medium", variant: str = "optimized", max_usd: float | None = None):
    """Upload the dataset (once) and record one experiment in LangSmith."""
    preflight(size, variant, max_usd)
    from langsmith import evaluate
    return evaluate(make_target(size, variant), data=push_golden_dataset().name, evaluators=EVALUATORS,
                    summary_evaluators=[pass_rate], experiment_prefix=f"{variant}-{size}",
                    metadata={"variant": variant, "size": size, "llm_backend": os.getenv("LLM_BACKEND", "mock")},
                    max_concurrency=1)
```

## Step 16.3 — CLI

**File:** `scripts/langsmith_eval.py`
```python
"""Run the golden set as a LangSmith experiment.

    # online: needs LANGSMITH_API_KEY; creates dataset 'hiring-intel-golden' once, one experiment per call.
    # Real LLM calls cost money: prints an estimate, refuses above --max-usd (default 1.0), and checks the
    # provider with one tiny call before creating the experiment.
    python scripts/langsmith_eval.py --variant optimized --size medium
    python scripts/langsmith_eval.py --variant all_large --size medium     # then compare both in the UI

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
```

## Step 16.4 — Run experiments and compare

```bash
python scripts/langsmith_eval.py --variant all_large --size medium
python scripts/langsmith_eval.py --variant optimized --size medium
```

**These make real LLM calls and cost real money.** The script prints an estimate first: about $0.38 for `all_large` and $0.07 for `optimized` on `medium` inputs, at the prices in `config/models.yaml`. Real token counts usually run higher. It refuses to start above `--max-usd` (default 1.0). It also makes one tiny call first, so an exhausted quota, a spend limit or a bad key stops it in about a second, before an experiment is created. If the provider refuses a call mid-run, the run stops (`LLMUnavailable`) and is not retried. Errored cases print as `ERR` and are left out of the pass rate: a run that never reached the model is not a 0/18.

In LangSmith: **Datasets & Experiments → hiring-intel-golden → Experiments**, tick both, **Compare**. Each row is a case; the columns are the evaluator scores per experiment. Open any cell to jump to its trace. This view is your before/after evidence for the optimization report.

## Verify (offline)

<!-- verify:16 -->
```bash
python scripts/langsmith_eval.py --offline --size medium | tail -6
python scripts/langsmith_eval.py --offline --size small | tail -1
```

Expected output (captured on a fresh extract, mock backend):

```text
i-pip         1      1.0  0.97
s-src         1      1.0  0.97
s-rej         1      1.0  0.97
s-pan         1      1.0  0.97
golden pass rate: 18/18 scored
golden pass rate: 15/18 scored
```

## Checkpoint

- [ ] offline: 18/18 on medium, 15/18 on small, matching chapter 15
- [ ] online: the dataset exists, two experiments are recorded, and the comparison view shows them
- [ ] traces show `call_structured` spans with cost metadata
