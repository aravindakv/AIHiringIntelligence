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
    """One pipeline run per scenario (cases share it). Returns the state as JSON."""
    from scripts.benchmark import VARIANTS

    from hiring_intel.graph import run
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
    if not outputs or "insights" not in outputs:  # target raised (e.g. LLM quota): count as a failed case
        return {"key": "golden_pass", "score": 0, "comment": "pipeline run failed"}
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
    from hiring_intel.llm import backend
    from hiring_intel.llm import preflight as llm_preflight
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