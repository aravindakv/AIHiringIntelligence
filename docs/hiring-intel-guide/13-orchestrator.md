# 13 — The LangGraph orchestrator

**Goal:** wire everything into one graph with parallel agents, a retry loop, a human-review pause, and error handling at every hop. Plus a CLI to run it.

**Time:** 1 hour.

## Concepts

```text
START -> ingest -> enrich -> [sourcing | rejection | panel_load | offers | pipeline] -> evaluate
      evaluate --(any score < 0.6, not yet retried)--> retry (large tier) -> evaluate
      evaluate --(otherwise)--> human_review -> optimize -> persist -> END
```

- **Fan-out / fan-in.** Five edges from `enrich` run the agents in parallel; five edges into `evaluate` wait for all of them.
- **Reducers.** `outputs` and `evals` merge dicts so parallel nodes don't overwrite each other; `errors` concatenates lists.
- **Checkpointable state.** Only paths, config and JSON-able dicts live in state, so `InMemorySaver` (or SQLite) can pause the graph at `human_review` and resume it later.
- **Retry once.** `retried` stops a loop: an agent gets at most one retry on the large model.
- **Demo switches** prove the error handling: `crash_agent="offers"` (the agent raises, the run recovers on retry) and `inject_fault="sourcing"` (invented numbers, which the evaluator catches before the retry).

| Hop | Failure | Handling |
| --- | --- | --- |
| ingest | missing table / column | hard fail with a clear message |
| enrich | low-confidence label | `unclassified`, never forced |
| retrieval | empty / Chroma error | continue, status `degraded`, confidence capped |
| LLM | timeout / 429 / bad JSON | LiteLLM retries, one repair turn, then `failed` |
| agent | any exception | `failed` output, recorded in `errors` |
| evaluate | score < 0.6 | one retry on the large model |
| A2A | remote task fails | `failed` output with task id and state |

## Step 13.1 — The graph

**File:** `hiring_intel/graph.py`
```python
"""LangGraph orchestrator.

ingest -> enrich (OpenJev) -> 5 insight agents in parallel -> evaluate -> [retry low scorers on large]
       -> human_review (HITL) -> optimize -> persist

State carries only paths, configs and JSON-able dicts, so it can be checkpointed (needed for HITL
interrupt/resume). A failed agent writes a 'failed' output instead of raising.
"""
from __future__ import annotations

import datetime as dt
import operator
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from hiring_intel import hitl
from hiring_intel.agents.evaluator import evaluate_one
from hiring_intel.agents.insight_agents import AGENTS
from hiring_intel.agents.optimizer import optimize
from hiring_intel.contracts import AgentOutput, Evidence, ReviewItem
from hiring_intel.enrich import enrich
from hiring_intel.ingest import load_all
from hiring_intel.observability import execute, log_timing, purge_run
from hiring_intel.openjev import decider


def merge(old: dict | None, new: dict | None) -> dict:
    return {**(old or {}), **(new or {})}


class RunState(TypedDict, total=False):
    run_id: str
    cfg: dict
    enrich: dict
    outputs: Annotated[dict, merge]
    evals: Annotated[dict, merge]
    retried: list[str]
    review: list[dict]
    suggestions: list[dict]
    errors: Annotated[list, operator.add]


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


# ------------------------------------------------------------------ nodes
def ingest_node(s: RunState):
    t0 = time.perf_counter()
    load_all(s["cfg"]["data_path"])
    log_timing(s["run_id"], "ingest", "-", _ms(t0), 0)
    return {}


def enrich_node(s: RunState):
    t0 = time.perf_counter()
    info = enrich(load_all(s["cfg"]["data_path"]), s["run_id"], decider(), mode=s["cfg"]["classify"])
    log_timing(s["run_id"], "enrich", "-", _ms(t0), 0)
    return {"enrich": info}


def _run_agent(name: str, s: RunState, force_tier: str | None = None, last_eval: float | None = None) -> AgentOutput:
    cfg = s["cfg"]
    if cfg.get("crash_agent") == name and not force_tier:
        return AgentOutput(agent=name, run_id=s["run_id"], status="failed", error="RuntimeError: injected crash")
    if cfg.get("transport") == "a2a":
        from hiring_intel.a2a import call_agent
        out = call_agent(name, {"data_path": cfg["data_path"], "run_id": s["run_id"], "force_tier": force_tier
                                or cfg.get("force_tier"), "compress": cfg["compress"], "use_cache": cfg["use_cache"],
                                "last_eval": last_eval})
    else:
        out = AGENTS[name].run(load_all(cfg["data_path"]), s["run_id"], force_tier=force_tier or cfg.get("force_tier"),
                               compress=cfg["compress"], use_cache=cfg["use_cache"], last_eval=last_eval)
    if cfg.get("inject_fault") == name and not force_tier and out.insights:
        # fabricated numbers: the evaluator should catch them and trigger a retry on the large model
        for ins in out.insights:
            ins.evidence += [Evidence(metric="industry_hire_rate", value=v, segment={"note": "invented"})
                             for v in (0.777, 0.913, 0.4242)]
    return out


def make_agent_node(name: str):
    def node(s: RunState):
        out = _run_agent(name, s)
        llm = out.cost_of_insight.latency_ms if out.cost_of_insight else 0
        log_timing(s["run_id"], "agents", name, out.compute_ms, llm)
        errs = [f"{name}: {out.error}"] if out.status == "failed" else []
        return {"outputs": {name: out.model_dump()}, "errors": errs}
    return node


def evaluate_node(s: RunState):
    t0 = time.perf_counter()
    data = load_all(s["cfg"]["data_path"])
    todo = [a for a in s["outputs"] if a not in s.get("evals", {}) or a in s.get("retried", [])]
    res = {}
    for a in todo:
        out = AgentOutput.model_validate(s["outputs"][a])
        res[a] = evaluate_one(AGENTS[a], out, data, s["run_id"], s["cfg"]["compress"]).model_dump()
    log_timing(s["run_id"], "evaluate", "-", _ms(t0), 0)
    return {"evals": res}


def needs_retry(s: RunState) -> str:
    low = [a for a, r in s["evals"].items() if r["overall"] < 0.6 and a not in s.get("retried", [])]
    return "retry" if low else "human_review"


def retry_node(s: RunState):
    low = [a for a, r in s["evals"].items() if r["overall"] < 0.6 and a not in s.get("retried", [])]
    new = {}
    for a in low:                                      # sequential here; small number of agents
        out = _run_agent(a, s, force_tier="large", last_eval=s["evals"][a]["overall"])
        log_timing(s["run_id"], "retry", a, out.compute_ms, out.cost_of_insight.latency_ms if out.cost_of_insight else 0)
        new[a] = out.model_dump()
    return {"outputs": new, "retried": s.get("retried", []) + low}


def human_review_node(s: RunState):
    t0 = time.perf_counter()
    outs = {a: AgentOutput.model_validate(o) for a, o in s["outputs"].items() if o["status"] != "failed"}
    items = hitl.build_queue(s["run_id"], outs, s["evals"])
    mode = s["cfg"]["hitl_mode"]
    pend = [i for i in items if i.status == "pending"]
    if pend and mode == "interrupt":
        # pauses the graph; the caller resumes with Command(resume={item_id: {"action":..., "note":...}})
        decisions = interrupt({"pending": [i.model_dump() for i in pend]})
        for iid, d in (decisions or {}).items():
            hitl.apply_decision(iid, d.get("action", "approved"), d.get("reviewer", "human"), d.get("note", ""),
                                d.get("edited_recommendation"))
    elif pend and mode == "simulate":
        for i in pend:
            act, note = hitl.simulated_reviewer(i, s["evals"].get(i.agent, {}).get("overall", 0))
            hitl.apply_decision(i.item_id, act, "simulated-policy", note)
    log_timing(s["run_id"], "human_review", "-", _ms(t0), 0)
    final = [ReviewItem.model_validate_json(j).model_dump() for j in
             hitl.query("SELECT json FROM review_queue WHERE run_id=?", (s["run_id"],)).json]
    return {"review": final}


def optimize_node(s: RunState):
    t0 = time.perf_counter()
    sug = optimize(s["run_id"], s["evals"])
    log_timing(s["run_id"], "optimize", "-", _ms(t0), 0)
    return {"suggestions": [i.model_dump() for i in sug]}


def persist_node(s: RunState):
    import json
    rid = s["run_id"]
    for a, o in s["outputs"].items():
        execute("INSERT INTO agent_outputs VALUES (?,?,?,?)", (rid, a, o["status"], json.dumps(o)))
    for a, e in s["evals"].items():
        execute("INSERT INTO eval_results VALUES (?,?,?,?)", (rid, a, e["overall"], json.dumps(e)))
    from hiring_intel.report import summarize_run
    summarize_run(rid, s)
    return {}


# ------------------------------------------------------------------ graph
def build(checkpointer=None):
    g = StateGraph(RunState)
    g.add_node("ingest", ingest_node)
    g.add_node("enrich", enrich_node)
    for name in AGENTS:
        g.add_node(name, make_agent_node(name))
        g.add_edge("enrich", name)                     # fan-out: five agents in parallel
        g.add_edge(name, "evaluate")                   # fan-in
    g.add_node("evaluate", evaluate_node)
    g.add_node("retry", retry_node)
    g.add_node("human_review", human_review_node)
    g.add_node("optimize", optimize_node)
    g.add_node("persist", persist_node)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "enrich")
    g.add_conditional_edges("evaluate", needs_retry, {"retry": "retry", "human_review": "human_review"})
    g.add_edge("retry", "evaluate")
    g.add_edge("human_review", "optimize")
    g.add_edge("optimize", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


_APP = None


def app():
    global _APP
    if _APP is None:
        _APP = build()
    return _APP


def run(data_path: str = "data/raw/baseline_small", *, scenario: str = "", size: str = "", variant: str = "routed",
        force_tier: str | None = None, compress: bool = False, use_cache: bool = False, classify: str = "openjev",
        hitl_mode: str = "queue", transport: str = "local", inject_fault: str | None = None,
        crash_agent: str | None = None, run_id: str | None = None, graph=None) -> dict:
    if run_id:
        purge_run(run_id)          # a fixed id (benchmark, golden set) re-runs from a clean slate
    run_id = run_id or f"{variant}-{dt.datetime.now().strftime('%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
    cfg = dict(data_path=data_path, scenario=scenario, size=size, variant=variant, force_tier=force_tier,
               compress=compress, use_cache=use_cache, classify=classify, hitl_mode=hitl_mode, transport=transport,
               inject_fault=inject_fault, crash_agent=crash_agent, started=time.perf_counter())
    g = graph or app()
    state = g.invoke({"run_id": run_id, "cfg": cfg, "outputs": {}, "evals": {}, "errors": [], "retried": []},
                     {"configurable": {"thread_id": run_id}})
    state["run_id"] = run_id
    return state


def resume(run_id: str, decisions: dict, graph=None) -> dict:
    """Resume a run paused at human_review (hitl_mode='interrupt')."""
    g = graph or app()
    return g.invoke(Command(resume=decisions), {"configurable": {"thread_id": run_id}})
```

## Step 13.2 — CLI

`--hitl interrupt` turns your terminal into the review queue.

**File:** `scripts/run_pipeline.py`
```python
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
```

## Verify

<!-- verify:13 -->
```bash
python scripts/run_pipeline.py --data data/raw/baseline_medium | tail -9
python - <<'EOF'
from hiring_intel.graph import run
s = run("data/raw/baseline_small", variant="ch13-faults", crash_agent="offers", inject_fault="sourcing")
print("retried:", s["retried"], "| errors:", s["errors"])
print("scores after retry:", {a: e["overall"] for a, e in s["evals"].items()})
print("optimizer:", [x["title"] for x in s["suggestions"]])
from hiring_intel.report import e2e_latency_ms
print("E2E by stage (ms):", e2e_latency_ms(s["run_id"])["by_stage"])
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
run optimized-1021123110-8e53 done; errors: []
  offers      [high  ] Offers below 90% of expected CTC are being declined
  panel_load  [high  ] 10 interviewers over load policy
  pipeline    [medium] SRE onsite breaches SLA
  pipeline    [high  ] 10 SRE requisitions open too long
  rejection   [high  ] [jd] Mobile screen: 'JD mismatch' drives rejections
  rejection   [medium] [sourcing] agency candidates fail tech_1 (Data)
  sourcing    [high  ] Backend: referral out-converts job_board
  sourcing    [medium] Data: referral out-converts agency
retried: ['offers', 'sourcing'] | errors: ['offers: RuntimeError: injected crash']
scores after retry: {'offers': 0.97, 'panel_load': 0.97, 'pipeline': 0.97, 'rejection': 0.97, 'sourcing': 0.97}
optimizer: ['Trial the small model for rejection', 'Compress rejection context', 'Trial the small model for sourcing']
E2E by stage (ms): {'ingest': 51, 'enrich': 53, 'evaluate': 183, 'human_review': 21, 'optimize': 4133, 'agents': 15375, 'retry': 12646}
```

Interactive review in the terminal:

```bash
python scripts/run_pipeline.py --data data/raw/comp_crunch_small --hitl interrupt
```

## Checkpoint

- [ ] a full run prints insights from all five agents
- [ ] the crashed agent and the faked agent are both retried and recover
- [ ] the optimizer returns suggestions and E2E latency is broken down by stage
