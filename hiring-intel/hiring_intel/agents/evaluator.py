"""Evaluation Agent: the quality gate.

Three layers, cheapest first:
  1. schema validity (Pydantic already enforced it)          -> free
  2. deterministic grounding: every numeric evidence value must appear in the features,
     every rag ref must be a chunk that was actually retrieved -> free
  3. LLM-as-judge rubric (actionability, grounding, hallucination risk). In mock mode the judge is
     a heuristic + an OpenJev "is this actionable?" check, clearly marked as such.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from hiring_intel.contracts import AgentOutput
from hiring_intel.llm import CFG, LLMUnavailable, backend, call_structured
from hiring_intel.metrics import to_text
from hiring_intel.observability import log_decision
from hiring_intel.openjev import decider


class JudgeScore(BaseModel):
    actionability: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    hallucination_risk: int = Field(ge=1, le=5)          # 5 = high risk
    rationale: str
    judge: str = "llm"


class EvalResult(BaseModel):
    agent: str
    schema_ok: bool
    numbers_grounded: float
    refs_valid: bool
    judge: JudgeScore | None = None
    overall: float
    flags: list[str] = []


RUBRIC = """You are a strict reviewer of hiring analytics. Score the insights against SOURCE DATA.
actionability 5 = names who, what and a measurable target or date; 1 = generic advice.
grounding 5 = every number and claim appears in SOURCE DATA; 1 = mostly unsupported.
hallucination_risk 5 = contains figures, benchmarks or causes not in SOURCE DATA; 1 = none.
Be harsh: a plausible but unsupported benchmark counts as hallucination."""

OWNER_WORDS = ["lead", "manager", "coordinator", "partner", "em ", "recruiter", "owner"]
TIME_WORDS = ["week", "sprint", "month", "monday", "friday", "48 hours", "quarter", "by "]


def _numbers(text: str) -> set[float]:
    return {round(float(x), 3) for x in re.findall(r"-?\d+\.?\d*", text)}


def grounding_check(out: AgentOutput, features_txt: str) -> tuple[float, bool]:
    src = _numbers(features_txt)
    vals = hits = 0
    for ins in out.insights:
        for e in ins.evidence:
            if e.source == "data" and isinstance(e.value, (int, float)):
                vals += 1
                v = round(float(e.value), 3)
                hits += any(abs(v - s) <= 0.0015 or abs(v * 100 - s) <= 0.5 for s in src)
    refs_ok = all(e.ref in out.retrieved_chunks for i in out.insights for e in i.evidence if e.source == "rag")
    return (hits / vals if vals else 0.0), refs_ok


def _mock_judge(out: AgentOutput, grounded: float, run_id: str) -> JudgeScore:
    acts = []
    for ins in out.insights:
        d = decider().yes_no("eval_actionable", ins.recommendation,
                             "Is this recommendation specific and actionable, naming an owner, an action and a "
                             "measurable target or date?", yes_keywords=OWNER_WORDS + TIME_WORDS)
        log_decision(run_id, d)
        acts.append(d.probs["yes"])
    act = sum(acts) / len(acts) if acts else 0
    return JudgeScore(actionability=max(1, min(5, round(1 + 4 * act))), grounding=max(1, round(1 + 4 * grounded)),
                      hallucination_risk=max(1, min(5, round(1 + 4 * (1 - grounded)))),
                      rationale=f"heuristic judge (mock mode): actionability p={act:.2f}, grounded={grounded:.2f}",
                      judge="heuristic+openjev")


def judge(out: AgentOutput, features_txt: str, grounded: float, run_id: str) -> JudgeScore:
    if backend() == "mock":
        return _mock_judge(out, grounded, run_id)
    s, _ = call_structured(agent="evaluator", run_id=run_id, model=CFG["judge"]["model"], system=RUBRIC,
                           user=f"SOURCE DATA:\n{features_txt}\n\nINSIGHTS:\n"
                                f"{out.model_dump_json(include={'insights'})}",
                           schema=JudgeScore, temperature=0)
    return s


def combine(grounded: float, refs_ok: bool, j: JudgeScore | None) -> tuple[float, list[str]]:
    flags = []
    if grounded < 0.8:
        flags.append("ungrounded_numbers")
    if not refs_ok:
        flags.append("invalid_rag_ref")
    jscore = 0.5 if j is None else (j.actionability + j.grounding + (6 - j.hallucination_risk)) / 15
    if j and j.hallucination_risk >= 4:
        flags.append("hallucination_risk")
    overall = 0.35 * grounded + 0.15 * refs_ok + 0.5 * jscore
    if "ungrounded_numbers" in flags:          # a number we cannot find in the data is a hard fail
        overall = min(overall, 0.55)
    return round(overall, 2), flags


def evaluate_one(agent, out: AgentOutput, data, run_id: str, compress: bool = False) -> EvalResult:
    if out.status == "failed":
        return EvalResult(agent=out.agent, schema_ok=False, numbers_grounded=0, refs_valid=False, overall=0.0,
                          flags=["agent_failed"])
    ftxt = to_text(agent.features(data, compress=compress))
    grounded, refs_ok = grounding_check(out, ftxt)
    try:
        j = judge(out, ftxt, grounded, run_id)
    except LLMUnavailable:
        raise
    except Exception:
        j = None
    overall, flags = combine(grounded, refs_ok, j)
    return EvalResult(agent=out.agent, schema_ok=True, numbers_grounded=round(grounded, 2), refs_valid=refs_ok,
                      judge=j, overall=overall, flags=flags)