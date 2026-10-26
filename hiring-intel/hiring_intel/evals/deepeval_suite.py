"""DeepEval suite (with optional upload to Confident AI, DeepEval's hosted evaluation hub).

Each insight becomes an LLMTestCase:
    input              the agent's task
    actual_output      title + recommendation + evidence values
    retrieval_context  everything the agent was given: its feature table + retrieved RAG chunks
    context            the feature table (ground truth for hallucination checks)

Judge modes (env DEEPEVAL_JUDGE):
    offline  (default) deterministic metrics only: no model, no key, runs in CI
    openjev  + Faithfulness and Hallucination in DeepEval's "system_one" mode, judged by OpenJev ($0 API)
    llm      + Faithfulness, Hallucination and a G-Eval actionability rubric judged by an LLM
             (OPENAI_API_KEY, or a local model via `deepeval set-ollama --model <name>`)

Confident AI: run `deepeval login` once; `deepeval test run tests_eval/` and evaluate() then upload
every test run to the hub, where runs can be compared and regressions tracked.
"""
from __future__ import annotations

import os
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

try:
    from deepeval.test_case import SingleTurnParams as P
except ImportError:                                  # older deepeval
    from deepeval.test_case import LLMTestCaseParams as P

TASKS = {
    "sourcing": "Which sourcing channels give the best candidates per role, and which should be deprioritised?",
    "rejection": "Where and why are candidates rejected, and is the JD, sourcing or interviewing the problem?",
    "panel_load": "Which interviewers are overloaded or idle, and how should interview load be rebalanced?",
    "offers": "Why are offers declined: compensation, timing or competing offers?",
    "pipeline": "Where are stage SLA breaches and which requisitions have been open too long?",
}
OWNER = r"\b(lead|manager|coordinator|partner|EM|recruiter|owner|hiring manager)\b"
WHEN = r"\b(week|sprint|month|monday|friday|hours|quarter|by )\b"


# ---------------------------------------------------------------- deterministic metrics
class GroundedNumbersMetric(BaseMetric):
    """Share of numbers in the evidence that appear in the context. No model needed."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        ctx = " ".join(test_case.context or [])
        src = {round(float(x), 3) for x in re.findall(r"-?\d+\.?\d*", ctx)}
        vals = (test_case.metadata or {}).get("evidence_values", [])
        hits = sum(any(abs(v - s) <= 0.0015 or abs(v * 100 - s) <= 0.5 for s in src) for v in vals)
        self.score = hits / len(vals) if vals else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"{hits}/{len(vals)} evidence numbers found in the agent's feature table"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))

    @property
    def __name__(self):
        return "Grounded numbers"


class ActionableRuleMetric(BaseMetric):
    """Cheap actionability check: names an owner and a time frame or number."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        rec = (test_case.metadata or {}).get("recommendation", test_case.actual_output)
        owner = bool(re.search(OWNER, rec, re.I))
        when = bool(re.search(WHEN, rec, re.I))
        number = bool(re.search(r"\d", rec))
        self.score = (owner + when + number) / 3
        self.success = self.score >= self.threshold
        self.reason = f"owner={owner}, time frame={when}, number={number}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))

    @property
    def __name__(self):
        return "Actionable (rule)"


# ---------------------------------------------------------------- model-judged metrics
def judge_mode() -> str:
    return os.getenv("DEEPEVAL_JUDGE", "offline")


def build_metrics(mode: str | None = None) -> list:
    mode = mode or judge_mode()
    metrics: list = [GroundedNumbersMetric(), ActionableRuleMetric()]
    if mode == "openjev":
        from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
        from hiring_intel.evals.openjev_deepeval import OpenJevSystemOne
        from hiring_intel.openjev import KeywordFallback
        oj = OpenJevSystemOne()
        if isinstance(oj.model.backend, KeywordFallback):
            raise RuntimeError("DEEPEVAL_JUDGE=openjev needs the real OpenJev model (OPENJEV_URL or local "
                               "weights); the keyword fallback cannot judge faithfulness.")
        metrics += [FaithfulnessMetric(threshold=0.7, eval_mode="system_one", system_one_model=oj,
                                       include_reason=False, async_mode=False),
                    HallucinationMetric(threshold=0.5, eval_mode="system_one", system_one_model=oj,
                                        include_reason=False, async_mode=False)]
    elif mode == "llm":
        from deepeval.metrics import FaithfulnessMetric, GEval, HallucinationMetric
        model = os.getenv("DEEPEVAL_MODEL") or None              # e.g. gpt-4o; None = DeepEval default
        metrics += [
            FaithfulnessMetric(threshold=0.7, model=model),
            HallucinationMetric(threshold=0.5, model=model),
            GEval(name="Actionability", model=model, threshold=0.6,
                  evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
                  evaluation_steps=[
                      "Check the recommendation names who should act (a role such as recruiting lead).",
                      "Check it states a concrete action and a time frame or measurable target.",
                      "Check the action follows from the evidence given, not from generic best practice.",
                      "Penalise vague advice such as 'monitor' or 'consider improving'."]),
        ]
    return metrics


# ---------------------------------------------------------------- test cases from a run
def _chunk_texts(ids: list[str]) -> list[str]:
    if not ids:
        return []
    from hiring_intel.rag.retriever import _collection
    try:
        got = _collection().get(ids=ids)
        return list(got["documents"])
    except Exception:
        return []


def test_cases_from_state(state: dict) -> list[LLMTestCase]:
    from hiring_intel.agents.insight_agents import AGENTS
    from hiring_intel.ingest import load_all
    from hiring_intel.metrics import to_text
    cfg = state["cfg"]
    data = load_all(cfg["data_path"])
    cases = []
    for agent, out in state["outputs"].items():
        if out["status"] == "failed":
            continue
        feats = to_text(AGENTS[agent].features(data, compress=cfg.get("compress", False)))
        chunks = _chunk_texts(out.get("retrieved_chunks", []))
        for k, ins in enumerate(out["insights"]):
            ev_vals = [float(e["value"]) for e in ins["evidence"]
                       if e["source"] == "data" and isinstance(e["value"], (int, float))]
            ev_txt = "; ".join(f"{e['metric']}={e['value']}" for e in ins["evidence"] if e["source"] == "data")
            cases.append(LLMTestCase(
                input=TASKS[agent],
                actual_output=f"{ins['title']}. {ins['recommendation']} Evidence: {ev_txt}",
                retrieval_context=[feats] + chunks,
                context=[feats],
                metadata={"agent": agent, "insight": k, "run_id": state["run_id"], "severity": ins["severity"],
                          "recommendation": ins["recommendation"], "evidence_values": ev_vals},
                name=f"{state['run_id']}:{agent}:{k}"))
    return cases