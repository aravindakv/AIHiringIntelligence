# 17 — External validation with DeepEval and Confident AI

**Goal:** a pytest-style regression suite over real pipeline output, with metrics written by a third party, uploaded to Confident AI (DeepEval's hosted evaluation hub). OpenJev can serve as a $0 judge.

**Time:** 1 hour.

## Concepts

- **One test case per insight.**
  - `input`: the agent's question;
  - `actual_output`: title + recommendation + evidence;
  - `retrieval_context`: everything the agent saw (features + RAG chunks);
  - `context`: the feature table, the ground truth.
- **Metrics:**

| Metric | Judge | Catches |
| --- | --- | --- |
| Grounded numbers (custom) | none | evidence numbers not in the feature table |
| Actionable (rule, custom) | none | no owner, time frame or number |
| Faithfulness | OpenJev or LLM | claims not supported by what the agent was given |
| Hallucination | OpenJev or LLM | statements contradicting the feature table |
| G-Eval "Actionability" | LLM | rubric: owner, action, time frame, follows from evidence |

- **Judge modes** (`DEEPEVAL_JUDGE`): `offline` (custom metrics only; CI), `openjev`, `llm`.
- **OpenJev inside DeepEval.** DeepEval 4.x metrics take `eval_mode="system_one"` with a `system_one_model`: a model that answers yes/no, choice and score questions with probabilities. `OpenJevSystemOne` implements that interface, so faithfulness and hallucination run locally with no LLM. This mode refuses to run on the keyword fallback, because a fallback score would be meaningless.
- **Confident AI.** After `deepeval login`, every `deepeval test run` uploads a test run you can compare with the last one.

## Step 17.1 — OpenJev as a DeepEval judge

**File:** `hiring_intel/evals/openjev_deepeval.py`
```python
"""OpenJev as a DeepEval "System One" model.

DeepEval 4.x metrics such as Faithfulness and Hallucination accept `eval_mode="system_one"` (or
"hybrid") plus a `system_one_model`: a model that answers typed questions (yes/no "noul", choice,
score) with probabilities instead of generating text. That is exactly what OpenJev does, so this
adapter lets DeepEval judge with OpenJev locally at $0 API cost.

    from deepeval.metrics import FaithfulnessMetric
    m = FaithfulnessMetric(eval_mode="system_one", system_one_model=OpenJevSystemOne(), include_reason=False)
"""
from __future__ import annotations

import json
from typing import Any

from deepeval.models.base_model import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion,
                                               ScoreAnswer, ScoreQuestion, SystemOneAnswers)

from hiring_intel.openjev import OpenJevDecider


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, default=str)


class OpenJevSystemOne(DeepEvalBaseSystemOneModel):
    def __init__(self, decider: OpenJevDecider | None = None):
        self._decider = decider
        super().__init__(model="openjev")

    def load_model(self, *args, **kwargs):
        return self._decider or OpenJevDecider()

    def get_model_name(self, *args, **kwargs) -> str:
        return f"openjev ({self.model.backend.name})"

    def decide(self, state: Any, questions: dict) -> tuple[SystemOneAnswers, float | None]:
        premise = _text(state)
        ans = SystemOneAnswers()
        for key, q in questions.items():
            instr = _text(q.instructions)
            if isinstance(q, ChoiceQuestion):
                opts = list(q.options)
                d = self.model.choose(f"deepeval:{key}", premise, instr, opts)
                ans.choices[key] = ChoiceAnswer(choice=d.choice, probabilities=d.probs, confidence=d.confidence)
            elif isinstance(q, ScoreQuestion):
                levels = [str(l) for l in q.levels]
                d = self.model.choose(f"deepeval:{key}", premise, instr, levels)
                probs = {i + 1: d.probs[l] for i, l in enumerate(levels)}
                ans.scores[key] = ScoreAnswer(score=levels.index(d.choice) + 1, probabilities=probs,
                                              confidence=d.confidence)
            else:                                    # NoulQuestion: probability of "yes"
                d = self.model.yes_no(f"deepeval:{key}", premise, instr)
                ans.nouls[key] = NoulAnswer(probability=d.probs["yes"])
        return ans, 0.0                              # $0 API cost

    async def a_decide(self, state: Any, questions: dict):
        return self.decide(state, questions)
```

## Step 17.2 — The suite

**File:** `hiring_intel/evals/deepeval_suite.py`
```python
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
        model = os.getenv("DEEPEVAL_MODEL")                      # e.g. gpt-4o; None = DeepEval default
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
```

## Step 17.3 — The tests

Note the import alias `cases_from_state`: pytest would collect a function named `test_cases_from_state` as a test.

**File:** `tests_eval/test_deepeval.py`
```python
"""DeepEval regression suite over real pipeline output.

    pytest tests_eval/ -q                         # offline metrics (CI)
    DEEPEVAL_JUDGE=openjev pytest tests_eval/     # + faithfulness/hallucination judged by OpenJev
    DEEPEVAL_JUDGE=llm deepeval test run tests_eval/test_deepeval.py   # + G-Eval; uploads to Confident AI if logged in
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import assert_test  # noqa: E402

from hiring_intel.evals.deepeval_suite import build_metrics
from hiring_intel.evals.deepeval_suite import test_cases_from_state as cases_from_state  # noqa: E402

INPUTS = os.getenv("DEEPEVAL_INPUTS", "baseline_medium,comp_crunch_medium,healthy_medium").split(",")


def _cases():
    from hiring_intel.graph import run
    for name in INPUTS:
        path = ROOT / "data/raw" / name
        if not path.exists():
            from scripts.generate_mock_data import generate
            scenario, size = name.rsplit("_", 1)
            generate(scenario, size)
        scenario, size = name.rsplit("_", 1)
        state = run(str(path.relative_to(ROOT)), scenario=scenario, size=size, variant="deepeval",
                    compress=True, classify="openjev")
        yield from cases_from_state(state)


CASES = list(_cases())


@pytest.mark.parametrize("test_case", CASES, ids=[c.name for c in CASES])
def test_insight(test_case):
    assert_test(test_case, build_metrics())
```

## Step 17.4 — Run it three ways

```bash
pytest tests_eval/ -q                                                    # offline
DEEPEVAL_JUDGE=openjev OPENJEV_URL=http://localhost:8200 pytest tests_eval/ -q
deepeval login --api-key <CONFIDENT_API_KEY>                             # once
DEEPEVAL_JUDGE=llm OPENAI_API_KEY=sk-... deepeval test run tests_eval/test_deepeval.py
```

Prefer a local LLM judge? `deepeval set-ollama --model llama3.1`.
Change the inputs with `DEEPEVAL_INPUTS=baseline_large,comp_crunch_large`.

In Confident AI open **Test Runs**: each insight shows every metric's score and reason; compare runs to see what a prompt change broke.

## What the suite found

The first offline run failed 2 of 20 cases. On the healthy input, the "no pipeline action" and "offer acceptance healthy" findings named no owner and no follow-up date. The fix went into the agents (chapter 09: every no-action finding starts with "Recruiting ops lead:" and gives a re-check date), not into the metric.

## Verify

<!-- verify:17 -->
```bash
pytest tests_eval/ -q -p no:cacheprovider 2>&1 | tail -1
DEEPEVAL_JUDGE=openjev OPENJEV_BACKEND=fallback pytest tests_eval/ -q -p no:cacheprovider -x 2>&1 | grep -o "RuntimeError: DEEPEVAL_JUDGE=openjev needs the real OpenJev model" | head -1
```

Expected output (captured on a fresh extract, mock backend):

```text
20 passed in 5.80s
RuntimeError: DEEPEVAL_JUDGE=openjev needs the real OpenJev model
```

The second command shows the guard: without the real OpenJev model, the OpenJev judge mode refuses to run.

## Checkpoint

- [ ] offline suite: 20 passed
- [ ] with a key: the test run appears in Confident AI with per-metric reasons
- [ ] you can explain why "no action" still needs an owner
