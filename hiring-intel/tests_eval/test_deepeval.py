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