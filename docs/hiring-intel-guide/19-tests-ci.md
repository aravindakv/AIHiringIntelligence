# 19 — Tests and CI

**Goal:** fast unit tests for the contract, the planted patterns, OpenJev, the graph's error handling, human review and A2A; and a GitHub Actions workflow that gates every push on offline evaluations and runs real-model evaluations nightly.

**Time:** 30 minutes.

## Concepts

- **Unit tests** check behaviour, not wording:
  - the contract rejects bad insights;
  - the generator's patterns are recoverable;
  - OpenJev escalates when unsure;
  - a crashed agent and a faked agent both recover;
  - interrupt/resume works;
  - A2A returns the contract;
  - no name from the dataset appears in any prompt.
- **CI tiers:**
  - every push: unit tests + golden set offline (must stay 18/18 on medium) + DeepEval offline;
  - nightly or on demand: LangSmith experiment and DeepEval with an LLM judge, uploading to Confident AI. Keys come from repository secrets: `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `CONFIDENT_API_KEY`.

## Step 19.1 — Unit tests

**File:** `tests/conftest.py`
```python
import os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("EMBEDDINGS", "hashing")
os.environ.setdefault("OPENJEV_BACKEND", "fallback")
os.environ["HI_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
if not (ROOT / "data/raw/baseline_small").exists():
    from scripts.generate_mock_data import generate
    for sc in ["baseline", "healthy"]:
        generate(sc, "small")
```

**File:** `tests/test_contracts.py`
```python
import pytest
from pydantic import ValidationError
from hiring_intel.contracts import Alternative, Evidence, Insight

ALT = Alternative(option="rules", trade_off="no narrative")


def ok(**kw):
    base = dict(title="t", recommendation="Recruiter lead: move 20% of agency budget to referrals this quarter.",
                evidence=[Evidence(metric="m", value=0.1)], confidence_score=0.7, alternative=ALT)
    base.update(kw)
    return Insight(**base)


def test_valid_insight():
    assert ok().confidence_score == 0.7


@pytest.mark.parametrize("kw", [dict(confidence_score=1.4), dict(evidence=[]),
                                dict(recommendation="Consider improving the sourcing process overall please.")])
def test_rejects_bad_insights(kw):
    with pytest.raises(ValidationError):
        ok(**kw)
```

**File:** `tests/test_metrics.py`
```python
from hiring_intel.ingest import load_all
from hiring_intel.metrics import rejection_features, sourcing_features


def test_planted_referral_pattern():
    g = sourcing_features(load_all("data/raw/baseline_small"))
    b = g[g.role == "Backend"].set_index("source")
    assert b.loc["referral", "onsite_rate"] > 2 * b.loc["job_board", "onsite_rate"]


def test_planted_mobile_screen_pattern():
    f = rejection_features(load_all("data/raw/baseline_small"))["by_reason"]
    top = f[(f.role == "Mobile") & (f.stage == "screen")].iloc[0]
    assert top.reason == "JD mismatch" and top.share_of_stage > 0.5
```

**File:** `tests/test_openjev.py`
```python
from hiring_intel.openjev import KeywordFallback, OpenJevDecider


def test_fallback_choose_and_escalate():
    d = OpenJevDecider(KeywordFallback())
    r = d.choose("t", "Weak on SQL joins and indexing.", "reason?", ["fundamentals", "coding"],
                 keywords={"fundamentals": ["sql", "joins"], "coding": ["bugs"]})
    assert r.choice == "fundamentals" and not r.escalated and r.backend == "keyword-fallback"
    amb = d.choose("t", "Not the right fit.", "reason?", ["fundamentals", "coding"],
                   keywords={"fundamentals": ["sql"], "coding": ["bugs"]})
    assert amb.escalated
```

**File:** `tests/test_graph.py`
```python
from hiring_intel.graph import resume, run


def test_run_survives_crash_and_retries():
    s = run("data/raw/baseline_small", variant="t", crash_agent="offers", inject_fault="sourcing")
    assert "offers" in s["retried"] and "sourcing" in s["retried"]
    assert all(o["status"] != "failed" for o in s["outputs"].values())   # recovered on retry


def test_hitl_interrupt_and_resume():
    s = run("data/raw/baseline_small", variant="t", hitl_mode="interrupt")
    pending = s["__interrupt__"][0].value["pending"]
    assert pending
    s2 = resume(s["run_id"], {p["item_id"]: {"action": "rejected", "reviewer": "test"} for p in pending})
    assert {r["status"] for r in s2["review"]} <= {"rejected", "auto_published"}


def test_healthy_input_raises_nothing_high_at_scale():
    s = run("data/raw/healthy_small", variant="t")
    assert s["evals"] and all(e["overall"] > 0 for e in s["evals"].values())
```

**File:** `tests/test_a2a.py`
```python
from fastapi.testclient import TestClient
from hiring_intel import a2a


def test_agent_card_and_message_send():
    c = TestClient(a2a.server)
    card = c.get("/agents/offers/.well-known/agent-card.json").json()
    assert card["skills"][0]["id"] == "offers.analyze"
    out = a2a.call_agent("offers", {"data_path": "data/raw/baseline_small", "run_id": "a2a-test"}, http=c)
    assert out.agent == "offers" and out.insights
```

The privacy test captures every prompt built during a full run (uncompressed features and LLM note labelling, the variant that sends the most text) and fails if any candidate, interviewer or hiring-manager name appears. Disable `strip_pii` and it fails; that is how you know it tests something.

**File:** `tests/test_privacy.py`
```python
"""Names are in the CSVs for demos; they must never appear in any prompt sent to a model."""
import pandas as pd

from hiring_intel import llm


def test_no_names_in_any_prompt(monkeypatch):
    prompts = []
    real = llm.call_structured

    def spy(**kw):
        prompts.append(kw["system"] + "\n" + kw["user"])
        return real(**kw)

    import hiring_intel.agents.base as base
    import hiring_intel.agents.optimizer as opt
    import hiring_intel.agents.router as router
    import hiring_intel.enrich as enrich
    for mod in (base, opt, router, enrich):
        monkeypatch.setattr(mod, "call_structured", spy)

    from hiring_intel.graph import run
    # all_large-style run: uncompressed features and LLM note labelling send the most text
    run("data/raw/baseline_small", variant="privacy", compress=False, classify="llm", force_tier="large")
    assert prompts, "no prompts captured"

    raw = "data/raw/baseline_small/"
    names = set(pd.read_csv(raw + "candidates.csv").candidate_name)
    names |= set(pd.read_csv(raw + "interviewers.csv").name)
    names |= set(pd.read_csv(raw + "requisitions.csv").hiring_manager)
    text = "\n".join(prompts)
    leaked = sorted(n for n in names if n in text)
    assert not leaked, f"names leaked into prompts: {leaked[:5]}"
```

## Step 19.2 — GitHub Actions

**File:** `.github/workflows/eval.yml`
```yaml
# Every push: unit tests + offline golden set + offline DeepEval (no keys, no cost).
# Nightly / manual: real-model experiment in LangSmith and DeepEval with an LLM judge (uses secrets).
name: eval
on:
  push:
  pull_request:
  schedule: [{cron: "30 20 * * *"}]      # 02:00 IST
  workflow_dispatch:

jobs:
  offline:
    runs-on: ubuntu-latest
    env: {LLM_BACKEND: mock, EMBEDDINGS: hashing, OPENJEV_BACKEND: fallback, DEEPEVAL_TELEMETRY_OPT_OUT: "YES"}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install -r requirements.txt
      - run: |
          python scripts/generate_mock_data.py --scenario baseline --size medium
          for s in healthy comp_crunch interviewer_burnout sourcing_shift; do
            python scripts/generate_mock_data.py --scenario $s --size medium; done
          python scripts/write_knowledge_base.py && python scripts/make_golden_set.py
      - run: pytest -q tests
      - name: Golden set must stay at 18/18 on medium inputs
        run: |
          python scripts/langsmith_eval.py --offline --size medium | tee golden.txt
          grep -q "golden pass rate: 18/18" golden.txt
      - run: pytest -q tests_eval

  real-models:
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    env:
      LLM_BACKEND: litellm
      EMBEDDINGS: hashing
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      LANGSMITH_TRACING: "true"
      LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
      LANGSMITH_PROJECT: hiring-intel-ci
      CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
      DEEPEVAL_JUDGE: llm
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: pip install -r requirements.txt
      - run: |
          for s in baseline healthy comp_crunch interviewer_burnout sourcing_shift; do
            python scripts/generate_mock_data.py --scenario $s --size medium; done
          python scripts/write_knowledge_base.py && python scripts/make_golden_set.py
      - run: python scripts/langsmith_eval.py --variant optimized --size medium
      - run: deepeval login --api-key "$CONFIDENT_API_KEY"
      - run: deepeval test run tests_eval/test_deepeval.py
```

## Verify

<!-- verify:19 -->
```bash
pytest -q tests 2>&1 | tail -1
python -c "import yaml; w=yaml.safe_load(open('.github/workflows/eval.yml')); print('jobs:', list(w['jobs']))"
```

Expected output (captured on a fresh extract, mock backend):

```text
12 passed in 5.15s
jobs: ['offline', 'real-models']
```

## Checkpoint

- [ ] 12 unit tests pass
- [ ] after a push, the `offline` job is green; add the three secrets for the nightly job
