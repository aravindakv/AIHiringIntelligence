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
  - LLM note labels land on the right rows even when the model skips or reorders;
  - a real model can't raise alarms on a healthy pipeline;
  - the RAG index rebuilds when the embedding model changes, keeping approved insights;
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


class _Fixed:
    """A fake model backend that returns fixed entailment scores."""
    name = "fixed"

    def __init__(self, scores):
        self.scores, self.seen = scores, None

    def entail(self, premise, hypotheses):
        self.seen = (premise, hypotheses)
        return self.scores


def test_hypotheses_floor_and_abstain():
    opts = ["a", "b", "c", "unclear"]
    b = _Fixed([0.4, 0.3, 0.2, 0.1])                      # top normalised prob 0.4: ok for 4 options
    d = OpenJevDecider(b).choose("t", "note", "q?", opts, hypotheses={o: f"It is {o}." for o in opts},
                                 premise_prefix="Note: ", abstain="unclear")
    assert d.choice == "a" and not d.escalated
    assert b.seen == ("Note: note", ["It is a.", "It is b.", "It is c.", "It is unclear."])
    d2 = OpenJevDecider(_Fixed([0.1, 0.1, 0.1, 0.7])).choose("t", "note", "q?", opts, abstain="unclear")
    assert d2.choice == "unclear" and d2.escalated          # abstaining always escalates
    d3 = OpenJevDecider(_Fixed([0.48, 0.52])).yes_no("t", "note", "q?")
    assert d3.escalated                                     # yes / no keeps the 0.55 floor
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

The enrichment test fakes a careless real model: it skips one note, reorders the rest and changes the case of a label. Labels must still land on the right rows. The first version matched labels by position and stored them by note text, so one skipped label relabelled every copy of the following notes; a real model scored 0.359 that way.

**File:** `tests/test_enrich.py`
```python
import json
from types import SimpleNamespace

import pandas as pd

from hiring_intel import enrich as E
from hiring_intel.openjev import KeywordFallback, OpenJevDecider


def test_llm_labels_match_by_number_not_position(monkeypatch):
    """A real model may skip a note, reorder, or change case; labels must still land on the right rows."""
    notes = pd.Series(["Code was messy and incorrect on edge cases.", "Weak on SQL joins.",
                       "Code was messy and incorrect on edge cases."], index=[10, 11, 12])

    def fake_completion(**kw):
        content = json.dumps({"labels": [{"i": 3, "label": "Coding"}, {"i": 2, "label": "fundamentals"}]})
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv("LLM_BACKEND", "litellm")
    got = E._classify(OpenJevDecider(KeywordFallback()), "t-enrich", "classify_rejection_note", notes,
                      E.REJECTION_LABELS, mode="llm")
    # note 1 was skipped: unclassified, not shifted; same text in row 12 keeps its own label
    assert got.to_dict() == {10: "unclassified", 11: "fundamentals", 12: "coding"}
```

Two more came from running a real model: the healthy-scenario false alarms, and every agent `degraded` after MiniLM was installed over an offline-built index.

**File:** `tests/test_signals.py`
```python
import json
from types import SimpleNamespace

from hiring_intel.agents.insight_agents import AGENTS
from hiring_intel.contracts import LLMInsights
from hiring_intel.ingest import load_all
from hiring_intel.rag.retriever import retrieve


def test_real_model_cannot_raise_alarms_on_a_healthy_pipeline(monkeypatch):
    """A real model given raw tables flagged noise as 'high' on the healthy scenario. The rule detector's
    SIGNALS go into the prompt, and guard() caps severity at the worst detected signal."""
    agent, data = AGENTS["panel_load"], load_all("data/raw/healthy_small")
    feats = agent.features(data)
    sig = agent.analyze(feats, retrieve(agent.rag_query, agent.name, None, 4))
    assert {i.severity for i in sig.insights} == {"low"}            # the detector says: no action

    alarm = sig.insights[0].model_copy(update={"title": "Interview Load Imbalance Detected", "severity": "high"})
    seen = {}

    def fake_completion(**kw):
        seen["user"] = kw["messages"][1]["content"]
        content = LLMInsights(insights=[alarm, alarm]).model_dump_json()
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=500, completion_tokens=200),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv("LLM_BACKEND", "litellm")
    out = agent.run(data, "t-signals", force_tier="small")
    assert "SIGNALS (passed thresholds in code)" in seen["user"]
    assert out.status != "failed", out.error
    assert [i.severity for i in out.insights] == ["low"]           # capped, and trimmed to one finding
```

**File:** `tests/test_rag_rebuild.py`
```python
import numpy as np
from chromadb import EmbeddingFunction

from hiring_intel.rag import retriever as R


class Fake384(EmbeddingFunction):
    """Stands in for MiniLM (384 dims) without a download."""
    def __init__(self):
        pass

    def __call__(self, input):
        return [np.ones(384, dtype=np.float32) / 20 for _ in input]

    @staticmethod
    def name():
        return "fake-384"


def test_index_rebuilds_when_embeddings_change(tmp_path, monkeypatch):
    """Index built offline with hashing (512 dims), then MiniLM installed (384 dims): every query used to
    fail with a dimension error and every agent showed 'degraded'. Approved insights must survive."""
    monkeypatch.setattr(R, "CHROMA_DIR", str(tmp_path))
    monkeypatch.setattr(R, "embedding_fn", lambda: (R.HashingEmbedding(), "hashing"))
    R._collection.cache_clear(); R.retrieve.cache_clear()
    R.build_index()
    R.add_past_insight("approved_1", "Referral hires converted well last quarter", "sourcing")

    monkeypatch.setattr(R, "embedding_fn", lambda: (Fake384(), "minilm"))
    monkeypatch.setitem(R.DEFAULT_MAX_DISTANCE, "minilm", 2.0)
    R._collection.cache_clear(); R.retrieve.cache_clear()
    ids = [c for c, _ in R.retrieve("referral hire rate", "sourcing", None, 10)]
    assert ids and "approved_1" in ids
    R._collection.cache_clear(); R.retrieve.cache_clear()
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
16 passed in 6.57s
jobs: ['offline', 'real-models']
```

## Checkpoint

- [ ] 16 unit tests pass
- [ ] after a push, the `offline` job is green; add the three secrets for the nightly job
