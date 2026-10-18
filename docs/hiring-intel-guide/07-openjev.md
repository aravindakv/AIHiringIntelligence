# 07 — OpenJev for simple decisions

**Goal:** a decision layer that answers closed-set questions (which label, yes or no) without an LLM, and uses it to label thousands of free-text notes.

**Time:** 45 minutes (plus model download if you run it locally).

## Concepts

- **What OpenJev is.** An open NLI cross-encoder ([huggingface.co/AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), MIT per its model card), built on Qwen3.5. Given a premise and a hypothesis it returns P(contradiction), P(entailment) and P(neutral). To choose between options, score one hypothesis per option ("…Answer: fundamentals") and take the highest entailment. It doesn't generate text, it is deterministic, and the option order doesn't matter.
- **Where it pays off.** Decisions with a fixed answer set, made many times:

| Decision | Options | Used in |
| --- | --- | --- |
| Label a rejection note | 7 reasons | this chapter |
| Label a decline note | comp / competing / timing / location | this chapter |
| "Does this task need causal reasoning?" | yes / no | Routing Agent, ch. 08 |
| "Does this change pay, headcount or workload?" | yes / no | human review, ch. 11 |
| "Is this recommendation actionable?" | yes / no | evaluator, ch. 10 |
| DeepEval faithfulness / hallucination | yes / no per claim | ch. 17 |

- **Escalate, don't guess.** Below `min_confidence` (0.55) a decision is marked `escalated`: a note becomes `unclassified`, routing falls back to a small LLM, and human review flags the item.
- **Three backends.** `http` (the Docker service), `local` (weights in-process), `fallback` (a keyword scorer with the same interface, for offline CI). Every decision records which one answered.
- **Caveats** from the model card: probabilities are not calibrated to your labels, it can be swayed by injected text, and its v5 checkpoints saw public benchmark test sets. Measure it on your own labelled notes.

## Step 7.1 — The decider

**File:** `hiring_intel/openjev.py`
```python
"""OpenJev for simple, closed-set decisions.

OpenJev (huggingface.co/AlexWortega/openjev, MIT) is Qwen3.5 fine-tuned as an NLI cross-encoder:
given (premise, hypothesis) it returns P(contradiction), P(entailment), P(neutral). A decision over
options = one hypothesis per option ("<question> Answer: <option>"), pick the highest entailment.
No text generation, deterministic, order-independent, and $0 API cost.

Used here for:
  1. classify free-text rejection / decline notes into reason categories (thousands of items)
  2. Routing Agent: "does this task need multi-step causal reasoning?" (borderline cases)
  3. HITL gate: "does this recommendation touch comp, headcount or a person's workload?"
  4. Evaluator pre-check: "is this recommendation specific and actionable?"

Backends (config/models.yaml -> openjev.backend, or env OPENJEV_BACKEND):
  http      POST to the OpenJev service (services/openjev_server.py, Docker) at $OPENJEV_URL
  local     load weights in-process with transformers
  fallback  keyword scorer with the same interface. Used when weights cannot be downloaded
            (e.g. offline CI). Every Decision records which backend produced it.
"""
from __future__ import annotations

import os
import re
import threading
import time
from functools import lru_cache

import yaml

from hiring_intel.contracts import Decision
from hiring_intel.llm import ROOT

CFG = yaml.safe_load((ROOT / "config/models.yaml").read_text())["openjev"]
MIN_CONF = float(CFG.get("min_confidence", 0.55))


def _softmax_norm(scores: list[float]) -> list[float]:
    s = sum(scores)
    return [x / s for x in scores] if s > 0 else [1 / len(scores)] * len(scores)


class LocalOpenJev:
    name = "openjev-local"

    def __init__(self, repo: str, subfolder: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=subfolder).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        labels = {v.lower(): int(k) for k, v in (self.model.config.id2label or {}).items()}
        # model card: label order is contradiction, entailment, neutral
        self.ent_idx = next((i for l, i in labels.items() if l.startswith("entail")), 1)
        self.template = getattr(self.model.config, "nli_template", None)
        self._lock = threading.Lock()

    def entail(self, premise: str, hypotheses: list[str]) -> list[float]:
        if self.template:
            texts = [self.template.format(premise=premise, hypothesis=h) for h in hypotheses]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
        else:
            enc = self.tok([premise] * len(hypotheses), hypotheses, return_tensors="pt",
                           padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._lock, self.torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)[:, self.ent_idx]
        return probs.float().cpu().tolist()


class HttpOpenJev:
    name = "openjev-http"

    def __init__(self, url: str):
        import httpx
        self.client = httpx.Client(base_url=url, timeout=30)

    def entail(self, premise: str, hypotheses: list[str]) -> list[float]:
        r = self.client.post("/entail", json={"premise": premise, "hypotheses": hypotheses})
        r.raise_for_status()
        return r.json()["entailment"]


class KeywordFallback:
    """Same interface, no model. Scores each option by keyword hits in the premise."""
    name = "keyword-fallback"

    def entail(self, premise: str, hypotheses: list[str], keywords: list[list[str]] | None = None) -> list[float]:
        text = premise.lower()
        if not keywords:
            return [0.5] * len(hypotheses)
        # an option with no keywords is the default answer: fixed baseline score
        return [0.05 + sum(1.0 for k in kws if re.search(r"\b" + re.escape(k.lower()), text)) if kws else 0.6
                for kws in keywords]


@lru_cache(maxsize=1)
def get_backend():
    mode = os.getenv("OPENJEV_BACKEND", CFG.get("backend", "auto"))
    url = os.getenv("OPENJEV_URL")
    if mode in ("http", "auto") and url:
        return HttpOpenJev(url)
    if mode in ("local", "auto"):
        try:
            return LocalOpenJev(CFG["repo"], CFG["subfolder"])
        except Exception:                               # no torch / no network / no weights
            if mode == "local":
                raise
    return KeywordFallback()


class OpenJevDecider:
    def __init__(self, backend=None):
        self.backend = backend or get_backend()
        self._memo: dict[tuple, Decision] = {}

    def choose(self, task: str, premise: str, question: str, options: list[str],
               keywords: dict[str, list[str]] | None = None) -> Decision:
        """Pick one option. `keywords` is only used by the fallback backend."""
        memo_key = (task, premise, question, tuple(options))
        if memo_key in self._memo:
            d = self._memo[memo_key]
            return d.model_copy(update={"latency_ms": 0})
        t0 = time.perf_counter()
        hyps = [f"{question} Answer: {o}" for o in options]
        if isinstance(self.backend, KeywordFallback):
            raw = self.backend.entail(premise, hyps, [keywords.get(o, [o]) if keywords is not None else [o]
                                                     for o in options])
        else:
            raw = self.backend.entail(premise, hyps)
        probs = _softmax_norm(raw)
        best = max(range(len(options)), key=lambda i: probs[i])
        d = Decision(task=task, choice=options[best], confidence=round(probs[best], 3),
                     probs={o: round(p, 3) for o, p in zip(options, probs)}, backend=self.backend.name,
                     latency_ms=int((time.perf_counter() - t0) * 1000))
        d.escalated = d.confidence < MIN_CONF
        self._memo[memo_key] = d
        return d

    def yes_no(self, task: str, premise: str, question: str, yes_keywords: list[str] | None = None,
               no_keywords: list[str] | None = None) -> Decision:
        return self.choose(task, premise, question, ["yes", "no"],
                           keywords={"yes": yes_keywords or [], "no": no_keywords or []})


_decider: OpenJevDecider | None = None


def decider() -> OpenJevDecider:
    global _decider
    if _decider is None:
        _decider = OpenJevDecider()
    return _decider
```

## Step 7.2 — Note labelling (enrichment)

`mode="llm"` labels the same notes with batched small-LLM calls. It is the expensive baseline the benchmark compares against.

**File:** `hiring_intel/enrich.py`
```python
"""Enrichment: fill missing rejection / decline reasons by classifying free-text notes with OpenJev.

This is the highest-volume "simple decision" in the system: thousands of notes, a closed label
set, no generation needed. Running it on an LLM would cost tokens per note; OpenJev costs $0 API.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from pydantic import BaseModel

from hiring_intel.ingest import HiringData
from hiring_intel.llm import call_structured, model_for, price_usd
from hiring_intel.observability import log_decision
from hiring_intel.openjev import KeywordFallback, OpenJevDecider


class NoteLabels(BaseModel):
    labels: list[str]

REJECTION_LABELS = {
    "JD mismatch": ["jd", "job description", "posting", "does not match", "different scope", "native"],
    "fundamentals": ["basic", "fundamentals", "data structures", "complexity", "sql", "joins", "indexing",
                     "recursion", "hashing"],
    "system design": ["design", "scaling", "partition", "consistency", "failure modes"],
    "communication": ["explanations", "clarify", "rambled", "structure the answer", "listen"],
    "coding": ["coding", "code", "bugs", "edge cases", "tests"],
    "experience gap": ["junior", "years of experience", "ownership"],
    "culture add": ["collaboration", "dismissive", "feedback"],
}
DECLINE_LABELS = {
    "compensation": ["below", "compensation", "ctc", "20% more", "expectation"],
    "competing offer": ["another company", "competing"],
    "timing": ["too long", "notice period", "start date", "moved on"],
    "location": ["relocate", "remote", "location"],
}


def _classify(decider: OpenJevDecider, run_id: str, task: str, notes: pd.Series, labels: dict,
              mode: str = "openjev") -> dict[str, str]:
    mapping = {}
    counts = notes.value_counts()
    if mode == "llm":
        # the expensive alternative (benchmark baseline): real notes are mostly unique, so every note
        # goes to the LLM, 50 notes per call
        kw = OpenJevDecider(KeywordFallback())
        notes_l = [str(n) for n in notes.tolist()]
        for b in range(0, len(notes_l), 50):
            batch = notes_l[b:b + 50]
            parsed, _ = call_structured(
                agent=f"enrich:{task}", run_id=run_id, model=model_for("small"),
                system=f"Label each numbered note with exactly one of: {list(labels)}. "
                       "Return {\"labels\": [...]} in the same order.",
                user="\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch)), schema=NoteLabels,
                mock_fn=lambda batch=batch: NoteLabels(labels=[
                    kw.choose(task, t, "reason?", list(labels), keywords=labels).choice for t in batch]))
            for t, lab in zip(batch, parsed.labels):
                mapping[t] = lab
        return mapping
    for note, n in counts.items():                      # unique notes only; results memoised
        d = decider.choose(task, str(note), "What is the main reason described in this note?",
                           list(labels), keywords=labels)
        log_decision(run_id, d, n_items=int(n))
        # low confidence: don't force a label; the note stays visible as "unclassified"
        mapping[note] = "unclassified" if d.escalated else d.choice
    return mapping


def enrich(data: HiringData, run_id: str, decider: OpenJevDecider, mode: str = "openjev") -> dict:
    t0 = time.perf_counter()
    ev = data.stage_events
    if "reason" not in ev.columns:
        ev["reason_source"] = "structured"
        ev["reason"] = ev["rejection_reason"]
    need = ev[(ev.result == "reject") & ev.rejection_reason.isna() & ev.rejection_note.notna()]
    rmap = _classify(decider, run_id, "classify_rejection_note", need.rejection_note, REJECTION_LABELS, mode)
    ev.loc[need.index, "reason"] = need.rejection_note.map(rmap)
    ev.loc[need.index, "reason_source"] = mode

    of = data.offers
    if "reason" not in of.columns:
        of["reason"] = of["decline_reason"]
    dneed = of[(of.status == "declined") & of.decline_reason.isna() & of.decline_note.notna()]
    dmap = _classify(decider, run_id, "classify_decline_note", dneed.decline_note, DECLINE_LABELS, mode)
    of.loc[dneed.index, "reason"] = dneed.decline_note.map(dmap)

    # what an LLM would have cost for the same job (one small-model call per note, ~120 tokens in / 10 out)
    n_items = len(need) + len(dneed)
    small = model_for("small")
    llm_usd = price_usd(small, 120 * n_items, 10 * n_items)
    accuracy = _accuracy(data)
    return {"classified": n_items, "unique_notes": len(rmap) + len(dmap),
            "backend": "llm-small" if mode == "llm" else decider.backend.name,
            "compute_ms": int((time.perf_counter() - t0) * 1000), "llm_equivalent_usd": round(llm_usd, 4),
            "accuracy_vs_truth": accuracy}


def _accuracy(data: HiringData) -> float | None:
    f = Path(data.path) / "_labels.csv"
    if not f.exists():
        return None
    lab = pd.read_csv(f)
    rej = lab[lab.kind == "rejection"].merge(
        data.stage_events[data.stage_events.reason_source != "structured"][["candidate_id", "stage", "reason"]],
        on=["candidate_id", "stage"])
    return round(float((rej.true_reason == rej.reason).mean()), 3) if len(rej) else None
```

## Step 7.3 — OpenJev as a service

The app containers stay small (no torch); the model loads once in its own container (chapter 20).

**File:** `services/openjev_server.py`
```python
"""OpenJev as a small HTTP service, so the app containers stay light (no torch) and the model loads once.

    uvicorn services.openjev_server:app --port 8200
    POST /entail  {"premise": "...", "hypotheses": ["...", "..."]}  -> {"entailment": [0.91, 0.04]}
    POST /decide  {"premise": "...", "question": "...", "options": ["yes", "no"]} -> Decision
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hiring_intel.openjev import CFG, LocalOpenJev, OpenJevDecider  # noqa: E402

app = FastAPI(title="OpenJev decision service")
_model = LocalOpenJev(os.getenv("OPENJEV_REPO", CFG["repo"]), os.getenv("OPENJEV_SUBFOLDER", CFG["subfolder"]))
_decider = OpenJevDecider(_model)


class EntailReq(BaseModel):
    premise: str
    hypotheses: list[str]


class DecideReq(BaseModel):
    premise: str
    question: str
    options: list[str]
    task: str = "api"


@app.get("/health")
def health():
    return {"ok": True, "device": _model.device, "subfolder": os.getenv("OPENJEV_SUBFOLDER", CFG["subfolder"])}


@app.post("/entail")
def entail(r: EntailReq):
    return {"entailment": _model.entail(r.premise, r.hypotheses)}


@app.post("/decide")
def decide(r: DecideReq):
    return _decider.choose(r.task, r.premise, r.question, r.options).model_dump()
```

## Step 7.4 — Run the real model (optional)

```bash
pip install -r requirements-ml.txt
OPENJEV_BACKEND=local python -c "from hiring_intel.openjev import decider; print(decider().yes_no('t','The offer is 20% below expectation.','Is compensation the problem?'))"
# or as a service:
uvicorn services.openjev_server:app --port 8200      # then OPENJEV_URL=http://localhost:8200
```

## Verify

<!-- verify:07 -->
```bash
python - <<'EOF'
from hiring_intel.ingest import load_all
from hiring_intel.enrich import enrich
from hiring_intel.openjev import decider
d = load_all("data/raw/baseline_large")
print("openjev:", enrich(d, "ch07", decider()))
d2 = load_all("data/raw/baseline_medium")
print("llm    :", enrich(d2, "ch07-llm", decider(), mode="llm"))
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
openjev: {'classified': 14724, 'unique_notes': 25, 'backend': 'keyword-fallback', 'compute_ms': 143, 'llm_equivalent_usd': 0.3534, 'accuracy_vs_truth': 0.92}
llm    : {'classified': 3660, 'unique_notes': 25, 'backend': 'llm-small', 'compute_ms': 237, 'llm_equivalent_usd': 0.0878, 'accuracy_vs_truth': 0.927}
```

The `llm` mode pays per note; OpenJev labels the same notes at $0 API cost. `llm_equivalent_usd` is what the OpenJev job would have cost on the small LLM. The keyword fallback leaves the ambiguous notes `unclassified` instead of guessing, which is where most of the 8% accuracy gap comes from.

## Checkpoint

- [ ] notes are labelled and accuracy is reported
- [ ] the `decisions` table records the backend for each decision
