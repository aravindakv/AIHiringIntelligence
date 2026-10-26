# 07 — OpenJev for simple decisions

**Goal:** a decision layer that answers closed-set questions (which label, yes or no) without an LLM, and uses it to label thousands of free-text notes.

**Time:** 45 minutes (plus model download if you run it locally).

## Concepts

- **What OpenJev is.** An open NLI cross-encoder ([huggingface.co/AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), MIT per its model card), built on Qwen3.5. Given a premise and a hypothesis it returns P(contradiction), P(entailment) and P(neutral). To choose between options, score one hypothesis per option and take the highest entailment. Write each hypothesis as a plain statement ("The candidate's system design was weak."). On real notes that took accuracy from 0.73 to 1.00 versus "…Answer: fundamentals". It doesn't generate text, it is deterministic, and the option order doesn't matter.
- **Where it pays off.** Decisions with a fixed answer set, made many times:

| Decision | Options | Used in |
| --- | --- | --- |
| Label a rejection note | 7 reasons | this chapter |
| Label a decline note | comp / competing / timing / location | this chapter |
| "Does this task need causal reasoning?" | yes / no | Routing Agent, ch. 08 |
| "Does this change pay, headcount or workload?" | yes / no | human review, ch. 11 |
| "Is this recommendation actionable?" | yes / no | evaluator, ch. 10 |
| DeepEval faithfulness / hallucination | yes / no per claim | ch. 17 |

- **Escalate, don't guess.** A decision is marked `escalated` below the floor for its option count (0.55 for yes/no, 0.30 for 3+ options) or when it picks the abstain option `unclear`. A note then becomes `unclassified`, routing falls back to a small LLM, and human review flags the item. A yes/no floor on a 7-way choice escalates correct answers: the first real-model run scored 0.385 for that reason.
- **Count the CPU time.** About 8.4 s per unique note on a laptop CPU. Fine for yes/no gates and for the 25 templates in the synthetic data; for thousands of unique real notes, use a GPU or the small LLM.
- **Three backends.** `http` (the Docker service), `local` (weights in-process), `fallback` (a keyword scorer with the same interface, for offline CI). Every decision records which one answered.
- **Caveats** from the model card: probabilities are not calibrated to your labels, it can be swayed by injected text, and its v5 checkpoints saw public benchmark test sets. Measure it on your own labelled notes.

## Step 7.1 — The decider

**File:** `hiring_intel/openjev.py`
```python
"""OpenJev for simple, closed-set decisions.

OpenJev (huggingface.co/AlexWortega/openjev, MIT) is Qwen3.5 fine-tuned as an NLI cross-encoder:
given (premise, hypothesis) it returns P(contradiction), P(entailment), P(neutral). A decision over
options = one hypothesis per option, pick the highest entailment. Hypotheses should be plain
statements ("The candidate's system design was weak."): NLI models are trained on those, and on
real notes this lifted labelling accuracy from 0.73 to 1.0 versus "<question> Answer: <option>".
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

import logging
import os
import re
import threading
import time
from functools import lru_cache

import yaml

from hiring_intel.contracts import Decision
from hiring_intel.llm import ROOT

CFG = yaml.safe_load((ROOT / "config/models.yaml").read_text())["openjev"]
log = logging.getLogger("hiring_intel.openjev")
MIN_CONF = float(CFG.get("min_confidence", 0.55))              # yes / no
MIN_CONF_MULTI = float(CFG.get("min_confidence_multi", 0.30))  # 3+ options: probability is spread wider


def floor_for(n_options: int) -> float:
    return MIN_CONF if n_options <= 2 else MIN_CONF_MULTI


def _softmax_norm(scores: list[float]) -> list[float]:
    s = sum(scores)
    return [x / s for x in scores] if s > 0 else [1 / len(scores)] * len(scores)


class LocalOpenJev:
    name = "openjev-local"

    def __init__(self, repo: str, subfolder: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _check_subfolder(repo, subfolder)
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


def _check_subfolder(repo: str, subfolder: str) -> None:
    """A missing subfolder makes transformers fail with a misleading tokenizer error; fail clearly instead."""
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo)
    except Exception:                                  # offline / no hub access: let transformers try the cache
        return
    if not any(f.startswith(subfolder + "/") for f in files):
        have = sorted({f.split("/")[0] for f in files if f.count("/") == 1 and f.endswith("/config.json")})
        raise ValueError(f"OpenJev checkpoint '{subfolder}' is not in {repo}. Available: {have}. "
                         "Set openjev.subfolder in config/models.yaml or env OPENJEV_SUBFOLDER.")


class OpenJevUnavailable(RuntimeError):
    """The OpenJev service can't be reached."""


START_HINT = "start it with `uvicorn services.openjev_server:app --port 8200`, or unset OPENJEV_URL"


class HttpOpenJev:
    name = "openjev-http"

    def __init__(self, url: str):
        import httpx
        self.url = url
        self.client = httpx.Client(base_url=url, timeout=300)   # one CPU decision can take 10-40 s

    def health(self) -> str | None:
        """None if the service answers /health, else the reason it doesn't."""
        try:
            r = self.client.get("/health", timeout=5)
            r.raise_for_status()
            return None
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    def entail(self, premise: str, hypotheses: list[str]) -> list[float]:
        import httpx
        try:
            r = self.client.post("/entail", json={"premise": premise, "hypotheses": hypotheses})
        except httpx.TransportError as e:                 # the service went away mid-run
            raise OpenJevUnavailable(f"OpenJev service at {self.url} stopped answering ({e}); {START_HINT}") from e
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
        http = HttpOpenJev(url)
        why = http.health()                              # checked once per process (lru_cache)
        if why is None:
            return http
        msg = f"OpenJev service at {url} is not reachable ({why}); {START_HINT}"
        if mode == "http":
            raise OpenJevUnavailable(msg)
        # auto: don't load the 0.8B model in-process by surprise; use the keyword fallback, which every
        # Decision records as backend="keyword-fallback"
        log.warning("%s. Using the keyword fallback for this process.", msg)
        return KeywordFallback()
    if mode in ("local", "auto"):
        try:
            return LocalOpenJev(CFG["repo"], os.getenv("OPENJEV_SUBFOLDER", CFG["subfolder"]))
        except Exception:                               # no torch / no network / no weights
            if mode == "local":
                raise
    return KeywordFallback()


class OpenJevDecider:
    def __init__(self, backend=None):
        self.backend = backend or get_backend()
        self._memo: dict[tuple, Decision] = {}

    def choose(self, task: str, premise: str, question: str, options: list[str],
               keywords: dict[str, list[str]] | None = None, hypotheses: dict[str, str] | None = None,
               premise_prefix: str = "", abstain: str | None = None) -> Decision:
        """Pick one option.

        hypotheses      option -> declarative sentence; without it, "<question> Answer: <option>"
        premise_prefix  context for the model, e.g. "Interviewer note: "
        abstain         an option meaning "no clear answer"; choosing it escalates the decision
        keywords        only used by the fallback backend
        """
        memo_key = (task, premise, question, tuple(options), tuple(sorted((hypotheses or {}).items())))
        if memo_key in self._memo:
            d = self._memo[memo_key]
            return d.model_copy(update={"latency_ms": 0})
        t0 = time.perf_counter()
        hyps = [hypotheses[o] if hypotheses and o in hypotheses else f"{question} Answer: {o}" for o in options]
        if isinstance(self.backend, KeywordFallback):
            raw = self.backend.entail(premise, hyps, [keywords.get(o, [o]) if keywords is not None else [o]
                                                     for o in options])
        else:
            raw = self.backend.entail(premise_prefix + premise, hyps)
        probs = _softmax_norm(raw)
        best = max(range(len(options)), key=lambda i: probs[i])
        d = Decision(task=task, choice=options[best], confidence=round(probs[best], 3),
                     probs={o: round(p, 3) for o, p in zip(options, probs)}, backend=self.backend.name,
                     latency_ms=int((time.perf_counter() - t0) * 1000))
        d.escalated = d.confidence < floor_for(len(options)) or d.choice == abstain
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


class NoteLabel(BaseModel):
    i: int              # the note's number in the batch, so a skipped or reordered label can't shift the rest
    label: str


class NoteLabels(BaseModel):
    labels: list[NoteLabel]

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
REJECTION_HYPOTHESES = {
    "JD mismatch": "The candidate's background does not match the job description.",
    "fundamentals": "The candidate is weak on basic computer science or SQL fundamentals.",
    "system design": "The candidate's system design was weak.",
    "communication": "The candidate communicated poorly.",
    "coding": "The candidate's code was buggy or incomplete.",
    "experience gap": "The candidate is too junior or lacks enough experience.",
    "culture add": "The candidate showed poor collaboration or teamwork.",
}
DECLINE_LABELS = {
    "compensation": ["below", "compensation", "ctc", "20% more", "expectation"],
    "competing offer": ["another company", "competing"],
    "timing": ["too long", "notice period", "start date", "moved on"],
    "location": ["relocate", "remote", "location"],
}
DECLINE_HYPOTHESES = {
    "compensation": "The candidate declined because the pay offered was too low.",
    "competing offer": "The candidate accepted an offer from another company.",
    "timing": "The candidate declined because the process or the start date did not work out in time.",
    "location": "The candidate declined because of the job location or relocation.",
}
# the abstain option: notes like "Panel was not convinced overall." should escalate, not get a label
UNCLEAR = "unclear"
UNCLEAR_HYPOTHESIS = "The note does not name any specific skill, problem or reason."
HYPOTHESES = {"classify_rejection_note": REJECTION_HYPOTHESES, "classify_decline_note": DECLINE_HYPOTHESES}
PREFIX = {"classify_rejection_note": "Interviewer note: ", "classify_decline_note": "Recruiter note on a declined offer: "}


def _canon(label: str, labels: dict) -> str:
    """Map an LLM's label onto the label set (case and spacing tolerant); anything else is unclassified."""
    norm = {k.lower().replace("_", " ").strip(): k for k in labels}
    return norm.get(str(label).lower().replace("_", " ").strip(), "unclassified")


def _classify(decider: OpenJevDecider, run_id: str, task: str, notes: pd.Series, labels: dict,
              mode: str = "openjev") -> pd.Series:
    """Return one label per row of `notes` (same index)."""
    if mode == "llm":
        # the expensive alternative (benchmark baseline): real notes are mostly unique, so every note
        # goes to the LLM, 50 notes per call. Labels are matched by number and per row, never by note
        # text, so one bad batch can't relabel every other copy of the same note.
        kw = OpenJevDecider(KeywordFallback())
        notes_l = [str(n) for n in notes.tolist()]
        out = ["unclassified"] * len(notes_l)
        for b in range(0, len(notes_l), 50):
            batch = notes_l[b:b + 50]
            parsed, _ = call_structured(
                agent=f"enrich:{task}", run_id=run_id, model=model_for("small"),
                system=f"Label each numbered interviewer or recruiter note with exactly one of: {list(labels)}. "
                       "Use the label text exactly. Return one entry per note: "
                       '{"labels": [{"i": <note number>, "label": "<label>"}, ...]}.',
                user="\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch)), schema=NoteLabels,
                mock_fn=lambda batch=batch: NoteLabels(labels=[
                    NoteLabel(i=i + 1, label=kw.choose(task, t, "reason?", list(labels), keywords=labels).choice)
                    for i, t in enumerate(batch)]))
            for item in parsed.labels:
                if 1 <= item.i <= len(batch):
                    out[b + item.i - 1] = _canon(item.label, labels)
        return pd.Series(out, index=notes.index)
    mapping = {}
    options = list(labels) + [UNCLEAR]
    hyps = {**HYPOTHESES[task], UNCLEAR: UNCLEAR_HYPOTHESIS}
    for note, n in notes.value_counts().items():         # unique notes only; results memoised
        d = decider.choose(task, str(note), "What is the main reason described in this note?", options,
                           keywords=labels, hypotheses=hyps, premise_prefix=PREFIX[task], abstain=UNCLEAR)
        log_decision(run_id, d, n_items=int(n))
        # unclear or low confidence: don't force a label; the note stays visible as "unclassified"
        mapping[note] = "unclassified" if d.escalated else d.choice
    return notes.map(mapping)


def enrich(data: HiringData, run_id: str, decider: OpenJevDecider, mode: str = "openjev") -> dict:
    t0 = time.perf_counter()
    ev = data.stage_events
    if "reason" not in ev.columns:
        ev["reason_source"] = "structured"
        ev["reason"] = ev["rejection_reason"]
    need = ev[(ev.result == "reject") & ev.rejection_reason.isna() & ev.rejection_note.notna()]
    ev.loc[need.index, "reason"] = _classify(decider, run_id, "classify_rejection_note", need.rejection_note,
                                             REJECTION_LABELS, mode)
    ev.loc[need.index, "reason_source"] = mode

    of = data.offers
    if "reason" not in of.columns:
        of["reason"] = of["decline_reason"]
    dneed = of[(of.status == "declined") & of.decline_reason.isna() & of.decline_note.notna()]
    of.loc[dneed.index, "reason"] = _classify(decider, run_id, "classify_decline_note", dneed.decline_note,
                                              DECLINE_LABELS, mode)

    # what an LLM would have cost for the same job (one small-model call per note, ~120 tokens in / 10 out)
    n_items = len(need) + len(dneed)
    small = model_for("small")
    llm_usd = price_usd(small, 120 * n_items, 10 * n_items)
    acc = _accuracy(data) or {}
    ms = int((time.perf_counter() - t0) * 1000)
    uniq = need.rejection_note.nunique() + dneed.decline_note.nunique()
    return {"classified": n_items, "unique_notes": uniq,
            "backend": "llm-small" if mode == "llm" else decider.backend.name,
            "compute_ms": ms, "ms_per_unique_note": round(ms / max(uniq, 1)),
            "llm_equivalent_usd": round(llm_usd, 4), **acc}


def _accuracy(data: HiringData) -> dict | None:
    """accuracy_vs_truth counts "unclassified" as wrong; accuracy_when_labelled ignores escalated notes.
    Keep both: a labeller that escalates the ambiguous 8% is doing its job, one that mislabels them is not."""
    f = Path(data.path) / "_labels.csv"
    if not f.exists():
        return None
    lab = pd.read_csv(f)
    rej = lab[lab.kind == "rejection"].merge(
        data.stage_events[data.stage_events.reason_source != "structured"][["candidate_id", "stage", "reason"]],
        on=["candidate_id", "stage"])
    if not len(rej):
        return None
    hit = rej.true_reason == rej.reason
    esc = rej.reason == "unclassified"
    return {"accuracy_vs_truth": round(float(hit.mean()), 3),
            "accuracy_when_labelled": round(float(hit[~esc].mean()), 3) if (~esc).any() else None,
            "escalated_share": round(float(esc.mean()), 3)}
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

The repo's real checkpoints are `qwen3.5-0.8b-nli-v2s-long` (CPU, the default) and `qwen3.5-4b-nli-v2` (GPU). The model card also mentions `-v5` checkpoints that aren't published; pointing at one gives a misleading tokenizer error, so `LocalOpenJev` checks the subfolder first and lists what exists.

```bash
pip install -r requirements-ml.txt
OPENJEV_BACKEND=local python -c "from hiring_intel.openjev import decider; print(decider().yes_no('t','The offer is 20% below expectation.','Is compensation the problem?'))"
# or as a service:
uvicorn services.openjev_server:app --port 8200      # then OPENJEV_URL=http://localhost:8200
```

Then measure labelling on your machine (about 3–4 minutes on a laptop CPU for the 21 + 4 unique notes):

```bash
time OPENJEV_BACKEND=http OPENJEV_URL=http://localhost:8200 python -c "
from hiring_intel.ingest import load_all; from hiring_intel.enrich import enrich; from hiring_intel.openjev import decider
print(enrich(load_all('data/raw/baseline_small'), 'real-openjev', decider()))"
```

On `qwen3.5-0.8b-nli-v2s-long` the question-style hypotheses with a 0.55 floor scored 0.385. Plain-statement hypotheses got every clear note right, at about 8.4 s per unique note. Expect `accuracy_when_labelled` near 1.0. `escalated_share` should be close to 0.08, the share of deliberately ambiguous notes; if it is much lower, the `unclear` option isn't catching them, so reword `UNCLEAR_HYPOTHESIS`.

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
openjev: {'classified': 14724, 'unique_notes': 25, 'backend': 'keyword-fallback', 'compute_ms': 152, 'ms_per_unique_note': 6, 'llm_equivalent_usd': 0.3534, 'accuracy_vs_truth': 0.92, 'accuracy_when_labelled': 1.0, 'escalated_share': 0.08}
llm    : {'classified': 3660, 'unique_notes': 25, 'backend': 'llm-small', 'compute_ms': 344, 'ms_per_unique_note': 14, 'llm_equivalent_usd': 0.0878, 'accuracy_vs_truth': 0.927, 'accuracy_when_labelled': 0.927, 'escalated_share': 0.0}
```

The `llm` mode pays per note; OpenJev labels the same notes at $0 API cost. `accuracy_when_labelled` ignores escalated notes and `escalated_share` counts them, so a labeller that refuses to guess on the ambiguous 8% is not mistaken for one that guesses wrong. `llm_equivalent_usd` is what the OpenJev job would have cost on the small LLM. The keyword fallback leaves the ambiguous notes `unclassified` instead of guessing, which is where most of the 8% accuracy gap comes from.

## Checkpoint

- [ ] notes are labelled and accuracy is reported
- [ ] the `decisions` table records the backend for each decision
