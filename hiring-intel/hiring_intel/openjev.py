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

import os
import re
import threading
import time
from functools import lru_cache

import yaml

from hiring_intel.contracts import Decision
from hiring_intel.llm import ROOT

CFG = yaml.safe_load((ROOT / "config/models.yaml").read_text())["openjev"]
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


class HttpOpenJev:
    name = "openjev-http"

    def __init__(self, url: str):
        import httpx
        # CPU inference can take 10-20s per call on a busy machine; override with OPENJEV_TIMEOUT
        self.client = httpx.Client(base_url=url, timeout=float(os.getenv("OPENJEV_TIMEOUT", "300")))

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