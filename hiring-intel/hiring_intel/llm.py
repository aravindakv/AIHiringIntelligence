"""One door for every LLM call: model call, JSON validation + repair, cost, latency, cache.

Backends (env LLM_BACKEND):
  litellm  real calls through LiteLLM (OpenAI, Anthropic, Gemini, Ollama...). Needs API keys.
  mock     offline. The agent's rule-based analyst (`mock_fn`) produces the JSON; tokens are
           estimated (chars/4) and latency is simulated from config/models.yaml. Everything
           downstream (validation, cost, eval, dashboard) runs exactly as with a real model.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from hiring_intel.contracts import Cost
from hiring_intel.observability import log_call

try:                                   # LangSmith tracing: a no-op unless LANGSMITH_TRACING=true
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:                    # pragma: no cover
    def traceable(**_kw):
        return lambda f: f

    def get_current_run_tree():
        return None

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config/models.yaml").read_text())
TIER_OF = {t["model"]: name for name, t in CFG["tiers"].items()}
_cache: dict[str, tuple[BaseModel, Cost]] = {}
_unavailable: str | None = None          # circuit breaker: set on the first permanent provider error


class LLMUnavailable(RuntimeError):
    """The provider refuses every call (quota or spend limit reached, bad key, no access to the model).
    Retrying can't help, so the run stops instead of failing the same way on every remaining call."""


PERMANENT_MARKERS = ("insufficient_quota", "spend limit", "spend_limit", "exceeded your current quota",
                     "billing", "invalid api key", "incorrect api key", "invalid_api_key")


def _is_permanent(e: Exception) -> bool:
    import litellm
    if isinstance(e, (litellm.AuthenticationError, litellm.PermissionDeniedError, litellm.NotFoundError,
                      litellm.BudgetExceededError)):
        return True
    return any(m in str(e).lower() for m in PERMANENT_MARKERS)


def _is_transient(e: Exception) -> bool:
    import litellm
    return isinstance(e, (litellm.RateLimitError, litellm.Timeout, litellm.APIConnectionError,
                          litellm.InternalServerError, litellm.ServiceUnavailableError))


def completion(**kw):
    """litellm.completion with our own retry policy: up to 2 retries with backoff for transient errors
    (rate limit per minute, timeout, 5xx); none for permanent ones, which trip the circuit breaker."""
    global _unavailable
    import litellm
    if _unavailable:
        raise LLMUnavailable(_unavailable)
    for attempt in range(3):
        try:
            return litellm.completion(num_retries=0, **kw)
        except Exception as e:
            if _is_permanent(e):
                _unavailable = (f"{kw.get('model')}: {type(e).__name__}: {str(e)[:300]}. Fix the provider account "
                                "(quota, spend limit or key), or run offline with LLM_BACKEND=mock.")
                raise LLMUnavailable(_unavailable) from e
            if not _is_transient(e) or attempt == 2:
                raise
            time.sleep(2 ** attempt * 2)


def reset_unavailable() -> None:
    global _unavailable
    _unavailable = None


def preflight() -> None:
    """One tiny call before an expensive job: fails in about a second if the account can't serve it."""
    if backend() == "mock":
        return
    completion(model=model_for("small"), messages=[{"role": "user", "content": "Reply with: ok"}], max_tokens=3,
               timeout=30)

def backend() -> str:
    return os.getenv("LLM_BACKEND", "mock")


def model_for(tier: str) -> str:
    return CFG["tiers"][tier]["model"]


def est_tokens(text: str) -> int:
    """~4 characters per token for English/CSV. Real runs use the provider's usage counts."""
    return max(1, len(text) // 4)


def price_usd(model: str, inp: int, out: int) -> float:
    tier = TIER_OF.get(model)
    if tier:
        p = CFG["tiers"][tier]
        return inp / 1e6 * p["input_per_1m"] + out / 1e6 * p["output_per_1m"]
    try:                                            # unknown model: LiteLLM's price table
        import litellm
        return float(sum(litellm.cost_per_token(model=model, prompt_tokens=inp, completion_tokens=out)))
    except Exception:
        return 0.0


def simulated_latency_ms(model: str, inp: int, out: int, seed: str) -> int:
    p = CFG["tiers"][TIER_OF.get(model, "small")]
    jitter = random.Random(seed).uniform(0.85, 1.25)       # deterministic per call
    ms = p["sim_base_ms"] + p["sim_ms_per_out_token"] * out + p["sim_ms_per_1k_in_tokens"] * inp / 1000
    return int(ms * jitter)


def clear_cache() -> None:
    _cache.clear()


@traceable(run_type="chain", name="call_structured")
def call_structured(*, agent: str, run_id: str, model: str, system: str, user: str,
                    schema: type[BaseModel], mock_fn: Callable[[], BaseModel] | None = None,
                    temperature: float = 0.1, use_cache: bool = False,
                    max_repairs: int = 1) -> tuple[BaseModel, Cost]:
    tier = TIER_OF.get(model, "custom")
    key = hashlib.sha256(f"{model}|{system}|{user}".encode()).hexdigest()
    if use_cache and key in _cache:
        parsed, cost = _cache[key]
        hit = cost.model_copy(update={"usd": 0.0, "latency_ms": 2, "cached": True})
        log_call(agent, run_id, hit)
        return parsed, hit

    full_system = (system + "\n\nReturn ONLY JSON matching this schema:\n"
                   + json.dumps(schema.model_json_schema()))

    if backend() == "mock":
        if mock_fn is None:
            raise RuntimeError(f"{agent}: mock backend needs a mock_fn")
        raw = mock_fn().model_dump_json()
        parsed = schema.model_validate_json(raw)            # same validation path as real output
        inp, out = est_tokens(full_system + user), est_tokens(raw)
        cost = Cost(model=model, tier=tier, input_tokens=inp, output_tokens=out,
                    usd=round(price_usd(model, inp, out), 6),
                    latency_ms=simulated_latency_ms(model, inp, out, key), simulated=True)
    else:
        messages = [{"role": "system", "content": full_system}, {"role": "user", "content": user}]
        inp = out = 0
        t0 = time.perf_counter()
        for attempt in range(max_repairs + 1):
            resp = completion(model=model, messages=messages, temperature=temperature,
                              response_format={"type": "json_object"}, timeout=90)
            inp += resp.usage.prompt_tokens
            out += resp.usage.completion_tokens
            raw = resp.choices[0].message.content
            try:
                parsed = schema.model_validate_json(raw)
                break
            except ValidationError as e:
                if attempt == max_repairs:
                    raise
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Your JSON failed validation: {e}. Return fixed JSON only."}]
        cost = Cost(model=model, tier=tier, input_tokens=inp, output_tokens=out,
                    usd=round(price_usd(model, inp, out), 6),
                    latency_ms=int((time.perf_counter() - t0) * 1000))

    _cache[key] = (parsed, cost)
    log_call(agent, run_id, cost)
    _tag_trace(agent, run_id, cost)
    return parsed, cost


def _tag_trace(agent: str, run_id: str, cost: Cost) -> None:
    """Put model, tokens, USD and latency on the LangSmith span so traces can be filtered by them."""
    rt = get_current_run_tree()
    if rt is not None:
        rt.add_metadata({"agent": agent, "run_id": run_id, **cost.model_dump()})
        rt.add_tags([agent, cost.tier, "cached" if cost.cached else "fresh"])