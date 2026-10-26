import litellm
import pytest

from hiring_intel import llm
from hiring_intel.agents.insight_agents import AGENTS
from hiring_intel.ingest import load_all

QUOTA = ("Your project has reached its configured enforced spend limit. "
         "{'error': {'type': 'insufficient_quota', 'code': 'project_spend_limit_exceeded'}}")


def test_spend_limit_stops_the_run_at_once(monkeypatch):
    """A real run hit OpenAI's spend limit: LiteLLM retried a permanent error, every golden case re-ran
    the pipeline, and the report said 0/18 as if quality were zero. Now the first refusal trips a
    breaker, the agent re-raises instead of reporting 'failed', and later calls never reach the network."""
    calls = []

    def refuse(**kw):
        calls.append(kw["model"])
        raise litellm.RateLimitError(message=QUOTA, llm_provider="openai", model=kw["model"])

    monkeypatch.setattr(litellm, "completion", refuse)
    monkeypatch.setenv("LLM_BACKEND", "litellm")
    llm.reset_unavailable()
    data = load_all("data/raw/baseline_small")
    with pytest.raises(llm.LLMUnavailable, match="spend limit"):
        AGENTS["pipeline"].run(data, "t-quota", force_tier="small")
    assert len(calls) == 1                                   # no retries on a permanent error
    with pytest.raises(llm.LLMUnavailable):
        llm.preflight()
    assert len(calls) == 1                                   # breaker open: no further network calls
    llm.reset_unavailable()


def test_transient_rate_limit_is_retried(monkeypatch):
    calls = []

    def flaky(**kw):
        calls.append(1)
        if len(calls) == 1:
            raise litellm.RateLimitError(message="Rate limit reached for requests per min", llm_provider="openai",
                                         model=kw["model"])
        return "ok"

    monkeypatch.setattr(litellm, "completion", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm.reset_unavailable()
    assert llm.completion(model="gpt-4o-mini", messages=[]) == "ok" and len(calls) == 2