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