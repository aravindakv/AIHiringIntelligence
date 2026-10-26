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


def test_service_down_is_caught_at_start(monkeypatch):
    """OPENJEV_URL set but the service not running: the run used to crash at human review with
    'Connection refused'. auto now falls back to keywords with a warning; http fails at once, clearly."""
    import pytest
    from hiring_intel import openjev
    monkeypatch.setenv("OPENJEV_URL", "http://127.0.0.1:9")          # nothing listens here
    try:
        monkeypatch.setenv("OPENJEV_BACKEND", "auto")
        openjev.get_backend.cache_clear()
        assert isinstance(openjev.get_backend(), KeywordFallback)
        monkeypatch.setenv("OPENJEV_BACKEND", "http")
        openjev.get_backend.cache_clear()
        with pytest.raises(openjev.OpenJevUnavailable, match="not reachable"):
            openjev.get_backend()
    finally:
        openjev.get_backend.cache_clear()