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