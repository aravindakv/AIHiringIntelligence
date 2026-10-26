
import pytest
from hiring_intel.contracts import Alternative, Evidence, Insight
from pydantic import ValidationError

ALT = Alternative(option="rules", trade_off="no narrative")


def ok(**kw):
    base = {"title": "t", "recommendation": "Recruiter lead: move 20% of agency budget to referrals this quarter.",
                "evidence": [Evidence(metric="m", value=0.1)], "confidence_score": 0.7, "alternative": ALT}
    base.update(kw)
    return Insight(**base)


def test_valid_insight():
    assert ok().confidence_score == 0.7


@pytest.mark.parametrize("kw", [{"confidence_score": 1.4}, {"evidence": []},
                                {"recommendation": "Consider improving the sourcing process overall please."}])
def test_rejects_bad_insights(kw):
    with pytest.raises(ValidationError):
        ok(**kw)