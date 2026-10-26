
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