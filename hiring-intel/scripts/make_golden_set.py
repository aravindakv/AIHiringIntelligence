"""Write data/golden/golden_set.jsonl: 18 pre-validated scenarios with expected outputs.

Each case runs the pipeline on a scenario dataset and checks the named agent's insights.
Checks:  title_any       at least one insight title contains one of these strings
         min_severity    the top insight is at least this severe
         max_severity    no insight is above this severity (healthy control)
         must_not_any    no insight title contains these strings
"""
import json
from pathlib import Path

CASES = [
    # baseline: all planted patterns
    ("b-src", "baseline", "sourcing", {"title_any": ["Backend: referral"], "min_severity": "medium"}),
    ("b-rej", "baseline", "rejection", {"title_any": ["[jd] Mobile screen"], "min_severity": "high"}),
    ("b-rej-data", "baseline", "rejection", {"title_any": ["agency candidates fail tech_1"]}),
    ("b-pan", "baseline", "panel_load", {"title_any": ["over load policy"]}),
    ("b-off", "baseline", "offers", {"title_any": ["below 90%"]}),
    ("b-pip", "baseline", "pipeline", {"title_any": ["SRE onsite", "SRE requisitions"]}),
    # healthy control: nothing should be raised above low
    ("h-src", "healthy", "sourcing", {"max_severity": "low"}),
    ("h-rej", "healthy", "rejection", {"max_severity": "low", "must_not_any": ["[jd]"]}),
    ("h-pan", "healthy", "panel_load", {"max_severity": "low"}),
    ("h-off", "healthy", "offers", {"max_severity": "low", "must_not_any": ["below 90%"]}),
    ("h-pip", "healthy", "pipeline", {"max_severity": "low"}),
    # comp crunch
    ("c-off", "comp_crunch", "offers", {"title_any": ["below 90%"], "min_severity": "high"}),
    ("c-rej", "comp_crunch", "rejection", {"must_not_any": ["[jd] Mobile"]}),
    # interviewer burnout
    ("i-pan", "interviewer_burnout", "panel_load", {"title_any": ["over load policy"], "min_severity": "high"}),
    ("i-pip", "interviewer_burnout", "pipeline", {"title_any": ["breaches SLA"]}),
    # sourcing shift
    ("s-src", "sourcing_shift", "sourcing", {"title_any": ["agency"], "min_severity": "high"}),
    ("s-rej", "sourcing_shift", "rejection", {"title_any": ["[sourcing] agency"]}),
    ("s-pan", "sourcing_shift", "panel_load", {"title_any": ["over load policy"]}),
]

if __name__ == "__main__":
    out = Path("data/golden/golden_set.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for cid, scenario, agent, expect in CASES:
            f.write(json.dumps({"id": cid, "scenario": scenario, "agent": agent, "expect": expect}) + "\n")
    print(f"wrote {len(CASES)} cases to {out}")