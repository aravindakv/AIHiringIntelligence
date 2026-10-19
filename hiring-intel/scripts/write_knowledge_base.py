"""Write the RAG knowledge base (markdown + front matter) into data/knowledge/.

All benchmark figures here are ILLUSTRATIVE, authored for this capstone. Replace them with your
organisation's real history before trusting any recommendation that cites them.
"""
from pathlib import Path

DOCS = {
    "bench_sourcing_backend": ("benchmark", ["sourcing"], "Backend",
        ("Sourcing benchmark (illustrative) for Backend roles. Referral-to-hire conversion is typically 8-12%; "
        "job boards convert at 2-4%. Referral candidates reach onsite about twice as often. Agency fees run ~15% "
        "of first-year CTC, so agencies should be reserved for niche or senior searches.")),
    "bench_sourcing_data": ("benchmark", ["sourcing", "rejection"], "Data",
        ("Sourcing benchmark (illustrative) for Data roles. Agency candidates historically fail SQL and statistics "
        "fundamentals more often than referrals. Require a take-home or SQL screen before agency candidates reach tech_1.")),
    "bench_sourcing_mobile": ("benchmark", ["sourcing"], "Mobile",
        ("Sourcing benchmark (illustrative) for Mobile roles. LinkedIn and referrals give the best native "
        "Android/iOS pipelines; careers-page applicants are often web developers applying to mobile postings.")),
    "bench_sourcing_sre": ("benchmark", ["sourcing"], "SRE",
        ("Sourcing benchmark (illustrative) for SRE roles. Expect 60-90 day time-to-fill. Referral and community "
        "channels (meetups, open source) outperform job boards for on-call experienced SREs.")),
    "bench_sourcing_all": ("benchmark", ["sourcing"], "all",
        ("General sourcing guidance (illustrative). Cut a channel when its hire rate is under half of the best channel "
        "for the same role over at least 50 applicants; re-test after one quarter. Shift recruiter hours, not just spend.")),
    "sla_policy": ("sla", ["pipeline", "panel_load"], "all",
        ("Stage SLA policy: applied 6 days, screen 6, tech_1 7, tech_2 7, onsite 7, offer 5. A stage is in breach when "
        "more than 30% of candidates exceed its SLA. Requisitions open over 45 days (60 for senior) need an EM review.")),
    "load_policy": ("policy", ["panel_load"], "all",
        ("Interviewer load policy: no more than 4 interviews per person per week, and no one above 2x the team median "
        "for two weeks running. Rebalance within the same team and level; pair new interviewers with shadows first.")),
    "rej_hist_q1": ("rejection_history", ["rejection"], "Mobile",
        ("Rejection history Q1 2026: Mobile screen rejection was 35%, mostly 'experience gap'. A JD rewrite in 2025 "
        "that listed must-have native skills cut screen rejections by a third within one quarter.")),
    "rej_hist_data": ("rejection_history", ["rejection"], "Data",
        ("Rejection history Q1 2026: Data tech_1 failures concentrated in candidates from two agencies; adding a SQL "
        "pre-screen reduced tech_1 failure from 62% to 38%.")),
    "rej_playbook": ("playbook", ["rejection"], "all",
        ("Diagnosis playbook: high screen rejection with 'JD mismatch' means fix the JD or sourcing filters; high "
        "technical rejection in one source means fix that source; high technical rejection across all sources means "
        "recalibrate the bar or the interviewers (run a calibration session with scored recordings).")),
    "offer_hist": ("offer_history", ["offers"], "all",
        ("Offer outcome history (illustrative): when the offer is below 90% of the candidate's expected CTC, acceptance "
        "falls to roughly one in four. Offers at or above 95% of expectation are accepted about 80% of the time.")),
    "comp_bands": ("offer_history", ["offers"], "all",
        ("Compensation band guidance: approve up to 10% above band midpoint for senior engineers with a competing offer. "
        "Offer approvals should complete within 48 hours; every extra week of delay lowers acceptance.")),
    "offer_timing": ("offer_history", ["offers", "pipeline"], "all",
        ("Offer timing: candidates who wait more than 7 days for an offer decision are twice as likely to decline for "
        "'timing' or a competing offer. Pre-approve bands before the onsite.")),
    "pipeline_ttf": ("benchmark", ["pipeline"], "all",
        ("Time-to-fill benchmark (illustrative): Backend/Frontend 35-45 days, Data 45 days, Mobile 40 days, SRE 60-90 days. "
        "Onsite scheduling is the most common bottleneck; block recurring onsite slots per week.")),
    "past_insight_panel": ("past_insight", ["panel_load"], "all",
        ("Past insight (accepted): two interviewers carried 35% of loops in Q1. Moving 3 slots/week each to trained "
        "shadows cut their load to policy within 3 weeks and reduced onsite scheduling delay.")),
    "past_insight_sourcing": ("past_insight", ["sourcing"], "Backend",
        "Past insight (accepted): Backend referral bonus raised in Q1; referral share rose from 12% to 20% of applicants."),
    "interview_calibration": ("playbook", ["rejection", "panel_load"], "all",
        ("Interviewer calibration: run a monthly calibration with 3 scored recordings per stage; flag interviewers whose "
        "pass rate deviates more than 20 points from the panel average.")),
}


def main() -> None:
    out = Path("data/knowledge")
    out.mkdir(parents=True, exist_ok=True)
    for name, (doc_type, agents, role, body) in DOCS.items():
        (out / f"{name}.md").write_text(
            f"---\ndoc_type: {doc_type}\nagents: [{', '.join(agents)}]\nrole: {role}\nperiod: 2026-Q1\n---\n"
            f"# {name.replace('_', ' ').title()}\n{body}\n")
    print(f"wrote {len(DOCS)} docs to {out}")


if __name__ == "__main__":
    main()