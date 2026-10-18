"""Generate mock ATS data with planted, known patterns.

Usage:
    python scripts/generate_mock_data.py --scenario baseline --size large
    python scripts/generate_mock_data.py --all          # every scenario x size

Sizes (approx.):  small ~1.3k candidates | medium ~6k | large ~25k | xlarge ~85k
Each scenario plants different patterns, so the same agents get different inputs.
The planted truth is written to <out>/_truth.json and is used by the golden-set eval.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ["Backend", "Mobile", "Data", "SRE", "Frontend"]
SOURCES = ["referral", "linkedin", "job_board", "agency", "careers_page"]
STAGES = ["applied", "screen", "tech_1", "tech_2", "onsite", "offer"]
INTERVIEW_STAGES = ["screen", "tech_1", "tech_2", "onsite"]
START = datetime(2026, 4, 1)
SIZES = {"small": 30, "medium": 150, "large": 600, "xlarge": 2000}

FIRST = ["Asha", "Ravi", "Meera", "Arjun", "Divya", "Kiran", "Neha", "Vikram", "Priya", "Rahul",
         "Sneha", "Aditya", "Kavya", "Rohan", "Isha", "Nikhil", "Pooja", "Sanjay", "Tara", "Varun",
         "Lena", "Omar", "Mia", "Jon", "Sara", "Ken", "Ana", "Leo", "Zoe", "Ivan"]
LAST = ["Rao", "Iyer", "Shah", "Menon", "Nair", "Gupta", "Reddy", "Das", "Bose", "Kulkarni"]

# Free-text notes an interviewer would type. The structured reason is often missing in
# real ATS exports, so the system must classify these (an OpenJev job).
REJECTION_NOTES = {
    "JD mismatch": ["Profile does not match the JD; candidate is mostly web, role needs native Android.",
                    "Expected Kotlin/Swift depth per job description, background is backend.",
                    "Applied for a different scope than the posting describes."],
    "fundamentals": ["Struggled with basic data structures and complexity analysis.",
                     "Could not explain joins or indexing; SQL fundamentals weak.",
                     "Weak on core CS basics, recursion and hashing."],
    "system design": ["Design was shallow, no thought on scaling or failure modes.",
                      "Could not partition the service or reason about consistency."],
    "communication": ["Hard to follow explanations, did not clarify requirements.",
                      "Rambled, could not structure the answer or listen to hints."],
    "coding": ["Did not finish the coding task, many bugs and no tests.",
               "Code was messy and incorrect on edge cases."],
    "experience gap": ["Too junior for the level; limited ownership of production systems.",
                       "Years of experience far below what the level needs."],
    "culture add": ["Low collaboration signals, dismissive of team feedback."],
}
# notes with no clear signal (a share of real notes look like this)
AMBIGUOUS_NOTES = ["Not the right fit for the team at this time.", "Mixed signals across the loop; not moving ahead.",
                   "Panel was not convinced overall."]
DECLINE_NOTES = {
    "compensation": ["Candidate said the offer is well below their expectation and current CTC.",
                     "Declined: compensation gap too large, asked for 20% more."],
    "competing offer": ["Accepted another company's offer that came in first.",
                        "Went with a competing offer from a product company."],
    "timing": ["Process took too long, candidate had moved on.",
               "Notice period and start date could not be aligned."],
    "location": ["Not willing to relocate to Bengaluru; wanted remote."],
}


@dataclass
class Scenario:
    name: str
    description: str
    referral_backend_boost: float = 0.25
    job_board_penalty: float = 0.15
    agency_data_tech1_pass: float | None = 0.30
    agency_all_tech1_pass: float | None = None
    mobile_screen_pass: float | None = 0.20
    source_weights: list[float] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    n_interviewers: int = 20
    hot_interviewers: int = 2
    hot_weight: float = 12.0
    cold_interviewers: int = 5
    comp_ratio_range: tuple[float, float] = (0.80, 1.10)
    sre_onsite_sla_mult: float = 2.5
    all_tech_sla_mult: float = 1.0
    aging_sre_reqs: bool = True
    structured_reason_rate: float = 0.4   # share of rejections with a structured reason field


SCENARIOS = {
    "baseline": Scenario("baseline", "All six planted patterns at moderate strength."),
    "healthy": Scenario("healthy", "Control: no strong pattern. Agents should report low confidence / no action.",
                        referral_backend_boost=0.0, job_board_penalty=0.0, agency_data_tech1_pass=None,
                        mobile_screen_pass=None, hot_interviewers=0, cold_interviewers=0,
                        comp_ratio_range=(0.95, 1.15), sre_onsite_sla_mult=1.0, aging_sre_reqs=False),
    "comp_crunch": Scenario("comp_crunch", "Offers lag market: most offers below 90% of expected CTC.",
                            comp_ratio_range=(0.72, 1.00), mobile_screen_pass=None),
    "interviewer_burnout": Scenario("interviewer_burnout", "Small, skewed panel: 3 people carry most interviews; tech stages slow.",
                                    n_interviewers=12, hot_interviewers=3, hot_weight=20.0, cold_interviewers=4,
                                    all_tech_sla_mult=1.8, mobile_screen_pass=None),
    "sourcing_shift": Scenario("sourcing_shift", "Agency-heavy sourcing; agency candidates fail tech_1 across all roles.",
                               source_weights=[0.3, 1, 1, 3, 0.7], agency_all_tech1_pass=0.35,
                               agency_data_tech1_pass=None, referral_backend_boost=0.25),
}


def pass_prob(sc: Scenario, role: str, source: str, stage: str) -> float:
    p = 0.6
    if source == "referral" and role == "Backend":
        p += sc.referral_backend_boost
    if source == "job_board" and stage == "screen":
        p -= sc.job_board_penalty
    if sc.agency_data_tech1_pass is not None and source == "agency" and role == "Data" and stage == "tech_1":
        p = sc.agency_data_tech1_pass
    if sc.agency_all_tech1_pass is not None and source == "agency" and stage == "tech_1":
        p = sc.agency_all_tech1_pass
    if sc.mobile_screen_pass is not None and role == "Mobile" and stage == "screen":
        p = sc.mobile_screen_pass
    return float(max(0.05, min(p, 0.95)))


def true_reason(sc: Scenario, role: str, source: str, stage: str, rnd: random.Random) -> str:
    if sc.mobile_screen_pass is not None and role == "Mobile" and stage == "screen":
        return "JD mismatch" if rnd.random() < 0.85 else "experience gap"
    if source == "agency" and stage == "tech_1" and (
            (sc.agency_data_tech1_pass is not None and role == "Data") or sc.agency_all_tech1_pass is not None):
        return "fundamentals" if rnd.random() < 0.8 else "coding"
    pool = ["communication", "system design", "culture add", "coding", "experience gap", "fundamentals"]
    if stage == "screen":
        pool = ["communication", "experience gap", "JD mismatch", "culture add"]
    return rnd.choice(pool)


def generate(scenario: str = "baseline", size: str = "small", seed: int = 42, out: str | None = None) -> Path:
    sc = SCENARIOS[scenario]
    n_reqs = SIZES[size]
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    out_dir = Path(out or f"data/raw/{scenario}_{size}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # interviewer pool scales with hiring volume so per-person load stays realistic
    scale = max(1, n_reqs // SIZES["small"])
    n_iv, n_hot, n_cold = sc.n_interviewers * scale, sc.hot_interviewers * scale, sc.cold_interviewers * scale
    interviewers = pd.DataFrame([{
        "interviewer_id": f"I{i:04d}", "name": f"{rnd.choice(FIRST)} {rnd.choice(LAST)}",
        "team": ROLES[i % len(ROLES)], "level": rnd.choice(["SE2", "SSE", "Staff"])}
        for i in range(n_iv)])
    w = np.ones(n_iv)
    w[:n_hot] = sc.hot_weight
    if n_cold:
        w[-n_cold:] = 0.1
    w = w / w.sum()
    iv_ids = interviewers.interviewer_id.to_numpy()
    src_p = np.array(sc.source_weights, dtype=float)
    src_p /= src_p.sum()

    reqs, cands, events, interviews, offers, labels = [], [], [], [], [], []
    for r in range(n_reqs):
        role = ROLES[r % len(ROLES)]
        level = rnd.choice(["mid", "senior"])
        opened = START + timedelta(days=int(rng.integers(0, 120)))
        if sc.aging_sre_reqs and role == "SRE" and r % 3 == 0:
            opened = START - timedelta(days=40)
        req_id = f"R{r:04d}"
        hired_any = False
        n_c = int(rng.integers(25, 60))
        for c in range(n_c):
            cid = f"C{r:04d}{c:03d}"
            source = SOURCES[int(rng.choice(len(SOURCES), p=src_p))]
            t = opened + timedelta(days=int(rng.integers(0, 30)))
            applied_at = t
            outcome, reached = "in_progress", "applied"
            for stage in STAGES:
                mult = sc.sre_onsite_sla_mult if (role == "SRE" and stage == "onsite") else 1.0
                if stage in ("tech_1", "tech_2", "onsite"):
                    mult *= sc.all_tech_sla_mult
                dwell = max(1, int(rng.gamma(2.0, 2.0) * mult))
                passed = rng.random() < pass_prob(sc, role, source, stage)
                reason = note = None
                if not passed:
                    reason_true = true_reason(sc, role, source, stage, rnd)
                    note = (rnd.choice(AMBIGUOUS_NOTES) if rnd.random() < 0.08
                            else rnd.choice(REJECTION_NOTES[reason_true]))
                    reason = reason_true if rnd.random() < sc.structured_reason_rate else None
                    labels.append({"kind": "rejection", "candidate_id": cid, "stage": stage, "true_reason": reason_true})
                events.append({"candidate_id": cid, "stage": stage, "entered_at": t,
                               "exited_at": t + timedelta(days=dwell), "result": "pass" if passed else "reject",
                               "rejection_reason": reason, "rejection_note": note})
                if stage in INTERVIEW_STAGES:
                    interviews.append({"interview_id": f"{cid}-{stage}", "candidate_id": cid,
                                       "interviewer_id": iv_ids[int(rng.choice(len(iv_ids), p=w))], "stage": stage,
                                       "scheduled_at": t + timedelta(days=max(0, dwell - 1)),
                                       "score": int(rng.integers(1, 5))})
                t += timedelta(days=dwell)
                reached = stage
                if not passed:
                    outcome = "rejected"
                    break
            if outcome != "rejected" and reached == "offer":
                expected = int(rng.integers(25, 60)) * 100_000
                lo, hi = sc.comp_ratio_range
                offered = expected * rng.uniform(lo, hi)
                low = offered < 0.9 * expected
                accepted = rng.random() < (0.25 if low else 0.8)
                if accepted:
                    dr, dnote = None, None
                else:
                    dr = "compensation" if low else rnd.choice(["competing offer", "timing", "location"])
                    dnote = rnd.choice(DECLINE_NOTES[dr])
                    labels.append({"kind": "decline", "candidate_id": cid, "stage": "offer", "true_reason": dr})
                offers.append({"candidate_id": cid, "req_id": req_id, "extended_at": t,
                               "responded_at": t + timedelta(days=int(rng.integers(1, 10))),
                               "status": "accepted" if accepted else "declined",
                               "decline_reason": dr if (dr and rnd.random() < 0.5) else None,
                               "decline_note": dnote, "offered_ctc": round(offered), "expected_ctc": expected})
                outcome = "hired" if accepted else "offer_declined"
                hired_any = hired_any or accepted
            cands.append({"candidate_id": cid, "req_id": req_id, "source": source, "applied_at": applied_at,
                          "current_stage": reached, "outcome": outcome})
        planted_aging = sc.aging_sre_reqs and role == "SRE" and r % 3 == 0
        reqs.append({"req_id": req_id, "role": role, "level": level, "team": role, "opened_at": opened,
                     "closed_at": None, "_hired": hired_any, "_aging": planted_aging})

    # requisition status relative to the data's "today" (last event): planted aging reqs stay open,
    # filled reqs close, recent unfilled reqs are open, older unfilled reqs were cancelled
    today = max(e["exited_at"] for e in events)
    for q in reqs:
        age = (today - q["opened_at"]).days
        aging, hired = q.pop("_aging"), q.pop("_hired")
        q["status"] = "open" if aging else "filled" if hired else "open" if age <= 40 else "closed"
    tables = {"requisitions": reqs, "candidates": cands, "stage_events": events,
              "interviews": interviews, "offers": offers}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    interviewers.to_csv(out_dir / "interviewers.csv", index=False)
    pd.DataFrame(labels).to_csv(out_dir / "_labels.csv", index=False)   # hidden truth, eval only
    truth = {"scenario": asdict(sc), "size": size, "counts": {k: len(v) for k, v in tables.items()}}
    (out_dir / "_truth.json").write_text(json.dumps(truth, indent=2, default=str))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline", choices=list(SCENARIOS))
    ap.add_argument("--size", default="small", choices=list(SIZES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="generate every scenario x size (xlarge only for baseline)")
    a = ap.parse_args()
    if a.all:
        for s in SCENARIOS:
            for z in ["small", "medium", "large"]:
                p = generate(s, z, a.seed)
                print(p, json.loads((p / "_truth.json").read_text())["counts"])
        p = generate("baseline", "xlarge", a.seed)
        print(p, json.loads((p / "_truth.json").read_text())["counts"])
    else:
        p = generate(a.scenario, a.size, a.seed)
        print(p, json.loads((p / "_truth.json").read_text())["counts"])