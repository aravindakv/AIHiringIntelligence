# 09 — The five insight agents

**Goal:** five agents, one job each, all returning the shared contract.

**Time:** 1–2 hours (read each `analyze()` carefully; this is where the hiring logic lives).

## Concepts

| Agent | Question | How it decides | Typical tier |
| --- | --- | --- | --- |
| Sourcing Quality | Which channels to grow or cut, per role? | onsite rate per role × source; two-proportion z-test (z ≥ 2.5) and a 2× gap; channel-level check for an over-used weak channel | small |
| Rejection Pattern | Where and why do rejections happen; JD, sourcing or interviewing? | reasons ≥ 30% of a stage; screen + "JD mismatch" → `[jd]`; one source failing tech_1 at ≤ 0.65× its peers → `[sourcing]` | large |
| Panel Load Balancer | Who is overloaded or idle; how to rebalance? | per-week load vs max(policy, 2× panel median); produces a concrete `rebalance_plan` | small |
| Offer Insights | Why are offers declined? | decline rate by CTC-ratio band; ≥ 25-point gap below 90% → comp floor | large (high stakes) |
| Pipeline Health | Where are SLA breaches and aging reqs? | stage × role breach rate > 30%; reqs open beyond the limit | small |

Design points:

- **Onsite rate, not hire rate, for sourcing.** Hire rate is diluted by every later stage and by offer outcomes, so it carries too little signal to compare channels.
- **Statistical gates.** A finding needs both a large effect and enough data. Even so, chapter 15 shows false positives on the small inputs, a known limitation.
- **"No action" is a valid answer.** On the healthy input each agent returns one low-severity insight saying so, with the numbers that show it. These still name an owner and a re-check date: the DeepEval suite in chapter 17 caught the earlier version that didn't.
- **Alternatives are honest.** Most agents offer "rules only (no LLM)" as the cheaper option, because the rules really produce the same numbers without the narrative.

## Step 9.1 — The agents

**File:** `hiring_intel/agents/insight_agents.py`
```python
"""The five insight agents. Each supplies: features(), rag_query, a prompt file, and analyze()
(the rule-based analyst used by the offline mock backend and as the 'rules only' alternative)."""
from __future__ import annotations

import math

import pandas as pd

from hiring_intel.agents.base import SLA, BaseInsightAgent, conf_from, ev, rag_ev
from hiring_intel.contracts import Alternative, Insight, LLMInsights, Move
from hiring_intel.metrics import (offer_features, panel_load_features, pipeline_features, rejection_features,
                                  sourcing_features)

RULES_ALT = Alternative(option="Rules only (no LLM)",
                        trade_off="Same thresholds and numbers at $0 and ~0 ms, but no narrative, no weighing of "
                                  "competing explanations and no benchmark comparison.", est_usd=0.0)


def _no_action(title: str, text: str, evidence) -> Insight:
    # found by the DeepEval suite: "no action" findings must still name an owner and a re-check date
    return Insight(title=title, recommendation=f"Recruiting ops lead: {text}", evidence=evidence, confidence_score=0.55,
                   alternative=RULES_ALT, severity="low")


def _z(p1: float, n1: int, p2: float, n2: int) -> float:
    p = (p1 * n1 + p2 * n2) / max(1, n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-9))
    return (p1 - p2) / se


# ---------------------------------------------------------------- Sourcing Quality
class SourcingQualityAgent(BaseInsightAgent):
    name = "sourcing"
    rag_query = "source conversion benchmark referral job board agency hire rate"

    def features(self, data, compress=False):
        return sourcing_features(data, compress)

    def analyze(self, g: pd.DataFrame, chunks) -> LLMInsights:
        """Quality = onsite_rate (share of applicants reaching onsite): more signal than hire rate,
        which is diluted by every later stage and by offer outcomes."""
        found, out = [], []
        g = g[g.applicants >= 20]
        for role, grp in g.groupby("role"):
            if len(grp) < 2:
                continue
            best, worst = grp.loc[grp.onsite_rate.idxmax()], grp.loc[grp.onsite_rate.idxmin()]
            z = _z(best.onsite_rate, best.applicants, worst.onsite_rate, worst.applicants)
            if z >= 2.5 and best.onsite_rate >= 2 * max(worst.onsite_rate, 0.005):
                found.append((z, role, best, worst))
        # channel-level check across roles (catches an agency-heavy mix)
        tot = g.assign(o=g.onsite_rate * g.applicants).groupby("source").agg(a=("applicants", "sum"), o=("o", "sum"))
        tot["rate"] = tot.o / tot.a
        share = tot.a / tot.a.sum()
        for src, row in tot.iterrows():
            rest = tot.drop(src)
            rest_rate = rest.o.sum() / rest.a.sum()
            z = _z(rest_rate, int(rest.a.sum()), row.rate, int(row.a))
            if share[src] >= 0.25 and z >= 3:
                rows = g[g.source == src].sort_values("applicants", ascending=False).head(2)
                out.append(Insight(
                    title=f"Deprioritise {src}: {share[src]:.0%} of applicants, weakest pipeline",
                    recommendation=(f"Talent lead: cap {src} at 15% of new applicants from next sprint and move the "
                                    f"freed recruiter hours to referrals. Only {row.rate:.1%} of {src} applicants "
                                    f"reach onsite vs {rest_rate:.1%} from all other channels."),
                    evidence=[ev("onsite_rate", float(r.onsite_rate), role=r.role, source=src) for r in rows.itertuples()]
                    + [ev("applicants", int(rows.iloc[0].applicants), role=rows.iloc[0].role, source=src)]
                    + rag_ev(chunks, "all"),
                    confidence_score=conf_from(int(row.a), z / 3), alternative=RULES_ALT, severity="high"))
        for z, role, best, worst in sorted(found, key=lambda t: -t[0])[:2]:
            out.append(Insight(
                title=f"{role}: {best.source} out-converts {worst.source}",
                recommendation=(f"Recruiting lead for {role}: shift 20% of {worst.source} effort to {best.source} "
                                f"this quarter. {best.onsite_rate:.1%} of {best.source} applicants reach onsite vs "
                                f"{worst.onsite_rate:.1%} from {worst.source}; re-check after 50 new applicants."),
                evidence=[ev("onsite_rate", float(best.onsite_rate), role=role, source=best.source),
                          ev("onsite_rate", float(worst.onsite_rate), role=role, source=worst.source),
                          ev("hire_rate", float(best.hire_rate), role=role, source=best.source),
                          ev("applicants", int(best.applicants), role=role, source=best.source)]
                + rag_ev(chunks, role.lower()),
                confidence_score=conf_from(int(best.applicants + worst.applicants), z / 3),
                alternative=RULES_ALT, severity="high" if z >= 4 else "medium"))
        if not out:
            top = g.sort_values("onsite_rate", ascending=False).iloc[0]
            out.append(_no_action("No sourcing change needed",
                                  "No sourcing change this cycle: no channel differs from the best channel for its role "
                                  "by a statistically clear margin. Re-run after 30 more days of applicants.",
                                  [ev("onsite_rate", float(top.onsite_rate), role=top.role, source=top.source)]))
        return LLMInsights(insights=out[:3])


# ---------------------------------------------------------------- Rejection Pattern
class RejectionPatternAgent(BaseInsightAgent):
    name = "rejection"
    rag_query = "rejection history stage reason JD mismatch fundamentals diagnosis playbook"
    causal = True

    def features(self, data, compress=False):
        return rejection_features(data, compress)

    def analyze(self, f: dict, chunks) -> LLMInsights:
        by, t1 = f["by_reason"], f["tech1_by_source"]
        out, seen = [], set()
        for r in by[by.share_of_stage >= 0.3].itertuples():
            if (r.role, r.stage) in seen:
                continue
            seen.add((r.role, r.stage))
            if r.stage == "screen" and r.reason in ("JD mismatch", "experience gap"):
                cause, action = "jd", (f"Hiring manager for {r.role}: rewrite the JD's must-have list and add a "
                                       f"knock-out question to the application form within 2 weeks")
            elif r.stage == "screen":
                cause, action = "sourcing", f"Recruiter for {r.role}: tighten sourcing filters for the screen stage"
            else:
                cause, action = "interview_calibration", (f"Panel lead for {r.role}: run a calibration session on "
                                                          f"{r.stage} with 3 scored recordings this month")
            out.append(Insight(
                title=f"[{cause}] {r.role} {r.stage}: '{r.reason}' drives rejections",
                recommendation=(f"{action}. '{r.reason}' explains {r.share_of_stage:.0%} of everyone entering "
                                f"{r.stage} for {r.role}."),
                evidence=[ev("share_of_stage", float(r.share_of_stage), role=r.role, stage=r.stage, reason=r.reason),
                          ev("entered", int(r.entered), role=r.role, stage=r.stage)] + rag_ev(chunks, "rej"),
                confidence_score=conf_from(int(r.entered), r.share_of_stage * 3), alternative=RULES_ALT,
                severity="high" if r.share_of_stage >= 0.5 else "medium"))
            if len(out) >= 2:
                break
        flagged = []
        for role, grp in t1[t1.entered >= 15].groupby("role"):
            worst = grp.loc[grp.pass_rate.idxmin()]
            others = grp.drop(worst.name)
            if len(others) and worst.pass_rate <= 0.65 * others.pass_rate.median():
                flagged.append((role, worst, float(others.pass_rate.median())))
        if flagged:
            srcs = {w.source for _, w, _ in flagged}
            src = max(srcs, key=lambda s: sum(1 for _, w, _ in flagged if w.source == s))
            rows = [(r, w, m) for r, w, m in flagged if w.source == src][:3]
            roles = ", ".join(r for r, _, _ in rows)
            out.append(Insight(
                title=f"[sourcing] {src} candidates fail tech_1 ({roles})",
                recommendation=(f"Talent lead: add a 20-minute fundamentals pre-screen for {src} candidates in "
                                f"{roles} before tech_1, starting next week, and share the failure data with the "
                                f"{src} partners. This is a sourcing problem, not an interviewer problem."),
                evidence=[ev("tech1_pass_rate", float(w.pass_rate), role=r, source=src) for r, w, _ in rows]
                + [ev("tech1_entered", int(rows[0][1].entered), role=rows[0][0], source=src)]
                + rag_ev(chunks, "data"),
                confidence_score=conf_from(int(sum(w.entered for _, w, _ in rows)), 1.5), alternative=RULES_ALT,
                severity="high" if len(rows) >= 2 else "medium"))
        if not out:
            top = by.iloc[0]
            out.append(_no_action("No dominant rejection pattern",
                                  "No action this cycle: no single reason explains 30% or more of a stage, and no "
                                  "source fails tech_1 far below its peers. Re-check in 30 days.",
                                  [ev("share_of_stage", float(top.share_of_stage), role=top.role, stage=top.stage)]))
        return LLMInsights(insights=out[:3])


# ---------------------------------------------------------------- Panel Load Balancer
class PanelLoadAgent(BaseInsightAgent):
    name = "panel_load"
    rag_query = "interviewer load policy interviews per week rebalance shadows"

    def features(self, data, compress=False):
        return panel_load_features(data, compress=compress, cap=SLA["max_interviews_per_week"])

    def analyze(self, load: pd.DataFrame, chunks) -> LLMInsights:
        cap = SLA["max_interviews_per_week"]
        med = float(load.panel_median_per_week.iloc[0]) if len(load) else 0
        hot = load[load.per_week > max(cap, 2 * med)]
        cold = load[load.per_week < 0.4 * med].sort_values("per_week")
        if hot.empty:
            top = load.iloc[0]
            return LLMInsights(insights=[_no_action(
                "Interviewer load within policy",
                f"No rebalance needed: the busiest interviewer runs {top.per_week} interviews a week, inside the "
                f"policy limit and under 2x the panel median. Re-check in 14 days.",
                [ev("per_week", float(top.per_week), interviewer=top.interviewer_id)])])
        plan, receivers = [], list(cold.itertuples()) or list(load.sort_values("per_week").head(5).itertuples())
        j = 0
        for h in hot.itertuples():
            excess = round(h.per_week - max(med, cap / 2), 1)
            n = max(1, min(3, len(receivers)))
            for _ in range(n):
                rcv = receivers[j % len(receivers)]
                j += 1
                plan.append(Move(from_interviewer=h.interviewer_id, to_interviewer=rcv.interviewer_id,
                                 slots_per_week=round(excess / n, 1)))
        top = hot.head(3)
        total = round(sum(m.slots_per_week for m in plan), 1)
        ins = Insight(
            title=f"{len(hot)} interviewers over load policy",
            recommendation=(f"Recruiting coordinator: from next Monday move {total} interview slots a week from "
                            f"{len(hot)} overloaded interviewers to {len(set(m.to_interviewer for m in plan))} "
                            f"under-used ones (plan attached). Busiest runs {top.iloc[0].per_week}/week vs policy {cap}."),
            evidence=[ev("per_week", float(r.per_week), interviewer=r.interviewer_id) for r in top.itertuples()]
            + [ev("share", float(top.iloc[0].share), interviewer=top.iloc[0].interviewer_id)] + rag_ev(chunks, "load"),
            confidence_score=conf_from(int(len(load) * 10), float(top.iloc[0].per_week) / max(med, 1)),
            alternative=Alternative(option="Pure pandas rebalance (no LLM)",
                                    trade_off="The move list is already computed by rules; the LLM only explains it. "
                                              "Skipping it saves the call but loses the rationale text.", est_usd=0.0),
            severity="high" if len(hot) >= 3 or float(top.iloc[0].share) >= 0.15 else "medium",
            rebalance_plan=plan[:12])
        return LLMInsights(insights=[ins])


# ---------------------------------------------------------------- Offer Insights
class OfferInsightsAgent(BaseInsightAgent):
    name = "offers"
    rag_query = "offer decline compensation band expected CTC acceptance timing competing offer"
    causal = True
    stakes = "high"

    def features(self, data, compress=False):
        return offer_features(data, compress)

    def analyze(self, f: dict, chunks) -> LLMInsights:
        band, reasons = f["by_ratio_band"].set_index("ratio_band"), f["decline_reasons"]
        out = []
        if "<0.9" in band.index:
            low = band.loc["<0.9"]
            ref_key = "0.9-1.0" if "0.9-1.0" in band.index else (">1.0" if ">1.0" in band.index else None)
            if ref_key is not None:
                ref = band.loc[ref_key]
                gap = float(low.decline_rate - ref.decline_rate)
                if gap >= 0.25 and low.offers >= 8:
                    comp = reasons[reasons.reason == "compensation"]
                    share_low = low.offers / band.offers.sum()
                    out.append(Insight(
                        title="Offers below 90% of expected CTC are being declined",
                        recommendation=(f"Comp partner and hiring managers: set an offer floor of 90% of the "
                                        f"candidate's expected CTC, with exceptions approved within 48 hours. Offers "
                                        f"under that line decline at {low.decline_rate:.0%} vs {ref.decline_rate:.0%} "
                                        f"in the {ref_key} band."),
                        evidence=[ev("decline_rate", float(low.decline_rate), ratio_band="<0.9"),
                                  ev("decline_rate", float(ref.decline_rate), ratio_band=ref_key),
                                  ev("offers", int(low.offers), ratio_band="<0.9")]
                        + ([ev("share_of_declines", float(comp.iloc[0].share_of_declines), reason="compensation")]
                           if len(comp) else []) + rag_ev(chunks, "offer_hist"),
                        confidence_score=conf_from(int(band.offers.sum()), gap * 3),
                        alternative=Alternative(option="Small model + rules",
                                                trade_off="Band math is deterministic; a small model can phrase it. "
                                                          "Large model adds value only when causes compete.",
                                                est_usd=None),
                        severity="high" if share_low >= 0.4 or gap >= 0.4 else "medium"))
        top = reasons.iloc[0] if len(reasons) else None
        if top is not None and top.reason != "compensation" and top.share_of_declines >= 0.45:
            out.append(Insight(
                title=f"'{top.reason}' is the main decline reason",
                recommendation=(f"Recruiting lead: pre-approve comp bands before onsite and send offers within 48 "
                                f"hours to cut '{top.reason}' declines ({top.share_of_declines:.0%} of all declines)."),
                evidence=[ev("share_of_declines", float(top.share_of_declines), reason=top.reason),
                          ev("declines", int(top.declines), reason=top.reason)] + rag_ev(chunks, "timing"),
                confidence_score=conf_from(int(reasons.declines.sum()), 1.0), alternative=RULES_ALT))
        if not out:
            row = band.reset_index().iloc[0]
            out.append(_no_action("Offer acceptance healthy",
                                  "No offer policy change needed: decline rates do not differ materially across CTC "
                                  "ratio bands and no single non-comp reason dominates. Re-check in 90 days.",
                                  [ev("decline_rate", float(row.decline_rate), ratio_band=row.ratio_band)]))
        return LLMInsights(insights=out[:3])


# ---------------------------------------------------------------- Pipeline Health
class PipelineHealthAgent(BaseInsightAgent):
    name = "pipeline"
    rag_query = "stage SLA targets time to fill onsite scheduling bottleneck"

    def features(self, data, compress=False):
        return pipeline_features(data, SLA, compress)

    def analyze(self, f: dict, chunks) -> LLMInsights:
        br, aging = f["sla_breaches"], f["aging_reqs_by_role"]
        out = []
        for r in br.head(2).itertuples():
            out.append(Insight(
                title=f"{r.role} {r.stage} breaches SLA",
                recommendation=(f"Hiring manager for {r.role}: block recurring {r.stage} slots (2 per week) and "
                                f"name a scheduling owner by Friday; {r.breach_rate:.0%} of candidates exceed the "
                                f"{SLA['stage_sla_days'][r.stage]}-day SLA, median {r.median_dwell} days."),
                evidence=[ev("breach_rate", float(r.breach_rate), role=r.role, stage=r.stage),
                          ev("median_dwell", float(r.median_dwell), role=r.role, stage=r.stage)]
                + rag_ev(chunks, "sla"),
                confidence_score=conf_from(int(r.entered), r.breach_rate * 3), alternative=RULES_ALT,
                severity="high" if r.breach_rate >= 0.6 else "medium"))
        if len(aging):
            a = aging.sort_values("aging_reqs", ascending=False).iloc[0]
            out.append(Insight(
                title=f"{int(a.aging_reqs)} {a.role} requisitions open too long",
                recommendation=(f"EM for {a.role}: review the {int(a.aging_reqs)} reqs open beyond "
                                f"{SLA['max_days_open']['default']} days (oldest {int(a.max_days_open)} days) and "
                                f"re-scope, re-level or close each by month end."),
                evidence=[ev("aging_reqs", int(a.aging_reqs), role=a.role),
                          ev("max_days_open", int(a.max_days_open), role=a.role)] + rag_ev(chunks, "ttf"),
                confidence_score=0.8, alternative=RULES_ALT,
                severity="high" if a.max_days_open > 120 else "medium"))
        if not out:
            fun = f["funnel"].iloc[0]
            out.append(_no_action("Pipeline within SLA",
                                  "No pipeline action needed: no stage breaches its SLA for more than 30% of "
                                  "candidates and no requisition is past the open-days limit. Re-check in 14 days.",
                                  [ev("candidates", int(fun.candidates), stage=fun.stage)]))
        return LLMInsights(insights=out[:3])


AGENT_CLASSES = [SourcingQualityAgent, RejectionPatternAgent, PanelLoadAgent, OfferInsightsAgent, PipelineHealthAgent]
AGENTS = {c.name: c() for c in AGENT_CLASSES}
```

## Verify

<!-- verify:09 -->
```bash
python - <<'EOF'
from hiring_intel.ingest import load_all
from hiring_intel.enrich import enrich
from hiring_intel.openjev import decider
from hiring_intel.agents.insight_agents import AGENTS
for scenario in ["baseline", "healthy"]:
    d = load_all(f"data/raw/{scenario}_medium")
    enrich(d, "ch09", decider())
    print(f"== {scenario}")
    for name, agent in AGENTS.items():
        out = agent.run(d, "ch09")
        print(f"  {name:10} {out.status:8} " + " | ".join(f"[{i.severity}] {i.title}" for i in out.insights))
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
== baseline
  sourcing   ok       [high] Backend: referral out-converts job_board | [medium] Data: referral out-converts agency
  rejection  ok       [high] [jd] Mobile screen: 'JD mismatch' drives rejections | [medium] [sourcing] agency candidates fail tech_1 (Data)
  panel_load ok       [high] 10 interviewers over load policy
  offers     ok       [high] Offers below 90% of expected CTC are being declined
  pipeline   ok       [medium] SRE onsite breaches SLA | [high] 10 SRE requisitions open too long
== healthy
  sourcing   ok       [low] No sourcing change needed
  rejection  ok       [low] No dominant rejection pattern
  panel_load ok       [low] Interviewer load within policy
  offers     ok       [low] Offer acceptance healthy
  pipeline   ok       [low] Pipeline within SLA
```

Baseline surfaces the planted patterns; the healthy control returns low-severity "no action" findings.

## Checkpoint

- [ ] all five agents return `ok` (or `degraded` if no RAG context)
- [ ] the baseline findings match the planted patterns from chapter 02
