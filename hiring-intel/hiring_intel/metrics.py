"""Deterministic features per agent. Agents never count rows themselves: pandas computes,
the LLM interprets. `compress=True` is optimization lever B (fewer rows/columns -> fewer tokens)."""
from __future__ import annotations

import pandas as pd

from hiring_intel.ingest import HiringData

STAGES = ["applied", "screen", "tech_1", "tech_2", "onsite", "offer"]

# Names exist in the CSVs for demos (dashboard candidate lookup). They must never reach an LLM prompt,
# a LangSmith trace or a DeepEval upload: agents reason over aggregates and IDs only.
PII_COLUMNS = {"name", "candidate_name", "interviewer_name", "hiring_manager", "email", "phone"}


def strip_pii(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c in PII_COLUMNS])


def _min_n(d: HiringData) -> int:
    return max(15, len(d.candidates) // 400)


def sourcing_features(d: HiringData, compress: bool = False) -> pd.DataFrame:
    c = d.candidates.merge(d.requisitions[["req_id", "role"]], on="req_id")
    g = c.groupby(["role", "source"]).agg(
        applicants=("candidate_id", "count"),
        hires=("outcome", lambda s: int((s == "hired").sum())),
        reached_onsite=("current_stage", lambda s: int(s.isin(["onsite", "offer"]).sum()))).reset_index()
    g["hire_rate"] = (g.hires / g.applicants).round(3)
    g["onsite_rate"] = (g.reached_onsite / g.applicants).round(3)
    if compress:
        g = g[["role", "source", "applicants", "hire_rate", "onsite_rate"]]
    return g


def rejection_features(d: HiringData, compress: bool = False) -> dict[str, pd.DataFrame]:
    reason_col = "reason" if "reason" in d.stage_events.columns else "rejection_reason"
    e = (d.stage_events.merge(d.candidates[["candidate_id", "req_id", "source"]], on="candidate_id")
         .merge(d.requisitions[["req_id", "role"]], on="req_id"))
    rej = e[e.result == "reject"]
    by = rej.groupby(["role", "stage", reason_col]).size().rename("rejections").reset_index()
    by = by.rename(columns={reason_col: "reason"})
    tot = e.groupby(["role", "stage"]).size().rename("entered").reset_index()
    out = by.merge(tot, on=["role", "stage"])
    out["share_of_stage"] = (out.rejections / out.entered).round(3)
    out = out[out.entered >= _min_n(d)].sort_values("share_of_stage", ascending=False)
    t1 = e[e.stage == "tech_1"].groupby(["role", "source"]).agg(
        entered=("result", "size"), pass_rate=("result", lambda s: round(float((s == "pass").mean()), 3))).reset_index()
    if compress:
        out = out.head(15)[["role", "stage", "reason", "entered", "share_of_stage"]]
        t1 = t1[t1.entered >= _min_n(d)]
    return {"by_reason": out, "tech1_by_source": t1}


def panel_load_features(d: HiringData, weeks: int = 8, compress: bool = False, cap: float = 4) -> pd.DataFrame:
    """Load over the busiest recent window: `weeks` weeks ending at the 90th percentile date
    (the export's last weeks are thin because candidates are still in process)."""
    i = d.interviews
    end = i.scheduled_at.quantile(0.9)
    i = i[(i.scheduled_at <= end) & (i.scheduled_at > end - pd.Timedelta(weeks=weeks))]
    load = i.groupby("interviewer_id").size().rename("interviews").reset_index()
    load = strip_pii(d.interviewers).merge(load, how="left", on="interviewer_id").fillna({"interviews": 0})
    load["interviews"] = load.interviews.astype(int)
    load["per_week"] = (load.interviews / weeks).round(1)
    load["share"] = (load.interviews / max(1, load.interviews.sum())).round(3)
    load = load.sort_values("interviews", ascending=False)
    # computed on ALL interviewers before any trimming. Found by evaluation:
    #  1. a median of only the top/bottom rows hid overload (golden set, chapter 15)
    #  2. keeping only the top 8 rows truncated the overloaded list ("8 over policy" instead of 10)
    med = float(load[load.per_week > 0].per_week.median())
    load["panel_median_per_week"] = med
    if compress:
        over = load[load.per_week > max(cap, 2 * med)]
        keep = pd.concat([over, load.head(8), load.tail(8)]).drop_duplicates("interviewer_id")
        load = keep[["interviewer_id", "team", "level", "per_week", "share", "panel_median_per_week"]]
    return load


def offer_features(d: HiringData, compress: bool = False) -> pd.DataFrame:
    reason_col = "reason" if "reason" in d.offers.columns else "decline_reason"
    o = d.offers.merge(d.requisitions[["req_id", "role", "level"]], on="req_id")
    o["ctc_ratio"] = (o.offered_ctc / o.expected_ctc).round(2)
    o["ratio_band"] = pd.cut(o.ctc_ratio, [0, 0.9, 1.0, 9], labels=["<0.9", "0.9-1.0", ">1.0"]).astype(str)
    g = o.groupby(["ratio_band", "status"]).size().unstack(fill_value=0).reset_index()
    for col in ("accepted", "declined"):
        if col not in g:
            g[col] = 0
    g["offers"] = g.accepted + g.declined
    g["decline_rate"] = (g.declined / g.offers).round(3)
    reasons = (o[o.status == "declined"].groupby(reason_col).size().rename("declines").reset_index()
               .rename(columns={reason_col: "reason"}))
    reasons["share_of_declines"] = (reasons.declines / max(1, reasons.declines.sum())).round(3)
    by_role = o.groupby("role").agg(offers=("status", "size"),
                                    decline_rate=("status", lambda s: round(float((s == "declined").mean()), 3)),
                                    median_ctc_ratio=("ctc_ratio", "median")).reset_index()
    g = g[["ratio_band", "offers", "declined", "decline_rate"]]
    g.attrs = {}
    out = {"by_ratio_band": g, "decline_reasons": reasons.sort_values("declines", ascending=False),
           "by_role": by_role}
    if not compress:
        out["offers_detail"] = o[["role", "level", "ctc_ratio", "status"]].head(200)
    return out


def pipeline_features(d: HiringData, sla: dict, compress: bool = False) -> dict:
    e = d.stage_events.copy()
    e["dwell"] = (e.exited_at - e.entered_at).dt.days
    e = e.merge(d.candidates[["candidate_id", "req_id"]], on="candidate_id").merge(
        d.requisitions[["req_id", "role"]], on="req_id")
    e["sla"] = e.stage.map(sla["stage_sla_days"])
    e["breach"] = e.dwell > e.sla
    breaches = (e.groupby(["role", "stage"]).agg(entered=("dwell", "size"), median_dwell=("dwell", "median"),
                                                  breach_rate=("breach", "mean")).reset_index())
    breaches["breach_rate"] = breaches.breach_rate.round(3)
    breaches = breaches[breaches.breach_rate > 0.3].sort_values("breach_rate", ascending=False)
    today = d.stage_events.exited_at.max()
    r = d.requisitions[d.requisitions.status == "open"].copy()
    r["days_open"] = (today - r.opened_at).dt.days
    aging = r[r.days_open > sla["max_days_open"]["default"]]
    aging_by_role = aging.groupby("role").agg(aging_reqs=("req_id", "size"),
                                              max_days_open=("days_open", "max")).reset_index()
    funnel = d.stage_events.groupby("stage").candidate_id.nunique().reindex(STAGES).fillna(0).astype(int)
    out = {"funnel": funnel.reset_index().rename(columns={"candidate_id": "candidates"}),
           "sla_breaches": breaches.head(10) if compress else breaches,
           "aging_reqs_by_role": aging_by_role}
    if not compress:
        out["aging_reqs"] = aging[["req_id", "role", "days_open"]].sort_values("days_open", ascending=False)
    return out


def to_text(feats) -> str:
    """The only way feature tables become prompt text: PII columns are dropped here as a last guard."""
    if isinstance(feats, pd.DataFrame):
        return strip_pii(feats).to_csv(index=False)
    return "\n".join(f"## {k}\n{strip_pii(v).to_csv(index=False) if hasattr(v, 'to_csv') else v}"
                     for k, v in feats.items())