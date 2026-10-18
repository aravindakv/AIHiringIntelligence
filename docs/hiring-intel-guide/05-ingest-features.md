# 05 — Ingest and deterministic features

**Goal:** load and validate the ATS export, then compute every number the agents will talk about, in pandas.

**Time:** 45 minutes.

## Concepts

- **Pandas computes, the LLM interprets.** No agent counts rows. Each agent gets a small feature table (tens of rows, not thousands) and only explains it. This one rule:
  - removes most hallucination risk, since every number in an insight can be checked against the table (chapter 10 does exactly that);
  - keeps prompts the same size whatever the input size, which is why run cost stays flat on 84k candidates (chapter 15).
- **Validate at the edge.** Ingest checks required columns, parses dates, drops duplicates and orphan events, and counts what it quarantined. A missing table is a hard failure with a clear message; bad rows are not.
- **Cache per path.** The graph state carries only the dataset path; the loaded tables live in a per-path cache. That keeps LangGraph checkpoints small, which human review needs (chapter 11).
- **`compress=True`** is optimization lever B: fewer rows and columns per table. Evaluation caught two bugs it caused (chapter 15), which is why the panel median is computed **before** trimming and every over-policy interviewer is kept.

| Feature function | Agent | Key columns |
| --- | --- | --- |
| `sourcing_features` | Sourcing Quality | role, source, applicants, hire_rate, onsite_rate |
| `rejection_features` | Rejection Pattern | by_reason: role, stage, reason, share_of_stage; tech1_by_source |
| `panel_load_features` | Panel Load | interviewer, per_week, share, panel_median_per_week |
| `offer_features` | Offer Insights | by CTC ratio band, decline reasons, by role |
| `pipeline_features` | Pipeline Health | funnel, SLA breaches, aging reqs |

## Step 5.1 — Ingest

`MockATSClient` exposes the same data as a tool interface; swap it for a real ATS client later.

**File:** `hiring_intel/ingest.py`
```python
"""Load, clean and validate the ATS export. Also exposed as a tool (MockATSClient)."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TABLES = {
    "requisitions": ["req_id", "role", "level", "opened_at", "status"],
    "candidates": ["candidate_id", "req_id", "source", "applied_at", "current_stage", "outcome"],
    "stage_events": ["candidate_id", "stage", "entered_at", "exited_at", "result"],
    "interviews": ["interview_id", "candidate_id", "interviewer_id", "stage", "scheduled_at"],
    "interviewers": ["interviewer_id", "name", "team", "level"],
    "offers": ["candidate_id", "req_id", "extended_at", "responded_at", "status", "offered_ctc", "expected_ctc"],
}
DATES = ["opened_at", "closed_at", "applied_at", "entered_at", "exited_at", "scheduled_at",
         "extended_at", "responded_at"]


@dataclass
class HiringData:
    path: str
    requisitions: pd.DataFrame
    candidates: pd.DataFrame
    stage_events: pd.DataFrame
    interviews: pd.DataFrame
    interviewers: pd.DataFrame
    offers: pd.DataFrame
    quarantined: int = 0
    meta: dict | None = None


_cache: dict[str, HiringData] = {}
_lock = threading.Lock()


def load_all(path: str = "data/raw/baseline_small") -> HiringData:
    """Cached per path: the graph state stores only the path, so checkpoints stay small."""
    with _lock:
        if path in _cache:
            return _cache[path]
        p = Path(path)
        frames, quarantined = {}, 0
        for name, required in TABLES.items():
            f = p / f"{name}.csv"
            if not f.exists():
                raise FileNotFoundError(f"missing table {f}")
            df = pd.read_csv(f)
            missing = set(required) - set(df.columns)
            if missing:
                raise ValueError(f"{name}: missing columns {missing}")
            for c in DATES:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
            before = len(df)
            df = df.drop_duplicates()
            df = df.dropna(subset=[c for c in required if c not in ("closed_at",)])
            bad = before - len(df)
            if bad:
                quarantine = p / "_quarantine"
                quarantine.mkdir(exist_ok=True)
                quarantined += bad
            frames[name] = df
        # referential integrity: drop events for unknown candidates
        known = set(frames["candidates"].candidate_id)
        n0 = len(frames["stage_events"])
        frames["stage_events"] = frames["stage_events"][frames["stage_events"].candidate_id.isin(known)]
        quarantined += n0 - len(frames["stage_events"])
        meta_f = p / "_truth.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else None
        data = HiringData(path=path, quarantined=quarantined, meta=meta, **frames)
        _cache[path] = data
        return data


class MockATSClient:
    """Tool interface over the CSV export. Swap for a real ATS API client later."""

    def __init__(self, path: str):
        self.data = load_all(path)

    def get_candidates(self, req_id: str | None = None) -> list[dict]:
        c = self.data.candidates
        return (c[c.req_id == req_id] if req_id else c).to_dict("records")

    def get_open_requisitions(self) -> list[dict]:
        r = self.data.requisitions
        return r[r.status == "open"].to_dict("records")
```

## Step 5.2 — Features

**File:** `hiring_intel/metrics.py`
```python
"""Deterministic features per agent. Agents never count rows themselves: pandas computes,
the LLM interprets. `compress=True` is optimization lever B (fewer rows/columns -> fewer tokens)."""
from __future__ import annotations

import pandas as pd

from hiring_intel.ingest import HiringData

STAGES = ["applied", "screen", "tech_1", "tech_2", "onsite", "offer"]


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
    load = d.interviewers.merge(load, how="left", on="interviewer_id").fillna({"interviews": 0})
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
    if isinstance(feats, pd.DataFrame):
        return feats.to_csv(index=False)
    return "\n".join(f"## {k}\n{v.to_csv(index=False) if hasattr(v, 'to_csv') else v}" for k, v in feats.items())
```

## Verify

<!-- verify:05 -->
```bash
python - <<'EOF'
from hiring_intel.ingest import load_all
from hiring_intel.metrics import sourcing_features, rejection_features
d = load_all("data/raw/baseline_small")
print(len(d.candidates), "candidates,", len(d.stage_events), "stage events, quarantined:", d.quarantined)
g = sourcing_features(d)
print(g[g.role == "Backend"][["source", "applicants", "onsite_rate"]].to_string(index=False))
r = rejection_features(d)["by_reason"]
print(r.head(3)[["role", "stage", "reason", "share_of_stage"]].to_string(index=False))
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
1179 candidates, 2703 stage events, quarantined: 0
      source  applicants  onsite_rate
      agency          58        0.121
careers_page          52        0.154
   job_board          66        0.121
    linkedin          53        0.132
    referral          59        0.559
  role  stage         reason  share_of_stage
Mobile screen    JD mismatch           0.213
   SRE screen experience gap           0.101
  Data onsite    culture add           0.091
```

Referral reaches onsite more than three times as often as any other channel for Backend, and "JD mismatch" at the Mobile screen tops the rejection table: two planted patterns, recovered by plain pandas. These shares count only the 40% of rejections with a structured reason; chapter 07 labels the rest from free-text notes, and the Mobile share rises from 0.21 to about 0.63.

## Checkpoint

- [ ] ingest quarantines nothing on clean data
- [ ] the planted referral and Mobile-screen patterns show in the features
