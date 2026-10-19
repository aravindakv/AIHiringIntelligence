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