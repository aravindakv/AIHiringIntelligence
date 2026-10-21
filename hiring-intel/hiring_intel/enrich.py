"""Enrichment: fill missing rejection / decline reasons by classifying free-text notes with OpenJev.

This is the highest-volume "simple decision" in the system: thousands of notes, a closed label
set, no generation needed. Running it on an LLM would cost tokens per note; OpenJev costs $0 API.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from hiring_intel.ingest import HiringData
from hiring_intel.llm import call_structured, model_for, price_usd
from hiring_intel.observability import log_decision
from hiring_intel.openjev import KeywordFallback, OpenJevDecider


class NoteLabel(BaseModel):
    i: int              # the note's number in the batch, so a skipped or reordered label can't shift the rest
    label: str

class NoteLabels(BaseModel):
    labels: list[NoteLabel]

REJECTION_LABELS = {
    "JD mismatch": ["jd", "job description", "posting", "does not match", "different scope", "native"],
    "fundamentals": ["basic", "fundamentals", "data structures", "complexity", "sql", "joins", "indexing",
                     "recursion", "hashing"],
    "system design": ["design", "scaling", "partition", "consistency", "failure modes"],
    "communication": ["explanations", "clarify", "rambled", "structure the answer", "listen"],
    "coding": ["coding", "code", "bugs", "edge cases", "tests"],
    "experience gap": ["junior", "years of experience", "ownership"],
    "culture add": ["collaboration", "dismissive", "feedback"],
}
REJECTION_HYPOTHESES = {
    "JD mismatch": "The candidate's background does not match the job description.",
    "fundamentals": "The candidate is weak on basic computer science or SQL fundamentals.",
    "system design": "The candidate's system design was weak.",
    "communication": "The candidate communicated poorly.",
    "coding": "The candidate's code was buggy or incomplete.",
    "experience gap": "The candidate is too junior or lacks enough experience.",
    "culture add": "The candidate showed poor collaboration or teamwork.",
}
DECLINE_LABELS = {
    "compensation": ["below", "compensation", "ctc", "20% more", "expectation"],
    "competing offer": ["another company", "competing"],
    "timing": ["too long", "notice period", "start date", "moved on"],
    "location": ["relocate", "remote", "location"],
}
DECLINE_HYPOTHESES = {
    "compensation": "The candidate declined because the pay offered was too low.",
    "competing offer": "The candidate accepted an offer from another company.",
    "timing": "The candidate declined because the process or the start date did not work out in time.",
    "location": "The candidate declined because of the job location or relocation.",
}
# the abstain option: notes like "Panel was not convinced overall." should escalate, not get a label
UNCLEAR = "unclear"
UNCLEAR_HYPOTHESIS = "The note does not name any specific skill, problem or reason."
HYPOTHESES = {"classify_rejection_note": REJECTION_HYPOTHESES, "classify_decline_note": DECLINE_HYPOTHESES}
PREFIX = {"classify_rejection_note": "Interviewer note: ", "classify_decline_note": "Recruiter note on a declined offer: "}


def _canon(label: str, labels: dict) -> str:
    """Map an LLM's label onto the label set (case and spacing tolerant); anything else is unclassified."""
    norm = {k.lower().replace("_", " ").strip(): k for k in labels}
    return norm.get(str(label).lower().replace("_", " ").strip(), "unclassified")


def _classify(decider: OpenJevDecider, run_id: str, task: str, notes: pd.Series, labels: dict,
              mode: str = "openjev") -> pd.Series:
    """Return one label per row of `notes` (same index)."""
    if mode == "llm":
        # the expensive alternative (benchmark baseline): real notes are mostly unique, so every note
        # goes to the LLM, 50 notes per call. Labels are matched by number and per row, never by note
        # text, so one bad batch can't relabel every other copy of the same note.
        kw = OpenJevDecider(KeywordFallback())
        notes_l = [str(n) for n in notes.tolist()]
        out = ["unclassified"] * len(notes_l)
        for b in range(0, len(notes_l), 50):
            batch = notes_l[b:b + 50]
            parsed, _ = call_structured(
                agent=f"enrich:{task}", run_id=run_id, model=model_for("small"),
                system=f"Label each numbered interviewer or recruiter note with exactly one of: {list(labels)}. "
                       "Use the label text exactly. Return one entry per note: "
                       '{"labels": [{"i": <note number>, "label": "<label>"}, ...]}.',
                user="\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch)), schema=NoteLabels,
                mock_fn=lambda batch=batch: NoteLabels(labels=[
                    NoteLabel(i=i + 1, label=kw.choose(task, t, "reason?", list(labels), keywords=labels).choice)
                    for i, t in enumerate(batch)]))
            for item in parsed.labels:
                if 1 <= item.i <= len(batch):
                    out[b + item.i - 1] = _canon(item.label, labels)
        return pd.Series(out, index=notes.index)
    mapping = {}
    options = list(labels) + [UNCLEAR]
    hyps = {**HYPOTHESES[task], UNCLEAR: UNCLEAR_HYPOTHESIS}
    for note, n in notes.value_counts().items():         # unique notes only; results memoised
        d = decider.choose(task, str(note), "What is the main reason described in this note?", options,
                           keywords=labels, hypotheses=hyps, premise_prefix=PREFIX[task], abstain=UNCLEAR)
        log_decision(run_id, d, n_items=int(n))
        # unclear or low confidence: don't force a label; the note stays visible as "unclassified"
        mapping[note] = "unclassified" if d.escalated else d.choice
    return notes.map(mapping)


def enrich(data: HiringData, run_id: str, decider: OpenJevDecider, mode: str = "openjev") -> dict:
    t0 = time.perf_counter()
    ev = data.stage_events
    if "reason" not in ev.columns:
        ev["reason_source"] = "structured"
        ev["reason"] = ev["rejection_reason"]
    need = ev[(ev.result == "reject") & ev.rejection_reason.isna() & ev.rejection_note.notna()]
    ev.loc[need.index, "reason"] = _classify(decider, run_id, "classify_rejection_note", need.rejection_note,
                                             REJECTION_LABELS, mode)
    ev.loc[need.index, "reason_source"] = mode

    of = data.offers
    if "reason" not in of.columns:
        of["reason"] = of["decline_reason"]
    dneed = of[(of.status == "declined") & of.decline_reason.isna() & of.decline_note.notna()]
    of.loc[dneed.index, "reason"] = _classify(decider, run_id, "classify_decline_note", dneed.decline_note,
                                              DECLINE_LABELS, mode)

    # what an LLM would have cost for the same job (one small-model call per note, ~120 tokens in / 10 out)
    n_items = len(need) + len(dneed)
    small = model_for("small")
    llm_usd = price_usd(small, 120 * n_items, 10 * n_items)
    acc = _accuracy(data) or {}
    ms = int((time.perf_counter() - t0) * 1000)
    uniq = need.rejection_note.nunique() + dneed.decline_note.nunique()
    return {"classified": n_items, "unique_notes": uniq,
            "backend": "llm-small" if mode == "llm" else decider.backend.name,
            "compute_ms": ms, "ms_per_unique_note": round(ms / max(uniq, 1)),
            "llm_equivalent_usd": round(llm_usd, 4), **acc}


def _accuracy(data: HiringData) -> dict | None:
    """accuracy_vs_truth counts "unclassified" as wrong; accuracy_when_labelled ignores escalated notes.
    Keep both: a labeller that escalates the ambiguous 8% is doing its job, one that mislabels them is not."""
    f = Path(data.path) / "_labels.csv"
    if not f.exists():
        return None
    lab = pd.read_csv(f)
    rej = lab[lab.kind == "rejection"].merge(
        data.stage_events[data.stage_events.reason_source != "structured"][["candidate_id", "stage", "reason"]],
        on=["candidate_id", "stage"])
    if not len(rej):
        return None
    hit = rej.true_reason == rej.reason
    esc = rej.reason == "unclassified"
    return {"accuracy_vs_truth": round(float(hit.mean()), 3),
            "accuracy_when_labelled": round(float(hit[~esc].mean()), 3) if (~esc).any() else None,
            "escalated_share": round(float(esc.mean()), 3)}