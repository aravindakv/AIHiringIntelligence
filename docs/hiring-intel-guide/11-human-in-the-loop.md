# 11 — Human in the loop

**Goal:** a policy that decides which insights a person must approve, a review queue, and decisions that flow back into the knowledge base.

**Time:** 40 minutes.

## Concepts

- **When a human must look.** If any rule in `config/sla.yaml → hitl` fires:
  - confidence below 0.6;
  - severity high;
  - evaluator flags (ungrounded numbers, invalid RAG ref, hallucination risk);
  - OpenJev says the recommendation changes **compensation, headcount or a person's workload/role**, or isn't sure.
- **Three modes:**

| Mode | Behaviour | Use |
| --- | --- | --- |
| `queue` | run completes; flagged items wait as `pending` | dashboard review later |
| `interrupt` | LangGraph `interrupt()` pauses the run; `resume()` continues it | approve-before-publish, CLI demo |
| `simulate` | scripted reviewer, recorded as `simulated-policy` | benchmarks only |

- **Feedback loop.** Approved and edited insights are written into Chroma as `past_insight`, so the next run can cite what people accepted. Rejections keep the reviewer's note for prompt fixes.
- **Watch review fatigue.** On the large baseline input 7 of 8 insights are held. That is safe but tiring; chapter 22 lists ways to tune it.

## Step 11.1 — HITL module

**File:** `hiring_intel/hitl.py`
```python
"""Human in the loop.

Policy (config/sla.yaml -> hitl): an insight needs a human if ANY of these fire
  - confidence below threshold
  - severity is high
  - the Evaluation Agent flagged it (ungrounded numbers, invalid RAG ref, hallucination risk)
  - OpenJev says it changes compensation, headcount or a person's workload/role
Otherwise it is auto-published.

Modes (RunConfig.hitl_mode):
  queue      non-blocking: the run finishes, flagged items wait in the review queue (dashboard)
  interrupt  blocking: the LangGraph run pauses with interrupt(); resume with Command(resume=...)
  simulate   a scripted reviewer decides (for benchmarks); recorded as reviewer="simulated-policy"

Approved (or edited) insights are written back to the RAG store as past_insight, so the next run
can cite what humans accepted. Rejected ones are kept with the reviewer's note for prompt tuning.
"""
from __future__ import annotations

import datetime as dt
import json

import yaml

from hiring_intel.contracts import Insight, ReviewItem
from hiring_intel.llm import ROOT
from hiring_intel.observability import execute, log_decision, query
from hiring_intel.openjev import decider

POLICY = yaml.safe_load((ROOT / "config/sla.yaml").read_text())["hitl"]
SENSITIVE_WORDS = ["ctc", "compensation", "offer floor", "headcount", "slots", "workload", "cap ", "close",
                   "re-level", "comp "]


def review_reasons(ins: Insight, eval_flags: list[str], run_id: str) -> list[str]:
    reasons = []
    if ins.confidence_score < POLICY["confidence_below"]:
        reasons.append(f"confidence {ins.confidence_score} < {POLICY['confidence_below']}")
    if ins.severity in POLICY["severity_in"]:
        reasons.append(f"severity {ins.severity}")
    hit = [f for f in eval_flags if f in POLICY["eval_flags_any"]]
    if hit:
        reasons.append("eval flags: " + ", ".join(hit))
    d = decider().yes_no("hitl_sensitive", ins.recommendation, POLICY["openjev_sensitive_question"],
                         yes_keywords=SENSITIVE_WORDS)
    log_decision(run_id, d)
    if d.choice == "yes" and not d.escalated:
        reasons.append(f"sensitive change (openjev p={d.probs['yes']})")
    elif d.escalated:
        reasons.append("sensitivity unclear (openjev low confidence)")
    return reasons


def build_queue(run_id: str, outputs: dict, evals: dict) -> list[ReviewItem]:
    items = []
    for agent, out in outputs.items():
        flags = evals.get(agent, {}).get("flags", []) if isinstance(evals.get(agent), dict) else []
        for k, ins in enumerate(out.insights):
            reasons = review_reasons(ins, flags, run_id)
            items.append(ReviewItem(item_id=f"{run_id}:{agent}:{k}", run_id=run_id, agent=agent, insight=ins,
                                    reasons=reasons, status="pending" if reasons else "auto_published"))
    for it in items:
        _save(it)
    return items


def simulated_reviewer(item: ReviewItem, eval_overall: float) -> tuple[str, str]:
    """Scripted stand-in for a hiring manager, used only in benchmarks."""
    if any("eval flags" in r for r in item.reasons) or eval_overall < 0.6:
        return "rejected", "evidence did not check out"
    return "approved", "numbers verified against dashboard"


def apply_decision(item_id: str, action: str, reviewer: str, note: str = "",
                   edited_recommendation: str | None = None) -> ReviewItem:
    row = query("SELECT json FROM review_queue WHERE item_id=?", (item_id,))
    if row.empty:
        raise KeyError(item_id)
    item = ReviewItem.model_validate_json(row.json.iloc[0])
    if action == "edited" and edited_recommendation:
        item.insight.recommendation = edited_recommendation
    item.status, item.reviewer, item.note = action, reviewer, note
    _save(item)
    if action in ("approved", "edited"):
        try:
            from hiring_intel.rag.retriever import add_past_insight
            seg = next((e.segment.get("role") for e in item.insight.evidence if e.segment.get("role")), "all")
            add_past_insight(f"past::{item.item_id}", f"Past insight (approved by {reviewer}): "
                             f"{item.insight.title}. {item.insight.recommendation}", item.agent, seg or "all")
        except Exception:
            pass
    return item


def pending(run_id: str | None = None) -> list[ReviewItem]:
    sql = "SELECT json FROM review_queue WHERE status='pending'" + (" AND run_id=?" if run_id else "")
    df = query(sql, (run_id,) if run_id else ())
    return [ReviewItem.model_validate_json(j) for j in df.json]


def _save(it: ReviewItem) -> None:
    execute("INSERT OR REPLACE INTO review_queue VALUES (?,?,?,?,?,?,?,?,?)",
            (it.item_id, it.run_id, it.agent, it.status, it.reviewer, it.note, json.dumps(it.reasons),
             it.model_dump_json(), dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
```

## Verify

<!-- verify:11 -->
```bash
python - <<'EOF'
from hiring_intel.ingest import load_all
from hiring_intel.agents.insight_agents import AGENTS
from hiring_intel import hitl
d = load_all("data/raw/comp_crunch_medium")
outs = {n: a.run(d, "ch11") for n, a in AGENTS.items()}
items = hitl.build_queue("ch11", outs, {})
for it in items:
    print(f"{it.status:15} {it.agent:10} {it.insight.title[:48]:48} {'; '.join(it.reasons)[:70]}")
first = next(i for i in items if i.status == "pending")
print("decision:", hitl.apply_decision(first.item_id, "approved", "aravinda", "numbers checked").status)
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
pending         sourcing   Backend: referral out-converts job_board         severity high
pending         sourcing   Data: referral out-converts agency               severity high
auto_published  rejection  [sourcing] agency candidates fail tech_1 (Data)  
pending         panel_load 10 interviewers over load policy                 severity high; sensitive change (openjev p=0.636)
pending         offers     Offers below 90% of expected CTC are being decli severity high; sensitive change (openjev p=0.836)
pending         pipeline   SRE onsite breaches SLA                          sensitive change (openjev p=0.636)
pending         pipeline   10 SRE requisitions open too long                severity high; sensitive change (openjev p=0.774)
decision: approved
```

## Checkpoint

- [ ] comp and workload recommendations are held for review
- [ ] approving one changes its status and adds it to the knowledge base
