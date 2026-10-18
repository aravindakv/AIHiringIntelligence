# Human in the loop

Code: [`hiring_intel/hitl.py`](../hiring_intel/hitl.py), graph node `human_review` in [`graph.py`](../hiring_intel/graph.py), UI tab **Human review**.

## Policy: when a human must look

Configured in `config/sla.yaml → hitl`. An insight goes to the review queue if **any** rule fires:

| Rule | Why |
| --- | --- |
| `confidence_score < 0.6` | The system itself is unsure |
| `severity == high` | High impact if wrong |
| Evaluator flags: `ungrounded_numbers`, `invalid_rag_ref`, `hallucination_risk` | Evidence did not check out |
| OpenJev says the recommendation changes **compensation, headcount, or a person's workload/role** | People decisions need an accountable owner |
| OpenJev unsure about sensitivity | Default to a human |

Everything else is `auto_published`.

## Three modes

| Mode | Behaviour | Use |
| --- | --- | --- |
| `queue` (default) | Run completes; flagged insights wait as `pending` | Dashboard workflow; reviewers come later |
| `interrupt` | LangGraph `interrupt()` pauses the run at `human_review`; `resume(run_id, decisions)` continues | CLI review, demos, approval-before-publish |
| `simulate` | Scripted reviewer approves unless evidence failed | Benchmarks only; recorded as `simulated-policy` |

Terminal review: `python scripts/run_pipeline.py --data data/raw/comp_crunch_small --hitl interrupt`.

## Decisions and what they do

| Action | Effect |
| --- | --- |
| Approve | Status `approved`; text indexed into Chroma as `past_insight`, so later runs can cite it |
| Edit | Status `edited`; your rewritten recommendation is stored and indexed |
| Reject | Status `rejected` with your note; never shown as an action; use these for prompt fixes |

Every decision keeps the reviewer name, note and the reasons it was flagged (`review_queue` table).

## What the benchmark showed

On the large baseline input 7 of 8 insights were sent for review: almost every real finding is either high-severity or touches workload/comp. That is safe but will cause review fatigue. See [IMPROVEMENTS.md](IMPROVEMENTS.md) for tuning ideas (a severity-only rule per agent, batching by owner, auto-approve repeats of approved insights).

## Learning from reviews

- **Approval rate per agent** tells you which agent's prompts need work.
- **Edits** are the best training data you will get: compare the original and edited recommendation to improve the prompt or rules.
- **Rejected with "evidence did not check out"** should match evaluator flags; if not, the evaluator is missing something.
