# Evaluation

## Layers

1. **Schema** (free): Pydantic rejects confidence outside 0–1, empty evidence, recommendations under 30 characters or with generic phrases ("consider improving", "monitor closely").
2. **Deterministic grounding** (free): every numeric evidence value must appear in the features the agent was given (0.721 or 72.1% both match); every `rag` evidence `ref` must be a chunk that was actually retrieved.
3. **Judge**: rubric of actionability, grounding, hallucination risk (1–5). Real mode: an LLM judge (`config/models.yaml → judge`). Mock mode: a heuristic judge + OpenJev "is this actionable?", labelled `heuristic+openjev` in results.

`overall = 0.35 × grounded + 0.15 × refs_valid + 0.5 × judge`, **capped at 0.55 when any number is ungrounded**. Below 0.6 → one retry on the large model.

Ask answers ([ASK.md](ASK.md)) get layers 1 and 2 plus the severity cap; they skip the judge to keep a question cheap. The grounding score is shown with every answer, and a score under 0.8 adds an `ungrounded_numbers` flag and a warning.

## Golden set

`data/golden/golden_set.jsonl`: 18 scenarios over 5 input types. Each names an agent and an expectation (title must mention X, minimum or maximum severity, forbidden titles). It includes a **healthy control** where nothing should be raised above low, and trap expectations (no `[jd]` root cause when the problem is sourcing).

| Input type | What it plants | Cases |
| --- | --- | --- |
| baseline | referral > job board (Backend), agency fails Data tech_1, Mobile JD mismatch, 2 overloaded interviewers, comp declines, SRE onsite SLA + aging reqs | 6 |
| healthy | nothing | 5 |
| comp_crunch | most offers below 90% of expected CTC | 2 |
| interviewer_burnout | 3 people carry most interviews; slow tech stages | 2 |
| sourcing_shift | half of applicants from agencies that fail tech_1 everywhere | 3 |

Run: `python scripts/run_eval.py --size medium --variant optimized`.

## Results (mock backend)

| Size | all_large | optimized |
| --- | --- | --- |
| small (~1.2k candidates) | 15/18 (83%) | 15/18 (83%) |
| medium (~6k) | 18/18 | 18/18 |
| large (~25k) | 18/18 | 18/18 |

**Small-input failures are real false positives:** with about 50 applicants per role × channel, random noise produced a "Data: referral out-converts job board" finding and a "timing is the main decline reason" finding on the healthy control. The z-test threshold is not enough at this sample size. See [IMPROVEMENTS.md](IMPROVEMENTS.md).

**A regression the golden set caught:** compressing the Panel Load features (top and bottom 8 interviewers only) changed the median the agent compared against, so overload went undetected (medium: 15/18). Fixed by computing the panel median on all interviewers before trimming. A related truncation (only the top 8 rows kept, so "8 over policy" instead of 10) was found by comparing chapter outputs in the implementation guide; compressed tables now keep every over-policy row.

## Fault tests

| Test | Expected | Result |
| --- | --- | --- |
| `inject_fault="sourcing"` adds 3 invented numbers per insight | grounding < 0.8 → overall ≤ 0.55 → retry on large | caught, retried, 0.97 after retry |
| `crash_agent="offers"` | run continues, offers retried | recovered |
| HITL interrupt/resume | run pauses with pending items; resumes with decisions | passes (`tests/test_graph.py`) |

## External validation

LangSmith experiments and a DeepEval suite (with Confident AI upload) run the same golden set and per-insight metrics outside the agent that produced them. See [EXTERNAL_EVALUATION.md](EXTERNAL_EVALUATION.md). The DeepEval suite found that "no action" findings lacked an owner and a re-check date; fixed in the agents.

## Judge validation (do this with real models)

Hand-score 10 outputs on the same rubric and report agreement with the LLM judge ("within 1 point on 8/10"). Use a judge model at least as strong as the generator and ideally from a different family.

## Note-label accuracy

`enrich()` compares labels with the generator's hidden truth (`_labels.csv`) three ways: `accuracy_vs_truth` (escalated notes count as wrong), `accuracy_when_labelled` (only notes that got a label) and `escalated_share`. Keyword fallback: 0.92 / 1.00 / 0.08, with ambiguous notes left `unclassified`. Real OpenJev on CPU: 0.385 with question-style hypotheses and a 0.55 floor, 1.00 on clear notes with plain-statement hypotheses ([OPENJEV.md](OPENJEV.md)).
