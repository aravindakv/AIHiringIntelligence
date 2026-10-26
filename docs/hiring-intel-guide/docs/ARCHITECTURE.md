# Architecture

## One-line summary

Pandas computes the numbers, rule-based detectors decide what counts as a finding, OpenJev makes closed-set decisions, the LLM interprets and recommends, a second agent checks the first, and a human signs off anything that changes pay, headcount or someone's workload.

## Graph

```mermaid
flowchart TD
    A[ATS export / MockATSClient] --> I[ingest<br/>validate, quarantine]
    I --> E[enrich<br/>OpenJev labels free-text notes]
    E --> S[Sourcing Quality]
    E --> R[Rejection Pattern]
    E --> P[Panel Load Balancer]
    E --> O[Offer Insights]
    E --> H[Pipeline Health]
    KB[(Chroma KB<br/>benchmarks, SLA, history,<br/>approved past insights)] -. top_k chunks .-> S & R & P & O & H
    RT{{Routing Agent<br/>rules -> OpenJev -> small LLM}} -. model per agent .-> S & R & P & O & H
    S & R & P & O & H --> EV[Evaluation Agent<br/>schema, number grounding, judge]
    EV -- score < 0.6 --> RY[retry on large model] --> EV
    EV --> HR[human_review<br/>queue / interrupt / simulate]
    HR -- approved --> KB
    HR --> OP[Optimization Agent]
    OP --> DB[(SQLite run store)]
    DB --> UI[Streamlit dashboard]
```

With `transport="a2a"` the five insight agents run behind the A2A server instead of in-process; nothing else in the graph changes.

## Ask: a second entry point

HR can also ask their own question in the dashboard's **Ask** tab. This path runs outside the graph, on demand, but reuses its parts:

```mermaid
flowchart LR
    Q[HR question] --> PG{person guard}
    PG -- names a person --> CT[Candidates tab]
    PG --> TR{{OpenJev topic routing}}
    TR -- out of scope --> X[explain what Ask covers]
    TR -- 1-2 topics --> CX[agent feature tables + SIGNALS + RAG]
    CX --> RT{{Routing Agent}} --> L[one LLM call: AskAnswer]
    L --> G[guard + calibrate + grounding check]
    G --> HR[human review policy] --> UI[Ask tab]
```

Details: [ASK.md](ASK.md).

## Components

| Component | File | Job | Model |
| --- | --- | --- | --- |
| Ingest | `ingest.py` | Load 6 CSVs, parse dates, drop duplicates, referential integrity, quarantine count | none |
| Enrich | `enrich.py` | Fill missing rejection/decline reasons from free-text notes | OpenJev |
| Sourcing Quality | `agents/insight_agents.py` | Channels to grow or cut, per role | routed |
| Rejection Pattern | same | Where/why rejections happen; JD vs sourcing vs interview calibration | routed (usually large) |
| Panel Load Balancer | same | Overloaded/idle interviewers + concrete move plan | routed (usually small) |
| Offer Insights | same | Why offers decline (comp, timing, competing) | routed (usually large) |
| Pipeline Health | same | SLA breaches, aging requisitions, funnel | routed (usually small) |
| Routing Agent | `agents/router.py` | Pick small vs large per agent | rules, OpenJev, small LLM |
| Evaluation Agent | `agents/evaluator.py` | Quality gate | deterministic + judge |
| Optimization Agent | `agents/optimizer.py` | Cheaper/faster suggestions from run stats | small LLM |
| HITL | `hitl.py` | Review policy, queue, decisions, RAG write-back | OpenJev for sensitivity |
| A2A | `a2a.py` | Expose agents as A2A services; client used by the graph | none |
| Ask | `ask.py` | Answer HR's free-form questions from the agents' data, with the same guard-rails | OpenJev for topic, routed LLM for the answer |

## State and checkpointing

The graph state holds only JSON-able values: the run config (incl. `data_path`), agent outputs as dicts, eval results, review items. DataFrames stay in a per-path cache in `ingest.py`. That keeps checkpoints small and makes `interrupt()`/`resume()` work for human review.

## Error handling per hop

| Hop | Failure | Handling |
| --- | --- | --- |
| Ingest | missing table/column | hard fail with a clear message |
| Ingest | bad rows, orphan events | dropped and counted as quarantined |
| Enrich | low-confidence label | stored as `unclassified`, never forced |
| Retrieval | empty or Chroma error | agent continues, status `degraded`, confidence capped |
| Routing | OpenJev unsure | escalates to a small-LLM decision |
| LLM call | timeout / 429 / 5xx | LiteLLM `num_retries=2`, then agent `failed` |
| LLM JSON | schema violation | one repair turn with the validation error |
| Agent | any exception | caught in `BaseInsightAgent.run`, recorded in `errors` |
| Evaluate | score < 0.6 or failed agent | one retry on the large model |
| Judge | judge call fails | deterministic checks still score it (judge = null) |
| A2A | remote task fails | `failed` AgentOutput with the task id and state |
| UI | failed agent | red card with the error, never hidden |

Two demo switches prove this: `crash_agent="offers"` (agent raises, run recovers on retry) and `inject_fault="sourcing"` (fabricated numbers, evaluator catches them, retry on large).

## Names and privacy

The generated CSVs carry `candidate_name`, `interviewer_name` and `hiring_manager` so demos are easy to follow (dashboard → **Candidates** tab). IDs remain the only join keys: names repeat (about 1,030 unique names for 1,179 candidates on the small input) and can change.

Names never leave the dashboard. Every feature table becomes prompt text through `metrics.to_text`, which drops `PII_COLUMNS`, and panel features strip interviewer names at the source. `tests/test_privacy.py` captures every prompt in a full run and fails if any candidate, interviewer or hiring-manager name appears. That keeps names out of LLM calls, LangSmith traces and DeepEval/Confident AI uploads, and out of the model's reasoning, where they could bias hiring recommendations.

## Why these choices

- **LangGraph, not n8n:** parallel fan-out + conditional retry + `interrupt()` for HITL, all testable in pytest.
- **Pandas-first features:** the LLM never counts rows; every evidence number can be checked mechanically.
- **OpenJev for closed-set decisions:** labelling thousands of notes or answering yes/no needs no generation; it costs $0 API and is deterministic.
- **Rules-first routing:** a model choosing a model on every call would eat the savings it creates.
- **A2A in its own module:** agents can be deployed and scaled separately or swapped for another team's agent; the graph only knows names and cards.
