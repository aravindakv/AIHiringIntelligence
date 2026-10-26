# 22 — Learning path and improvement points

**Goal:** turn the finished build into understanding, and know what to fix next.

## How to learn from this build

Rebuilding the code teaches less than changing it and predicting the result. For each chapter:

1. **Predict** what a change will do (for example, "routing Offers to small saves X and fails case c-off").
2. **Change** one thing.
3. **Measure** with the golden set, the benchmark and a LangSmith experiment.
4. **Explain** the gap between prediction and result in one sentence in your notes.

The learning path gives one such exercise per concept, in build order.

**File:** `docs/LEARNING_PATH.md`
```markdown
# Learning path: build understanding in the order the system is built

Each step names the concept, the file to read, a small exercise, and what "understood" looks like. About 2 weeks at 1–2 hours a day; the exercises are the point.

| # | Concept | Read | Exercise | You understand it when |
| --- | --- | --- | --- | --- |
| 1 | Data first | `scripts/generate_mock_data.py` | Add a 6th scenario (for example `slow_offers`: offers wait 15+ days) | You can predict which agent should fire before running it |
| 2 | Features, not prompts | `hiring_intel/metrics.py` | Add `time_to_offer` per role to `pipeline_features` | You can explain why the LLM never counts rows |
| 3 | Structured output | `contracts.py`, `llm.py` | Make the LLM return a bad JSON on purpose; watch the repair turn | You can name every field's owner (LLM vs code) |
| 4 | LLM plumbing | `llm.py` | Switch to `LLM_BACKEND=litellm` with a cheap model; compare tokens with the mock estimate | You can compute a call's USD by hand |
| 5 | RAG | `rag/retriever.py`, `scripts/write_knowledge_base.py` | Write 10 queries with expected chunk ids; measure hit@2 and hit@4 for hashing vs MiniLM | You can justify top_k and the distance cutoff with numbers |
| 6 | One agent end to end | `agents/base.py`, `agents/insight_agents.py` | Write a 6th agent: Interviewer Calibration (pass-rate deviation per interviewer) | It returns a valid AgentOutput and appears in the dashboard |
| 7 | Orchestration | `graph.py` | Add your agent to the graph; draw the graph with `app().get_graph().draw_mermaid()` | You can explain fan-out, fan-in and the retry edge |
| 8 | Model routing | `agents/router.py` | Change thresholds so Offers goes small; rerun the golden set | You can show the cost/quality trade-off in one table |
| 9 | Simple decisions with OpenJev | `openjev.py`, `enrich.py` | Run the real model (`requirements-ml.txt`); label 50 notes by hand; tune `min_confidence` | You know where OpenJev beats an LLM and where it must escalate |
| 10 | Evaluation | `agents/evaluator.py`, `scripts/run_eval.py` | Add 5 golden cases, including one trap; hand-score 10 outputs vs the judge | You trust the eval score more than your own eyeballing |
| 11 | Human in the loop | `hitl.py` | Run `--hitl interrupt`; edit one insight; confirm it appears as `past_insight` in the next run's evidence | You can explain why each rule sends an item to a human |
| 12 | Cost & latency | `report.py`, `docs/COST_AND_LATENCY.md` | Compute one run's E2E by hand from the `timings` table | Your number matches the dashboard |
| 13 | Optimization | `scripts/benchmark.py` | Add a lever (prompt compression or batching evaluator calls); measure before/after | You report the saving next to the golden score |
| 13b | External validation | `hiring_intel/evals/`, `docs/EXTERNAL_EVALUATION.md` | Run two LangSmith experiments (optimized vs all_large) and compare them; run the DeepEval suite with an LLM judge and upload to Confident AI | You can show a regression in a tool someone else wrote, not only in your own evaluator |
| 14 | A2A | `a2a.py`, `docs/A2A.md` | Run the A2A server; call one agent with curl; then run the graph with `--transport a2a` | You can explain A2A vs MCP in two sentences |
| 15 | Deploy | `docker-compose.yml`, `docs/DEPLOYMENT.md` | Deploy on your Linux box behind Caddy with basic auth | A colleague opens the dashboard from their laptop |

## Reading list (concepts, not this repo)

- LangGraph docs: graphs, state reducers, `interrupt()` and checkpointers.
- The A2A protocol specification (Agent Card, JSON-RPC methods, task lifecycle).
- The OpenJev model card on Hugging Face (NLI decisions, limitations).
- LLM-as-judge literature: position bias, self-preference, judge validation against humans.
- Any text on two-proportion z-tests and minimum sample sizes; it explains this repo's small-input false positives.

## How to practise "would I act on this?"

For every insight in a run, write one line: act / push back / ignore, and why. Then compare with the evaluator's score and the HITL decision. Where you disagree, either the agent, the grounding or the evaluation needs work; that is the brief's closing point, made into a habit.
```

## What to improve next

Ordered by impact on trust, then cost, then operations. Each item comes from something observed in this build.

**File:** `docs/IMPROVEMENTS.md`
```markdown
# Improvement points

Ordered by impact on trust first, then cost, then operations. Each item says what was observed in this build and a concrete next step.

## Quality and trust

1. **Small-sample false positives.** On small inputs the healthy control raised two medium-severity findings (golden 15/18). *Next:* require a minimum sample per segment (e.g. 100 applicants), apply a multiple-comparison correction across role × channel cells, and downgrade severity when n is small.
2. **The mock judge is too kind.** Every agent scores ~0.97 in mock mode, so the eval tab cannot separate good from mediocre. *Next:* run the real LLM judge, validate it against 10 hand scores, and add pairwise comparison (old vs new prompt) instead of absolute scores.
3. **RAG evidence relevance.** With `top_k=2` the Offers agent sometimes cites the "offer timing" chunk for a compensation finding. *Next:* add a relevance check (OpenJev: "does this chunk support this claim?") and drop unsupported rag evidence.
4. **Confidence calibration.** The cap is a heuristic. *Next:* log human approve/reject per confidence bucket and fit a calibration curve (reliability diagram on the Evaluation tab).
5. **Causal claims.** "JD problem vs interview problem" is inferred from patterns, not tested. *Next:* track interventions (JD rewritten on date X) and compare before/after; store results as `past_insight`.
6. **Real OpenJev accuracy.** Fallback accuracy (0.92) is on templated notes. *Next:* label 200 real notes, measure the model, tune `min_confidence`, report escalation rate.

7a. **Ask: measure topic routing.** *Next:* label 30 real HR questions with their topic, measure OpenJev vs the keyword fallback, tune the hypotheses, and add the set to CI. Then add multi-turn context (the previous answer) and a semantic cache for repeated questions.

## Human in the loop

7. **Review fatigue.** 7 of 8 insights went to review on the large baseline input. *Next:* per-agent rules (Pipeline aging reqs → auto), auto-approve insights matching a previously approved one, group the queue by owner, and add an SLA for reviews.
8. **Use edits as training data.** *Next:* store original vs edited text and feed the top edits into the agent prompt as few-shot examples.
9. **Role-based access.** *Next:* reviewers see only their org's insights; comp-related items need a comp partner.

## Cost and latency

10. **Batch the evaluator.** One judge call per agent; batch all five into one call with per-agent sections.
11. **Prompt caching at the provider.** The system prompt + schema repeat on every call; provider-side prompt caching cuts input cost on long prefixes.
12. **Streaming and partial results.** Show each agent's card as it finishes instead of after the whole run.
13. **Semantic cache.** Cache by feature-table hash per agent so unchanged segments skip the LLM between days.

## Engineering and operations

14. **Postgres instead of SQLite** once more than one person runs pipelines; add migrations.
15. **Tracing.** Add OpenTelemetry or Langfuse via LiteLLM callbacks; link `run_id` across graph, A2A and LLM spans.
16. **A2A hardening.** Auth on Agent Cards, `data_path` allow-list, streaming (`message/stream`), and moving to the official SDK.
17. **Real ATS connector.** Replace `MockATSClient` with Greenhouse/Lever/Workday APIs; expose it as an MCP tool.
18. **Scheduling.** A daily run with a diff ("new since yesterday") is more useful to an EM than ad-hoc runs.
19. **CI.** Run pytest + the golden set (mock) on every push; fail the build on a golden regression, like the compression bug this build caught.
20. **Fine-tuned rejection-reason classifier.** The brief's stretch goal: compare a small fine-tuned classifier with OpenJev on the same 200 labelled notes.
```

## Suggested two-week plan

| Days | Chapters | Done when |
| --- | --- | --- |
| 1 | 01–03 | data generated, contract validates |
| 2 | 04–06 | gateway prices calls, RAG retrieves per agent |
| 3 | 07–08 | notes labelled; routing decisions logged |
| 4–5 | 09–11 | five agents, evaluator, HITL; **Interim Connect 1** |
| 6 | 12–13 | full graph with faults recovering |
| 7 | 14–15 | A2A, golden set, benchmark |
| 8 | 16–17 | LangSmith experiments, DeepEval on Confident AI |
| 9 | 18–19 | dashboard, CI; **Interim Connect 2** |
| 10–11 | 20–21 | deployed on your box; docs written |
| 12 | — | real-model run, numbers refreshed, demo recorded |
