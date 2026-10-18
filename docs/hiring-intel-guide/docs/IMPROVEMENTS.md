# Improvement points

Ordered by impact on trust first, then cost, then operations. Each item says what was observed in this build and a concrete next step.

## Quality and trust

1. **Small-sample false positives.** On small inputs the healthy control raised two medium-severity findings (golden 15/18). *Next:* require a minimum sample per segment (e.g. 100 applicants), apply a multiple-comparison correction across role × channel cells, and downgrade severity when n is small.
2. **The mock judge is too kind.** Every agent scores ~0.97 in mock mode, so the eval tab cannot separate good from mediocre. *Next:* run the real LLM judge, validate it against 10 hand scores, and add pairwise comparison (old vs new prompt) instead of absolute scores.
3. **RAG evidence relevance.** With `top_k=2` the Offers agent sometimes cites the "offer timing" chunk for a compensation finding. *Next:* add a relevance check (OpenJev: "does this chunk support this claim?") and drop unsupported rag evidence.
4. **Confidence calibration.** The cap is a heuristic. *Next:* log human approve/reject per confidence bucket and fit a calibration curve (reliability diagram on the Evaluation tab).
5. **Causal claims.** "JD problem vs interview problem" is inferred from patterns, not tested. *Next:* track interventions (JD rewritten on date X) and compare before/after; store results as `past_insight`.
6. **Real OpenJev accuracy.** Fallback accuracy (0.92) is on templated notes. *Next:* label 200 real notes, measure the model, tune `min_confidence`, report escalation rate.

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
