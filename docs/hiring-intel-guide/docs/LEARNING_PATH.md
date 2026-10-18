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
