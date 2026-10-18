# External evaluation: LangSmith and DeepEval (Confident AI)

The in-repo Evaluation Agent grades every run as the pipeline runs. External tools add three things it cannot: **traces** you can open call by call, **experiments** you can compare across prompt and model versions, and **independent metrics** written by someone other than you, so the system is not only grading itself.

| Tool | What it does here | Files |
| --- | --- | --- |
| LangSmith | Tracing of every graph node and LLM call (model, tokens, USD, latency on each span); the golden set as a hosted dataset; one experiment per variant × size | `hiring_intel/evals/langsmith_eval.py`, `scripts/langsmith_eval.py`, `llm.py` (`@traceable`) |
| DeepEval | Pytest-style regression suite over real pipeline output: grounded numbers, actionability, faithfulness, hallucination, G-Eval rubric | `hiring_intel/evals/deepeval_suite.py`, `tests_eval/test_deepeval.py` |
| Confident AI | DeepEval's hosted hub: stores each test run, compares runs, tracks regressions | `deepeval login`, then `deepeval test run` |
| OpenJev in DeepEval | DeepEval's `system_one` eval mode judged by OpenJev: faithfulness and hallucination with no LLM | `hiring_intel/evals/openjev_deepeval.py` |
| GitHub Actions | Offline checks on every push; real-model experiments nightly | `.github/workflows/eval.yml` |

## 1. LangSmith

### Set up

1. Create an account at smith.langchain.com and an API key (Settings → API Keys).
2. In `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...`, `LANGSMITH_PROJECT=hiring-intel` (EU accounts also set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`).
3. Run anything (`python scripts/run_pipeline.py`). Open the project: each run is a trace with the LangGraph nodes as children and every `call_structured` span tagged with agent, tier, tokens, USD and latency.

### Golden set as an experiment

```bash
python scripts/langsmith_eval.py --variant all_large --size medium
python scripts/langsmith_eval.py --variant optimized --size medium
```

The first call uploads `data/golden/golden_set.jsonl` as dataset `hiring-intel-golden`. Each call records an experiment (`optimized-medium-…`). In the dataset's **Experiments** tab, select both and **Compare**: per case you see the pass/fail, grounding, eval score, cost and latency side by side. That is your before/after for the optimization report, with the evidence one click away.

| Evaluator | Score |
| --- | --- |
| `golden_pass` | 1 if the case's expectation holds (title, severity, forbidden titles) |
| `numbers_grounded` | share of evidence numbers found in the agent's features |
| `eval_agent_overall` | the in-repo Evaluation Agent's score, so you can see if it agrees with the golden check |
| `run_cost_usd`, `run_e2e_seconds` | cost and latency of the run that produced the case |
| `golden_pass_rate` (summary) | share of cases passed in the experiment |

### Offline

`python scripts/langsmith_eval.py --offline` runs the same target and evaluators locally with no LangSmith client: nothing is uploaded. CI uses it as a gate (must stay 18/18 on medium inputs).

Results in this build (mock backend): medium 18/18, small 15/18, matching `scripts/run_eval.py`.

## 2. DeepEval and Confident AI

### Metrics

| Metric | Judge | What it catches |
| --- | --- | --- |
| Grounded numbers (custom) | none | evidence numbers not present in the feature table |
| Actionable (rule, custom) | none | recommendations with no owner, time frame or number |
| Faithfulness | OpenJev (`system_one`) or LLM | claims not supported by what the agent was given (features + RAG chunks) |
| Hallucination | OpenJev (`system_one`) or LLM | statements contradicting the feature table |
| G-Eval "Actionability" | LLM | a rubric judgement: owner, action, time frame, follows from evidence |

### Run

```bash
pytest tests_eval/ -q                                   # offline: custom metrics only
DEEPEVAL_JUDGE=openjev OPENJEV_URL=http://localhost:8200 pytest tests_eval/ -q
DEEPEVAL_JUDGE=llm OPENAI_API_KEY=... deepeval test run tests_eval/test_deepeval.py
```

Local LLM judge instead of OpenAI: `deepeval set-ollama --model llama3.1`.
Choose inputs with `DEEPEVAL_INPUTS=baseline_large,comp_crunch_large`.

### Confident AI hub

```bash
deepeval login --api-key <your Confident AI key>     # once
deepeval test run tests_eval/test_deepeval.py         # results upload as a test run
```

On the hub, open **Test Runs** to see each insight's metric scores and reasons, compare against the previous run, and spot regressions after a prompt or model change.

### What it found in this build

The offline suite failed 2 of 20 cases on the first run: on the healthy input, the "no action needed" findings named no owner and no follow-up date. The fix was in the agents (every no-action finding now starts "Recruiting ops lead:" and gives a re-check date), not in the metric. After the fix: 20/20.

### OpenJev as DeepEval's judge

DeepEval 4.x metrics accept `eval_mode="system_one"` with a `system_one_model` that answers yes/no, choice and score questions with probabilities. `OpenJevSystemOne` implements that interface, so faithfulness and hallucination can run on OpenJev alone: local, deterministic, $0 API. It needs the real model; with the keyword fallback the suite refuses to run this mode rather than produce meaningless scores. Validate it once against the LLM judge on 20 insights before you trust it.

## 3. Which to use when

| Question | Use |
| --- | --- |
| Why did this run produce that insight? | LangSmith trace |
| Did my prompt change make things better or worse? | LangSmith experiment comparison |
| Did quality regress since yesterday? | DeepEval test run on Confident AI (nightly CI) |
| Is every number traceable to data? | In-repo evaluator + DeepEval grounded numbers |
| Can I judge cheaply at scale? | DeepEval `system_one` with OpenJev |

Alternatives with the same role: Langfuse (open-source tracing + evals; LiteLLM has a callback), Ragas (RAG-specific metrics), Arize Phoenix (tracing + evals, self-hosted).
