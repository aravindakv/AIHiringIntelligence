# Hiring Intelligence System

A multi-agent system that turns ATS hiring data into insights an Engineering Manager can act on. Every insight carries its **recommendation, evidence, confidence, cost and a cheaper alternative**, is checked by an **Evaluation Agent**, and is held for **human review** when it is high-stakes.

Built for *Capstone 2: AI-Powered Hiring Intelligence System*.

| | |
| --- | --- |
| Orchestration | LangGraph: parallel fan-out, retry loop, HITL interrupt/resume |
| Agents | 5 insight agents + Routing, Evaluation and Optimization agents |
| Simple decisions | **OpenJev** (open NLI cross-encoder, $0 API): note labelling, routing tie-breaks, HITL sensitivity, actionability pre-check |
| Agent interop | **A2A** (Agent Cards + JSON-RPC `message/send`) in [`hiring_intel/a2a.py`](hiring_intel/a2a.py) |
| RAG | Chroma + MiniLM (offline fallback: hashing embeddings), per-agent metadata filters |
| LLMs | LiteLLM (OpenAI / Anthropic / Ollama) or an offline **mock** backend |
| External validation | **LangSmith** (traces, hosted golden dataset, experiments) and **DeepEval** + Confident AI hub (regression suite; OpenJev as a $0 judge) — [docs/EXTERNAL_EVALUATION.md](docs/EXTERNAL_EVALUATION.md) |
| UI | Streamlit: overview, **Ask** (HR's own questions, answered from the same data and checks: [docs/ASK.md](docs/ASK.md)), insights, human review, candidates, evaluation, cost & latency, compare inputs |
| Deploy | Docker Compose on a Linux box ([docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)) |

## Quick start (offline, no API key, about 3 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_mock_data.py --all        # 16 datasets, up to ~84k candidates
python scripts/write_knowledge_base.py && python scripts/make_golden_set.py
python scripts/benchmark.py                        # 80 runs: inputs x optimization variants
streamlit run app/dashboard.py                     # http://localhost:8501
```

Real models: copy `.env.example` to `.env`, set `LLM_BACKEND=litellm` and an API key. Real OpenJev: `pip install -r requirements-ml.txt` (or run the `openjev` Docker service).

## What runs where

```text
ingest -> enrich (OpenJev labels free-text notes)
       -> [sourcing | rejection | panel_load | offers | pipeline]   (parallel, each: features -> RAG -> route -> LLM -> validate)
       -> evaluate (schema, number grounding, judge) -> retry < 0.6 on large model
       -> human_review (queue | interrupt | simulate) -> optimize -> persist -> dashboard
```

## Repository map

| Path | What |
| --- | --- |
| `hiring_intel/contracts.py` | Shared output contract (Insight, AgentOutput, Decision, ReviewItem) |
| `hiring_intel/llm.py` | LLM door: LiteLLM or mock, JSON repair, cost, latency, cache |
| `hiring_intel/openjev.py` | OpenJev decider (local / HTTP / fallback) |
| `hiring_intel/enrich.py` | Note classification (OpenJev vs LLM) |
| `hiring_intel/metrics.py` | Deterministic pandas features per agent |
| `hiring_intel/agents/` | Insight agents, router, evaluator, optimizer |
| `hiring_intel/hitl.py` | Human-in-the-loop policy, queue, decisions, RAG write-back |
| `hiring_intel/a2a.py` | A2A server + client (separate file, as required) |
| `hiring_intel/graph.py` | LangGraph orchestrator |
| `hiring_intel/ask.py` | Ask: HR's free-form questions → topic routing → grounded insights |
| `hiring_intel/report.py` | Cost and latency accounting |
| `app/dashboard.py` | Streamlit UI |
| `services/openjev_server.py` | OpenJev as an HTTP service |
| `hiring_intel/evals/` | LangSmith experiment, DeepEval suite, OpenJev as DeepEval judge |
| `tests_eval/` | DeepEval regression suite (`deepeval test run`) |
| `scripts/` | data generator, benchmark, golden eval, LangSmith eval, CLI run |
| `docs/` | everything below |

## Docs

- [Architecture](docs/ARCHITECTURE.md) · [Output contract](docs/CONTRACTS.md) · [LangGraph flow, node by node](docs/GRAPH_FLOW.md) ([HTML](docs/graph_flow.html))
- [Cost & latency definitions](docs/COST_AND_LATENCY.md) · [Optimization report](docs/OPTIMIZATION_REPORT.md)
- [Evaluation](docs/EVALUATION.md) · [External evaluation: LangSmith, DeepEval](docs/EXTERNAL_EVALUATION.md) · [Human in the loop](docs/HUMAN_IN_THE_LOOP.md)
- [Ask: HR questions in the dashboard](docs/ASK.md) · [OpenJev](docs/OPENJEV.md) · [A2A](docs/A2A.md)
- [Deployment (Docker on Linux)](docs/DEPLOYMENT.md)
- [Learning path](docs/LEARNING_PATH.md) · [Improvement points](docs/IMPROVEMENTS.md)
- [Reflection](docs/REFLECTION.md) · [Demo script](docs/DEMO_SCRIPT.md)

## Honest limits of the numbers in this repo

The committed benchmark ran with `LLM_BACKEND=mock`, `EMBEDDINGS=hashing` and the OpenJev **keyword fallback**, because the build machine had no API key and no Hugging Face access. Token counts are estimated (chars/4) and LLM latency is **simulated** from `config/models.yaml`. Cost *ratios* between variants are meaningful; absolute quality and latency must be re-measured with real models (`LLM_BACKEND=litellm`, real OpenJev) before you quote them.
