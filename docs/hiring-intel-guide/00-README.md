# Hiring Intelligence System — Implementation Guide

Build Capstone 2, the AI-Powered Hiring Intelligence System, chapter by chapter. You end with a multi-agent system in Python. It turns ATS data into insights an Engineering Manager can act on, checks them with a second agent and two external tools (LangSmith and DeepEval), holds high-stakes ones for a human, and runs in Docker on your Linux box.

Every chapter adds working code, explains why it is built that way, and ends with a **Verify** step. Each "Expected output" block was captured by running that chapter's commands on a fresh extract of the code, in order. Your numbers will match when you use the same seed and the offline mock backend.

## What you build

```text
ATS CSVs -> ingest -> enrich (OpenJev labels notes)
         -> 5 insight agents in parallel (features -> RAG -> route -> LLM -> validate)
         -> Evaluation Agent -> retry on large model if < 0.6
         -> human review -> Optimization Agent -> SQLite -> Streamlit
         + A2A services, LangSmith traces/experiments, DeepEval suite on Confident AI
```

## Chapter map

| # | Chapter | You add | Capstone criterion |
| --- | --- | --- | --- |
| 01 | Project setup | requirements, config, Makefile | setup instructions |
| 02 | Mock ATS data | generator with 5 scenarios × 4 sizes | ingestion, large test data |
| 03 | Contracts and observability | shared output contract, SQLite run store | decision justification, observability |
| 04 | LLM gateway | one door for every model call; mock + LiteLLM | structured output, cost tracking |
| 05 | Ingest and features | CSV validation, pandas features | grounding |
| 06 | RAG | knowledge base, Chroma retriever | RAG grounding |
| 07 | OpenJev | closed-set decisions, note labelling | model routing, cost |
| 08 | Router and base agent | Routing Agent, shared agent pipeline, prompts | model routing |
| 09 | Insight agents | the five agents | agentic design |
| 10 | Evaluation Agent | grounding check, judge, retry signal | quality |
| 11 | Human in the loop | review policy, queue, interrupt/resume | trust |
| 12 | Optimization, cost and latency | Optimization Agent, E2E latency | cost awareness |
| 13 | Orchestrator | LangGraph graph, CLI | orchestration, error handling |
| 14 | A2A | agents as A2A services | interoperability |
| 15 | Golden set and benchmark | 18 cases, 80-run benchmark | evaluation, before/after |
| 16 | LangSmith | traces, hosted dataset, experiments | external validation |
| 17 | DeepEval and Confident AI | regression suite, OpenJev judge | external validation |
| 18 | Dashboard | Streamlit app | dashboard |
| 19 | Tests and CI | pytest, GitHub Actions | reproducibility |
| 20 | Docker deployment | compose on your Linux box | deployment |
| 21 | Documentation package | README and docs | documentation |
| 22 | Learning path and improvements | what to practise and fix next | reflection |
| 23 | Troubleshooting | known problems and fixes | — |

## Prerequisites

- Linux or macOS, Python 3.11+ (3.12 recommended), git, 8 GB RAM.
- Docker with Compose v2 for chapter 20.
- Optional: an OpenAI or Anthropic key (real models), a LangSmith key (chapter 16), a Confident AI key (chapter 17), a GPU for OpenJev 4b.

Nothing needs a key until you choose to use one. The whole guide runs offline with the mock LLM backend.

## How to use this guide

Every file appears in full under a caption like this:

````text
**File:** `hiring_intel/contracts.py`
```python
...
```
````

Don't copy and paste by hand. `extract_code.py` writes every captioned file to its path:

```bash
mkdir hiring-intel && cd hiring-intel
python3 ../hiring-intel-guide/extract_code.py ../hiring-intel-guide/01-project-setup.md          # dry run: lists files
python3 ../hiring-intel-guide/extract_code.py ../hiring-intel-guide/01-project-setup.md --write  # writes them
python3 ../hiring-intel-guide/extract_code.py ../hiring-intel-guide/*.md --write                 # everything at once
```

It refuses to overwrite a file you changed unless you pass `--force`, and `--check` reports any file that differs from the guide.

Work one chapter at a time: read it, extract it, run its Verify step, then move on. When something fails, check chapter 23 first.

## Honest limits

The expected outputs come from the offline mock backend: the agents' rule-based analysts write the JSON an LLM would write, tokens are estimated and LLM latency is simulated. OpenJev ran as its keyword fallback and embeddings as hashing vectors, because the machine that produced this guide could not download model weights. The code supports the real models; chapters 04, 06, 07 and 17 show how to switch. Re-measure quality and latency with real models before you quote them.
