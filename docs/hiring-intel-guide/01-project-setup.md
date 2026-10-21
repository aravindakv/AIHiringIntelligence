# 01 — Project setup

**Goal:** an empty project that installs, knows its model tiers and SLA policy, and has one-word commands for everything later chapters do.

**Time:** 20 minutes.

## Concepts

- **Configuration, not constants.** Model names, prices, latency assumptions, SLA days and the human-review policy all live in YAML under `config/`. The Routing Agent, cost reports and HITL policy read them, so you can change a price or a threshold without touching code.
- **Two dependency files.** `requirements.txt` is everything the app needs. `requirements-ml.txt` holds the heavy, optional parts (torch, transformers, sentence-transformers) for real OpenJev and MiniLM embeddings. Keeping them apart keeps the app's Docker image small.
- **Offline first.** `.env.example` defaults to `LLM_BACKEND=mock`, so everything runs without keys.

## Step 1.1 — Create the project and virtual environment

```bash
mkdir hiring-intel && cd hiring-intel
git init
python3 -m venv .venv && source .venv/bin/activate
```

## Step 1.2 — Dependencies

**File:** `requirements.txt`
```text
langgraph>=1.0
langgraph-checkpoint-sqlite>=2.0
litellm>=1.50
pydantic>=2.7
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
chromadb>=1.0
fastapi>=0.110
uvicorn>=0.29
httpx>=0.27
streamlit>=1.35
plotly>=5.20
pyyaml>=6.0
python-dotenv>=1.0
tabulate>=0.9
pytest>=8.0
langsmith>=0.4
deepeval>=4.0
```

**File:** `requirements-ml.txt`
```text
# heavier, optional: real embeddings + OpenJev weights (install where you run them)
sentence-transformers>=3.0
transformers>=4.57          # Qwen3.5 support
torch>=2.3
accelerate>=0.30
```

```bash
pip install -r requirements.txt
```

## Step 1.3 — Environment and ignore files

`.env.example` documents every switch the system reads. Copy it to `.env`, which stays out of git.

**File:** `.env.example`
```bash
# offline demo: mock | real models: litellm
LLM_BACKEND=mock
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
# embeddings: auto (MiniLM, falls back to hashing) | minilm | hashing
EMBEDDINGS=auto
# OpenJev: auto | http | local | fallback ; when http, where the service runs
OPENJEV_BACKEND=auto
OPENJEV_URL=
# Hugging Face read token: faster model downloads, no rate-limit warnings (huggingface.co -> Settings -> Access Tokens)
HF_TOKEN=
# A2A: where the agent server runs (transport=a2a)
A2A_BASE_URL=http://localhost:8100
HI_DB=data/runs.db
# --- external evaluation ---
# LangSmith: tracing + hosted golden dataset + experiments (https://smith.langchain.com)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=hiring-intel
# LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # EU region accounts
# DeepEval judge: offline | openjev | llm ; Confident AI upload: run `deepeval login` once
DEEPEVAL_JUDGE=offline
DEEPEVAL_MODEL=
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

**File:** `.gitignore`
```text
.venv/
__pycache__/
data/raw/
data/chroma/
data/state/
*.db
.env
.deepeval/
.pytest_cache/
```

**File:** `.dockerignore`
```text
.venv
data/raw
data/chroma
data/eval/runs
*.db
.env
```

```bash
cp .env.example .env
```

`.env` is where every secret goes: `OPENAI_API_KEY`, `LANGSMITH_API_KEY` and `HF_TOKEN` (a free Hugging Face read token from huggingface.co → Settings → Access Tokens; it speeds up model downloads and removes the "unauthenticated requests" warning). It is in `.gitignore`; never commit it.

## Step 1.4 — Model tiers and prices

Two tiers: **small** for agents that mostly read thresholds, **large** for causal reasoning and compensation decisions. The `sim_*` values are used only by the mock backend to simulate latency. The prices are examples: replace them from your provider's pricing page before you quote USD.

**File:** `config/models.yaml`
```yaml
# Model tiers used by the Routing Agent.
# Prices are USD per 1M tokens. They are EXAMPLES: copy current prices from your
# provider's pricing page before you quote USD figures anywhere.
tiers:
  small:
    model: gpt-4o-mini            # or anthropic/claude-haiku-*, ollama/llama3.1
    input_per_1m: 0.15
    output_per_1m: 0.60
    # used only by the offline mock backend to simulate latency
    sim_base_ms: 350
    sim_ms_per_out_token: 9
    sim_ms_per_1k_in_tokens: 40
  large:
    model: gpt-4o                 # or anthropic/claude-sonnet-*
    input_per_1m: 2.50
    output_per_1m: 10.00
    sim_base_ms: 700
    sim_ms_per_out_token: 22
    sim_ms_per_1k_in_tokens: 90

judge:
  model: gpt-4o                   # judge >= generator, ideally a different family

# OpenJev handles simple closed-set decisions (no generation, $0 API cost).
openjev:
  repo: AlexWortega/openjev
  # checkpoints that exist in the repo: qwen3.5-0.8b-nli-v2s-long (CPU-friendly), qwen3.5-4b-nli-v2 (GPU),
  # qwen3.5-4b-nli, qwen3.5-35b-a3b-nli. Override with env OPENJEV_SUBFOLDER.
  subfolder: qwen3.5-0.8b-nli-v2s-long
  # backend: auto | local | http | fallback
  #   auto  = HTTP service if OPENJEV_URL is set, else local weights, else keyword fallback
  backend: auto
  min_confidence: 0.55            # yes / no: below this, a decision escalates to the LLM / a human
  min_confidence_multi: 0.30      # 3+ options: probability spreads across options, so the floor is lower
```

## Step 1.5 — SLA and human-review policy

**File:** `config/sla.yaml`
```yaml
stage_sla_days:
  applied: 6
  screen: 6
  tech_1: 7
  tech_2: 7
  onsite: 7
  offer: 5
max_days_open:
  default: 45
  senior: 60
max_interviews_per_week: 4

# Human-in-the-loop policy: an insight goes to the review queue if ANY rule fires.
hitl:
  confidence_below: 0.6
  severity_in: [high]
  eval_flags_any: [hallucination_risk, ungrounded_numbers, invalid_rag_ref]
  openjev_sensitive_question: "Does this recommendation change compensation, headcount, or an individual person's workload or role?"
```

## Step 1.6 — Packages and Makefile

Python needs `__init__.py` files to treat the folders as packages. The top-level one also loads `.env` on first import, so your keys and switches work for plain `python …` runs, not only in Docker. Variables set in your shell win over `.env`.

**File:** `hiring_intel/__init__.py`
```python
"""Hiring Intelligence System.

Loads the repo's .env on first import, so settings such as LLM_BACKEND, OPENAI_API_KEY, LANGSMITH_API_KEY
and HF_TOKEN work for plain `python ...` runs too (Docker Compose passes .env via env_file). Variables
already set in the shell win over .env.
"""
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:  # python-dotenv not installed: rely on the shell environment
    pass
```

**File:** `hiring_intel/agents/__init__.py`
```python
```

**File:** `hiring_intel/rag/__init__.py`
```python
```

**File:** `hiring_intel/evals/__init__.py`
```python
```

**File:** `scripts/__init__.py`
```python
```

**File:** `Makefile`
```makefile
.PHONY: setup data index golden run eval bench app a2a test docker-up ls-eval ls-offline deepeval deepeval-hub
setup:      ; python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
data:       ; python scripts/generate_mock_data.py --all
index:      ; python scripts/write_knowledge_base.py && python -c "from hiring_intel.rag.retriever import build_index; print(build_index())"
golden:     ; python scripts/make_golden_set.py
run:        ; python scripts/run_pipeline.py --data data/raw/baseline_medium
eval:       ; python scripts/run_eval.py --size medium --variant optimized
bench:      ; python scripts/benchmark.py
app:        ; streamlit run app/dashboard.py
a2a:        ; uvicorn hiring_intel.a2a:server --port 8100
test:       ; pytest -q
docker-up:  ; docker compose up -d --build
ls-eval:     ; python scripts/langsmith_eval.py --variant optimized --size medium
ls-offline:  ; python scripts/langsmith_eval.py --offline --size medium
deepeval:    ; pytest -q tests_eval
deepeval-hub:; deepeval test run tests_eval/test_deepeval.py
```

Some targets point at scripts from later chapters; they work once those chapters are done.

## Verify

<!-- verify:01 -->
```bash
python -c "import langgraph, litellm, chromadb, pydantic, pandas, langsmith, deepeval; print('imports ok')"
python -c "import yaml; c=yaml.safe_load(open('config/models.yaml')); print({k: v['model'] for k, v in c['tiers'].items()})"
```

Expected output (captured on a fresh extract, mock backend):

```text
imports ok
{'small': 'gpt-4o-mini', 'large': 'gpt-4o'}
```

## Checkpoint

- [ ] `pip install -r requirements.txt` finished without errors
- [ ] `.env` exists and says `LLM_BACKEND=mock`
- [ ] the two tiers print
