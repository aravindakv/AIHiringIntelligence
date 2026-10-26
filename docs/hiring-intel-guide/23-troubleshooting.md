# 23 — Troubleshooting

## Setup and data

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: hiring_intel` | running from another folder | run from the repo root, or `export PYTHONPATH=$PWD` |
| `missing table data/raw/.../offers.csv` | dataset not generated | `python scripts/generate_mock_data.py --all` |
| extractor says `exists, differs` | you edited the file | keep your edit, or rerun with `--force` |
| `No datasets in data/raw` (dashboard) | Docker volume hides the image's data | generate data on the host (chapter 20) |

## Models and retrieval

| Symptom | Cause | Fix |
| --- | --- | --- |
| every agent `degraded` right after installing `requirements-ml.txt` | index built with hashing (512 dims), queries now use MiniLM (384) | update: the index now rebuilds itself; on older code run `build_index()` once |
| a real LLM raises `high` findings on `healthy` data | the model saw raw tables without thresholds | keep `SIGNALS` on (default); `guard()` caps severity at the detectors' |
| agents show `degraded` | RAG returned nothing | `python -c "from hiring_intel.rag.retriever import build_index; print(build_index())"`; check `CHROMA_DIR` is writable |
| MiniLM download fails | no access to Hugging Face | `EMBEDDINGS=hashing` (offline), or pre-download the model |
| `Couldn't instantiate the backend tokenizer… need sentencepiece or tiktoken` when loading OpenJev | the subfolder doesn't exist in the repo (the model card lists `-v5` checkpoints that aren't published) | use `qwen3.5-0.8b-nli-v2s-long` (CPU) or `qwen3.5-4b-nli-v2` (GPU) in `config/models.yaml` or `OPENJEV_SUBFOLDER`; the code now says which ones exist |
| `deepeval requires click<8.4.0` warning | `huggingface_hub` needs `click>=8.4.2`; the two can't share one environment | run DeepEval from a separate venv (chapter 17), or ignore the warning if you don't use the `deepeval` CLI |
| real OpenJev labels notes badly or marks most `unclassified` | question-style hypotheses, or a yes/no confidence floor on a multi-class choice | use plain-statement hypotheses (`REJECTION_HYPOTHESES` in `enrich.py`) and `min_confidence_multi` (0.30); compare `accuracy_when_labelled` with `escalated_share` |
| note labelling takes minutes | ~8 s per unique note on CPU | expected for the 0.8B model on CPU; use a GPU, or `mode="llm"` for many unique notes |
| `OpenJev service at http://localhost:8200 is not reachable` / `Connection refused` | `OPENJEV_URL` is set (usually in `.env`) but the server isn't running | start it: `uvicorn services.openjev_server:app --port 8200` (it takes a minute to load), or remove `OPENJEV_URL` from `.env`. With `OPENJEV_BACKEND=auto` the run continues on the keyword fallback and logs a warning; with `http` it stops at start-up |
| OpenJev decisions say `keyword-fallback` | no weights / no `OPENJEV_URL` | `pip install -r requirements-ml.txt` or run the `openjev` service |
| `DEEPEVAL_JUDGE=openjev needs the real OpenJev model` | fallback can't judge faithfulness | start OpenJev, or use `DEEPEVAL_JUDGE=llm` |
| `LLMUnavailable: ... insufficient_quota / spend limit` | the OpenAI project hit its spend limit or has no credit | raise the limit under the project's **Limits** settings, or wait for the reset; meanwhile run with `LLM_BACKEND=mock`. Nothing is retried, and errored cases show `ERR`, not 0 |
| LiteLLM 401 | key missing or wrong | check `OPENAI_API_KEY` in `.env`; it fails at once (`LLMUnavailable`) |
| 429 "rate limit reached for requests per min" | short-term rate limit | retried twice with backoff; then that agent shows `failed` and the run continues |
| JSON validation error after repair | model ignored the schema | use a stronger tier for that agent, or tighten its prompt |
| USD looks wrong | prices in `config/models.yaml` are examples | copy current prices from your provider |

## Graph, HITL and A2A

| Symptom | Cause | Fix |
| --- | --- | --- |
| run never finishes in `--hitl interrupt` | waiting for decisions | answer the prompts; `queue` mode never blocks |
| `resume()` does nothing | wrong `run_id` / new process lost the in-memory checkpointer | resume in the same process, or build the graph with `SqliteSaver` |
| `database is locked` | several heavy runs on SQLite at once | run one at a time, or move to Postgres |
| A2A `Connection refused` | server not running | `uvicorn hiring_intel.a2a:server --port 8100` and set `A2A_BASE_URL` |
| A2A agents `degraded` | concurrent first use of Chroma | already guarded with a lock; restart the A2A server after rebuilding the index |

## Ask tab

| Symptom | Cause | Fix |
| --- | --- | --- |
| a sensible question comes back "out of scope" | the keyword fallback missed the phrasing | use a word from the topic (offer, reject, interviewer, sourcing, SLA), or run real OpenJev, which handles paraphrase better |
| the first Ask on a dataset takes minutes | `enrich` labels the notes on first use; real OpenJev on a CPU is slow | expected once per process; use `OPENJEV_BACKEND=fallback` for demos |
| the answer has a red "Evaluator flags" box | the model wrote a number not in the tables | trust the insight cards' evidence, not the prose; report the question so the prompt can be tightened |
| "This looks like a question about one person" for a group question | the question contains a full name or an ID-shaped token (`C1234`) | rephrase without the name or ID |

## External evaluation

| Symptom | Cause | Fix |
| --- | --- | --- |
| LangSmith 401 | bad key or wrong region | check `LANGSMITH_API_KEY`; EU accounts set `LANGSMITH_ENDPOINT` |
| no traces in LangSmith | tracing off | `LANGSMITH_TRACING=true` in the environment of the process |
| `pytest` collects `test_cases_from_state` | imported name starts with `test` | import it under an alias (chapter 17) |
| DeepEval prompts for login | Confident AI upload | `deepeval login --api-key ...`, or run plain `pytest` to stay local |
| DeepEval LLM judge too expensive | many test cases | `DEEPEVAL_INPUTS=baseline_medium`, or the OpenJev judge |

## Getting help from logs

```bash
LOG_LEVEL=INFO python scripts/run_pipeline.py 2> run.log      # one JSON line per LLM call and decision
sqlite3 data/runs.db "select agent, tier, usd, latency_ms from llm_calls order by ts desc limit 10"
sqlite3 data/runs.db "select task, backend, confidence, escalated from decisions order by ts desc limit 10"
```
