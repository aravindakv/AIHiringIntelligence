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
| agents show `degraded` | RAG returned nothing | `python -c "from hiring_intel.rag.retriever import build_index; print(build_index())"`; check `CHROMA_DIR` is writable |
| MiniLM download fails | no access to Hugging Face | `EMBEDDINGS=hashing` (offline), or pre-download the model |
| `Couldn't instantiate the backend tokenizer… need sentencepiece or tiktoken` when loading OpenJev | the subfolder doesn't exist in the repo (the model card lists `-v5` checkpoints that aren't published) | use `qwen3.5-0.8b-nli-v2s-long` (CPU) or `qwen3.5-4b-nli-v2` (GPU) in `config/models.yaml` or `OPENJEV_SUBFOLDER`; the code now says which ones exist |
| `deepeval requires click<8.4.0` warning | `huggingface_hub` needs `click>=8.4.2`; the two can't share one environment | run DeepEval from a separate venv (chapter 17), or ignore the warning if you don't use the `deepeval` CLI |
| real OpenJev labels notes badly or marks most `unclassified` | question-style hypotheses, or a yes/no confidence floor on a multi-class choice | use plain-statement hypotheses (`REJECTION_HYPOTHESES` in `enrich.py`) and `min_confidence_multi` (0.30); compare `accuracy_when_labelled` with `escalated_share` |
| note labelling takes minutes | ~8 s per unique note on CPU | expected for the 0.8B model on CPU; use a GPU, or `mode="llm"` for many unique notes |
| OpenJev decisions say `keyword-fallback` | no weights / no `OPENJEV_URL` | `pip install -r requirements-ml.txt` or run the `openjev` service |
| `DEEPEVAL_JUDGE=openjev needs the real OpenJev model` | fallback can't judge faithfulness | start OpenJev, or use `DEEPEVAL_JUDGE=llm` |
| LiteLLM 401 / 429 | key missing or rate limited | check `.env`; LiteLLM retries twice, then the agent shows `failed` |
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
