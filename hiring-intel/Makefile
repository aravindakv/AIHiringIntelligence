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