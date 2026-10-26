# App image: Streamlit dashboard + LangGraph pipeline + A2A agent server (no torch -> small image)
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# build data + RAG index at image build time so the container starts ready
RUN python scripts/generate_mock_data.py --scenario baseline --size medium && \
    python scripts/write_knowledge_base.py && python scripts/make_golden_set.py && \
    EMBEDDINGS=hashing python -c "from hiring_intel.rag.retriever import build_index; build_index()"
EXPOSE 8501 8100
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]