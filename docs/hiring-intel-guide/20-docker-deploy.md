# 20 — Docker deployment on your Linux box

**Goal:** the dashboard in your browser at `http://<your-box>:8501`, with the A2A agents and (optionally) the OpenJev model as separate services.

**Time:** 45 minutes.

## Concepts

| Service | Image | Port | Notes |
| --- | --- | --- | --- |
| `app` | `Dockerfile` (no torch, small) | 8501 | Streamlit + pipeline |
| `a2a` | same image | 8100 | A2A agent server; the app calls it at `http://a2a:8100` |
| `openjev` | `Dockerfile.openjev` (CPU torch) | 8200 | profile `openjev`; weights cached in a volume |

- Run history (`runs.db`) and the Chroma index live in the `state` volume, shared by `app` and `a2a`.
- `./data/raw` is mounted, so ATS exports you drop there show up in the dataset picker without a rebuild.
- The Human review tab changes data: never expose it publicly without authentication.

## Step 20.1 — Images and compose

**File:** `Dockerfile`
```dockerfile
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
```

**File:** `Dockerfile.openjev`
```dockerfile
# OpenJev decision service. CPU by default; for GPU use a CUDA base image and the 4b checkpoint.
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/models
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install "transformers>=4.57" fastapi uvicorn pyyaml pydantic httpx
COPY hiring_intel ./hiring_intel
COPY services ./services
COPY config ./config
EXPOSE 8200
CMD ["uvicorn", "services.openjev_server:app", "--host", "0.0.0.0", "--port", "8200"]
```

**File:** `docker-compose.yml`
```yaml
# docker compose up -d --build          -> dashboard on http://<host>:8501
# docker compose --profile openjev up -d -> also run the OpenJev model service
services:
  app:
    build: .
    image: hiring-intel:latest
    ports: ["8501:8501"]
    env_file: .env
    environment:
      HI_DB: /app/data/state/runs.db
      CHROMA_DIR: /app/data/state/chroma
      A2A_BASE_URL: http://a2a:8100
    volumes:
      - state:/app/data/state
      - ./data/raw:/app/data/raw          # drop your own ATS exports here
    depends_on: [a2a]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]
      interval: 30s
      timeout: 5s
      retries: 3

  a2a:
    image: hiring-intel:latest
    build: .
    command: ["uvicorn", "hiring_intel.a2a:server", "--host", "0.0.0.0", "--port", "8100"]
    env_file: .env
    environment:
      HI_DB: /app/data/state/runs.db
      CHROMA_DIR: /app/data/state/chroma
      A2A_PUBLIC_URL: http://a2a:8100
    ports: ["8100:8100"]
    volumes:
      - state:/app/data/state
      - ./data/raw:/app/data/raw
    restart: unless-stopped

  openjev:
    profiles: ["openjev"]
    build:
      context: .
      dockerfile: Dockerfile.openjev
    environment:
      OPENJEV_SUBFOLDER: qwen3.5-0.8b-nli-v5
    ports: ["8200:8200"]
    volumes:
      - models:/models                      # weights cached across restarts
    restart: unless-stopped

volumes:
  state:
  models:
```

## Step 20.2 — Deployment runbook

**File:** `docs/DEPLOYMENT.md`
````markdown
# Deployment: Docker on your Linux machine

Result: the dashboard at `http://<your-box>:8501`, the A2A agents at `:8100` (internal), and optionally the OpenJev model service at `:8200`.

## 1. Prerequisites

```bash
# Ubuntu / Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
docker compose version          # needs Compose v2
```

Hardware: the app needs ~1 GB RAM. OpenJev 0.8b on CPU needs ~4 GB RAM; the 4b checkpoint wants a GPU.

## 2. Get the code and configure

```bash
unzip hiring-intel.zip && cd hiring-intel        # or git clone your repo
cp .env.example .env
```

Edit `.env`:

| Variable | Offline demo | Real models |
| --- | --- | --- |
| `LLM_BACKEND` | `mock` | `litellm` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | empty | your key |
| `EMBEDDINGS` | `hashing` | `minilm` (add `sentence-transformers` to the image) |
| `OPENJEV_BACKEND` | `fallback` | `http` |
| `OPENJEV_URL` | empty | `http://openjev:8200` |

Model names and prices: `config/models.yaml`.

## 3. Data

```bash
python3 scripts/generate_mock_data.py --all       # or copy your ATS exports into data/raw/<name>/
```

Compose mounts `./data/raw` into both containers, so new folders show up in the dashboard's dataset picker without a rebuild. Each folder needs the six CSVs listed in `hiring_intel/ingest.py`.

## 4. Start

```bash
docker compose up -d --build                      # app + a2a
docker compose --profile openjev up -d --build    # also the OpenJev service (first start downloads weights)
docker compose ps
docker compose logs -f app
```

Open `http://<your-box>:8501`, pick a dataset, press **Run now**.

## 5. Reach it from your laptop

| Situation | Do |
| --- | --- |
| Same LAN | `sudo ufw allow 8501/tcp`, open `http://<box-ip>:8501` |
| Remote, just you | `ssh -L 8501:localhost:8501 you@box`, open `http://localhost:8501` |
| Share with others | Put it behind a reverse proxy with HTTPS **and auth** (below) |

The **Human review** tab changes data. Do not expose it publicly without authentication.

Caddy example (`/etc/caddy/Caddyfile`), automatic HTTPS + basic auth:

```text
hiring.example.com {
    basic_auth {
        aravinda <hash from: caddy hash-password>
    }
    reverse_proxy localhost:8501
}
```

## 6. Operate

| Task | Command |
| --- | --- |
| Update after code changes | `docker compose up -d --build` |
| Run the benchmark inside the container | `docker compose exec app python scripts/benchmark.py --quick` |
| Run the golden eval | `docker compose exec app python scripts/run_eval.py --size medium` |
| Back up run history | `docker run --rm -v hiring-intel_state:/s -v $PWD:/b alpine tar czf /b/state.tgz -C /s .` |
| Reset run history | `docker compose down && docker volume rm hiring-intel_state` |
| Health | `curl localhost:8501/_stcore/health`, `curl localhost:8100/health`, `curl localhost:8200/health` |

Containers restart automatically (`restart: unless-stopped`) and come back after a reboot as long as the Docker service is enabled (`sudo systemctl enable docker`).

## 7. GPU for OpenJev (optional)

Use a CUDA base image in `Dockerfile.openjev` (for example `pytorch/pytorch` with CUDA runtime), install the NVIDIA Container Toolkit, set `OPENJEV_SUBFOLDER=qwen3.5-4b-nli-v5`, and add to the `openjev` service:

```yaml
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
```

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| "No datasets in data/raw" | Generate data on the host (step 3); the volume mount hides the image's copy |
| Agents show `degraded` | RAG index empty: it rebuilds on first query; check `CHROMA_DIR` is writable |
| `openjev` restarts | Not enough RAM; use the 0.8b checkpoint or a GPU |
| LLM 401 / 429 | Key missing or rate limited; LiteLLM retries twice, then the agent shows `failed` |
| `database is locked` | Several heavy runs at once on SQLite; run one at a time or move to Postgres |

## Without Docker

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
uvicorn hiring_intel.a2a:server --port 8100 &      # optional
streamlit run app/dashboard.py --server.address 0.0.0.0
```

A systemd unit keeps it running:

```ini
# /etc/systemd/system/hiring-intel.service
[Unit]
Description=Hiring Intelligence dashboard
After=network.target
[Service]
WorkingDirectory=/opt/hiring-intel
EnvironmentFile=/opt/hiring-intel/.env
ExecStart=/opt/hiring-intel/.venv/bin/streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
User=aravinda
[Install]
WantedBy=multi-user.target
```

`sudo systemctl daemon-reload && sudo systemctl enable --now hiring-intel`
````

## Step 20.3 — Deploy

```bash
python3 scripts/generate_mock_data.py --all      # host data, mounted into the containers
docker compose up -d --build
docker compose --profile openjev up -d --build   # optional: real OpenJev
docker compose ps
```

## Verify

<!-- verify:20 -->
```bash
python -c "import yaml; c=yaml.safe_load(open('docker-compose.yml')); print({k: v.get('ports') for k, v in c['services'].items()})"
```

Expected output (captured on a fresh extract, mock backend):

```text
{'app': ['8501:8501'], 'a2a': ['8100:8100'], 'openjev': ['8200:8200']}
```

On your box, after `docker compose up`:

```bash
curl -s localhost:8501/_stcore/health     # ok
curl -s localhost:8100/health              # {"ok":true}
curl -s localhost:8200/health              # with the openjev profile
```

## Checkpoint

- [ ] the dashboard opens from your laptop (LAN, SSH tunnel, or behind Caddy with basic auth)
- [ ] a run started in the UI appears in **Compare inputs**
- [ ] `docker compose restart` keeps your run history
