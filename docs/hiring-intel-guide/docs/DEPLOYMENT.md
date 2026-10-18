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
