# Cost and latency: definitions

Implemented in `hiring_intel/llm.py` (per call) and `hiring_intel/report.py` (per agent, per run). Every number on the dashboard follows these definitions.

## Cost

| Term | Definition |
| --- | --- |
| **Call cost (USD)** | `input_tokens × input_price_per_1M / 1,000,000 + output_tokens × output_price_per_1M / 1,000,000` |
| Prices | `config/models.yaml` per tier (falls back to LiteLLM's price table for unknown models). Examples only: update from your provider before quoting USD. |
| Tokens | Real runs: the provider's `usage` counts. Mock runs: estimated as `characters / 4`. |
| **Agent cost** | Sum of the agent's call costs, including JSON-repair retries and its retry on the large tier. |
| **Run cost** | Sum of every call in the run: insight agents, router LLM fallbacks, note labelling (if done by LLM), judge, optimizer. |
| Cache hit | $0 and ~2 ms; still logged so you can count hits. |
| **OpenJev decision** | $0 API cost. It uses local CPU/GPU time, which counts toward latency. Hardware cost is outside the USD figure; note it if you self-host on paid GPUs. |
| **Cost of insight** | The `cost_of_insight` field on each `AgentOutput`: model, tier, tokens, USD, latency for the call that produced it. |
| **LLM-equivalent cost** | For OpenJev labelling, what the same job would cost on the small LLM (`enrich.llm_equivalent_usd`), so the saving is visible. |

```latex
\text{USD}_{call} = \frac{T_{in}\cdot P_{in} + T_{out}\cdot P_{out}}{10^{6}}
```

## Latency

| Term | Definition |
| --- | --- |
| **Call latency** | Request sent to parsed, validated response. Real runs: wall clock. Mock runs: **simulated** as `base_ms + ms_per_out_token × out + ms_per_1k_in × in/1000`, ×0.85–1.25 deterministic jitter, per tier. |
| **Compute latency** | Local work: pandas features, retrieval, OpenJev, validation. Always measured. |
| **Agent latency** | compute + its LLM latency. |
| **End-to-end (E2E) latency** | Critical path through the graph: `ingest + enrich + max(agent) + evaluate + max(retry) + human_review + optimize`. Parallel stages count only their slowest branch. LLM note labelling assumes 8 concurrent requests. |
| **Wall time** | Measured clock time of the process. In mock mode it excludes simulated LLM waits, so it shows pure compute. |
| p50 / p95 | Per agent across runs, from the `llm_calls` table. |
| Human review time | Not counted in E2E. With `hitl_mode=queue` the run completes and review is async; with `interrupt` the run pauses until resumed. |

Simulated latency parameters (from `config/models.yaml`):

| Tier | base | per output token | per 1k input tokens |
| --- | --- | --- | --- |
| small | 350 ms | 9 ms | 40 ms |
| large | 700 ms | 22 ms | 90 ms |

## Where to see them

- Dashboard → **Cost & latency**: USD per agent and tier, critical-path latency by stage, routing decisions, OpenJev decisions.
- Dashboard → **Compare inputs**: run cost and E2E latency per input size and variant.
- `data/runs.db` tables: `llm_calls`, `timings`, `decisions`, `routing`, `runs`.
- `docs/OPTIMIZATION_REPORT.md` for the before/after numbers.
