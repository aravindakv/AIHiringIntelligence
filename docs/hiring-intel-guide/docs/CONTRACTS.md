# Output contract

Defined in [`hiring_intel/contracts.py`](../hiring_intel/contracts.py). One shape for every agent, so evaluation, HITL, cost tracking, A2A and the dashboard all consume the same thing.

## Required fields (from the brief)

| Field | Type | Filled by | Notes |
| --- | --- | --- | --- |
| `recommendation` | str (≥ 30 chars) | LLM | Must name who / what / when or a target; generic phrases are rejected |
| `evidence` | list[Evidence] (≥ 1) | LLM | `metric`, `value`, `segment`, `source` = data or rag, `ref` = chunk id for rag |
| `confidence_score` | float 0–1 | LLM, then capped by code | cap = min(0.95, 0.5 + 0.1 × evidence items + 0.1 if any rag evidence) |
| `cost_of_insight` | Cost | **code**, never the LLM | model, tier, tokens, USD, latency, simulated?, cached? |
| `alternative` | Alternative | LLM | cheaper/faster option, its trade-off, est. USD |

Extras: `title`, `severity` (low/medium/high), `rebalance_plan` (Panel Load only: list of moves).

## AgentOutput

`agent`, `run_id`, `status` (ok / degraded / failed), `insights[]`, `cost_of_insight`, `compute_ms`, `retrieved_chunks[]`, `error`.

## Other contracts

- `Decision`: an OpenJev decision: task, choice, confidence, per-option probabilities, backend, latency, escalated.
- `ReviewItem`: a HITL queue entry: insight, reasons, status (pending / approved / edited / rejected / auto_published), reviewer, note.
- `EvalResult` (`agents/evaluator.py`): schema_ok, numbers_grounded, refs_valid, judge scores, overall, flags.

## Real example (Offer Insights, comp_crunch input, medium)

```json
{
  "agent": "offers", "status": "ok", "compute_ms": 241,
  "retrieved_chunks": ["offer_timing", "comp_bands"],
  "cost_of_insight": {"model": "gpt-4o", "tier": "large", "input_tokens": 1069, "output_tokens": 300,
                      "usd": 0.005673, "latency_ms": 7240, "simulated": true, "cached": false},
  "insights": [{
    "title": "Offers below 90% of expected CTC are being declined",
    "recommendation": "Comp partner and hiring managers: set an offer floor of 90% of the candidate's expected CTC, with exceptions approved within 48 hours. Offers under that line decline at 72% vs 18% in the 0.9-1.0 band.",
    "evidence": [
      {"metric": "decline_rate", "value": 0.721, "segment": {"ratio_band": "<0.9"}, "source": "data"},
      {"metric": "decline_rate", "value": 0.176, "segment": {"ratio_band": "0.9-1.0"}, "source": "data"},
      {"metric": "offers", "value": 197, "segment": {"ratio_band": "<0.9"}, "source": "data"},
      {"metric": "share_of_declines", "value": 0.861, "segment": {"reason": "compensation"}, "source": "data"},
      {"metric": "benchmark", "value": "Offer timing: candidates who wait more than 7 days ...", "source": "rag", "ref": "offer_timing"}
    ],
    "confidence_score": 0.95, "severity": "high",
    "alternative": {"option": "Small model + rules",
                    "trade_off": "Band math is deterministic; a small model can phrase it. Large model adds value only when causes compete."}
  }]
}
```

Its evaluation: `numbers_grounded 1.0`, `refs_valid true`, judge 4/5/1, overall 0.97. HITL: pending (severity high, OpenJev: changes compensation).
