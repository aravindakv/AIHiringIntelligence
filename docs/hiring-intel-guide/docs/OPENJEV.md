# OpenJev for simple decisions

Code: [`hiring_intel/openjev.py`](../hiring_intel/openjev.py), service: [`services/openjev_server.py`](../services/openjev_server.py).

## What it is

OpenJev ([huggingface.co/AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev), MIT licence per its model card) is Qwen3.5 fine-tuned as an NLI **cross-encoder**. Given a premise and a hypothesis it returns P(contradiction), P(entailment), P(neutral). To decide between options you score one hypothesis per option and take the highest entailment. It does not generate text, it is deterministic, and its option scores do not depend on option order.

Checkpoints (subfolders of the repo): `qwen3.5-4b-nli-v5` (recommended for typed decisions), `qwen3.5-2b-nli-v5`, `qwen3.5-0.8b-nli-v5` (CPU-friendly; the default here).

## Where this system uses it

| Decision | Premise | Options | Why not the LLM |
| --- | --- | --- | --- |
| Label rejection notes | interviewer's free-text note | 7 reasons | Thousands of items per run; closed label set |
| Label decline notes | recruiter's note | comp / competing / timing / location | Same |
| Routing tie-break | task profile text | yes / no "needs causal reasoning?" | Runs before every borderline agent; must be ~free |
| HITL sensitivity | the recommendation | yes / no "changes comp, headcount or workload?" | Gate on every insight |
| Actionability pre-check | the recommendation | yes / no "specific and actionable?" | Cheap signal for the evaluator |

Every decision is logged to the `decisions` table with its backend, confidence and whether it **escalated**: below `openjev.min_confidence` (0.55) the system does not trust it. Note labels become `unclassified`, routing falls back to a small-LLM call, and HITL marks the item "sensitivity unclear" for a human.

## Backends

| Backend | When | Set up |
| --- | --- | --- |
| `http` | Docker Compose `openjev` service (recommended) | `docker compose --profile openjev up -d`, `OPENJEV_URL=http://openjev:8200` |
| `local` | Weights loaded in-process | `pip install -r requirements-ml.txt` |
| `fallback` | No weights available (offline CI, this build) | none; keyword scorer with the same interface |
| `auto` (default) | http if `OPENJEV_URL` set, else local, else fallback | — |

The committed benchmark used **fallback** because the build machine could not reach Hugging Face. Re-run the benchmark with the real model and replace the numbers.

## Loading the model directly

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sub = "qwen3.5-0.8b-nli-v5"
tok = AutoTokenizer.from_pretrained("AlexWortega/openjev", subfolder=sub)
model = AutoModelForSequenceClassification.from_pretrained("AlexWortega/openjev", subfolder=sub)
text = model.config.nli_template.format(premise="...", hypothesis="...")
```

`LocalOpenJev` uses `config.nli_template` when present and finds the entailment index from `config.id2label` (model card order: contradiction, entailment, neutral).

## Caveats to put in your eval notes

- **Calibration:** probabilities are not calibrated for your labels. Tune `min_confidence` on 50 hand-labelled notes.
- **Prompt injection:** the model card reports that one adversarial line sharply lowers accuracy. Never use OpenJev alone as a safety gate; pair it with deterministic checks (the HITL policy also uses severity, confidence and eval flags).
- **Benchmark contamination:** the v5 checkpoints were trained on test splits of public benchmarks (per the card), so public benchmark scores are not evidence. Your own labelled notes are.
- **Licence:** the Hugging Face weights are MIT per the card. A separate Docker Hub image (`ai/openjev`) states CC BY-NC 4.0 for its source weights; check which artefact you deploy.
- **Cost model:** $0 API, but CPU/GPU time. Measure notes/second on your box (`/decide` endpoint) and add it to the latency table.
