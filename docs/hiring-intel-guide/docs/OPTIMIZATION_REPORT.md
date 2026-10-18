# Optimization report: before / after

Generated from `data/eval/benchmark.csv` (80 runs: 5 scenarios × 3 sizes + one xlarge input, × 5 variants). Each variant adds one lever to the previous one. Mock backend: costs use example prices; latency is simulated (see [COST_AND_LATENCY.md](COST_AND_LATENCY.md)). Re-run `python scripts/benchmark.py` with `LLM_BACKEND=litellm` to replace these with real measurements.

## Headline: baseline scenario, large input (25k candidates, 14.7k notes to label)

| Variant | Lever added | Run cost (USD) | vs previous | E2E latency | Input tokens | Golden pass |
| --- | --- | --- | --- | --- | --- | --- |
| all_large | none: everything on the large model, LLM labels notes | 0.1304 | — | 101.2 s | 270,845 | 18/18 |
| routed | **A. model routing** | 0.0963 | −26% | 101.1 s | 270,773 | — |
| routed_compressed | **B. context compression** | 0.0874 | −9% | 98.4 s | 262,966 | 15/18 before fix, 18/18 after |
| optimized | **C. OpenJev for simple decisions** | 0.0162 | −81% | 16.9 s | 6,794 | 18/18 |
| optimized_cached | **D. response cache (re-run)** | 0.0003 | −98% | 3.4 s | 6,781 | 18/18 |

Overall: **$0.1304 → $0.0162 per run (−88%) and 101 s → 17 s E2E** with no golden-set regression on medium and large inputs.

## How cost scales with input size (baseline scenario)

| Input | Candidates | all_large | routed | routed_compressed | optimized |
| --- | --- | --- | --- | --- | --- |
| small | 1.2k | $0.0490 | $0.0243 | $0.0195 | $0.0159 |
| medium | 6.3k | $0.0680 | $0.0421 | $0.0339 | $0.0161 |
| large | 25k | $0.1304 | $0.0963 | $0.0874 | $0.0162 |
| xlarge | 84k | $0.3205 | $0.2623 | $0.2518 | $0.0163 |

With an LLM labelling notes, cost grows linearly with the number of notes. With OpenJev the run cost is flat: agents see aggregated features whose size does not depend on candidate count.

## Per lever

**A. Routing (−26% on large input, −50% on small).** Panel Load, Pipeline Health and Sourcing go to the small model on rule scores; Rejection and Offers go large through an OpenJev tie-break ("does this need causal reasoning?"). Golden score unchanged on medium/large.

**B. Compression (−9% of run cost here; insight-agent input tokens −28% on small and −56% on large inputs: 7,933 → 5,697 and 13,843 → 6,074).** Run cost moves less because, in this variant, note labelling still dominates. Top-N rows, pruned columns, `top_k` 2. The golden set caught a regression: trimming the panel table to the top and bottom 8 interviewers distorted the median, so Panel Load stopped flagging overload (15/18). Fix: compute the panel median before trimming and pass it as a column. After the fix: 18/18. A second issue surfaced when the guide's chapter outputs were compared: keeping only the top 8 rows truncated the overloaded list ("8 over policy" where the full table showed 10). Compressed tables now keep every over-policy row.

**C. OpenJev (−81%).** Labelling 14.7k free-text notes moved from 295 batched small-LLM calls to local NLI decisions. It also removed ~81 s of simulated E2E latency. In this build OpenJev ran as the keyword fallback (no Hugging Face access); label accuracy vs planted truth was 0.92 with the 3 deliberately ambiguous note templates (about 8% of notes) left `unclassified` instead of guessed. Measure the real model's accuracy and CPU time before you quote this lever.

**D. Cache (−98% on repeats).** Identical prompt + model returns the stored response. It helps re-runs on unchanged data and the demo; it does nothing on fresh data, so it is reported separately.

## What I would not do

- Downgrade Rejection/Offers to the small model just because the mock judge scores them 0.97: the mock judge cannot see reasoning quality. Decide with a real judge and human ratings.
- Cache across data refreshes: the key includes the features text, so this is safe, but a TTL is still wise.
