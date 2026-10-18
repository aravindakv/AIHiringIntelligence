# Demo video script (8–9 minutes)

| Time | Show | Say |
| --- | --- | --- |
| 0:00 | README diagram | The problem; 8 agents; pandas computes, OpenJev decides, LLM interprets, evaluator checks, human approves |
| 1:00 | Dashboard → Run now on `comp_crunch_medium` | Five agents in parallel; routing picks small or large per agent |
| 2:00 | Overview tab | Funnel, top actions with confidence and review status |
| 2:45 | Insights → Offers card | **One decision end to end:** feature row (decline 72% vs 18%) → retrieved chunks → routed to large via OpenJev tie-break → insight + evidence + confidence + cost + alternative |
| 4:00 | Evaluation tab | Grounding 1.0, refs valid, judge scores; golden set 18/18 medium/large, 15/18 small and why |
| 5:00 | Sidebar: fault "fabricate numbers" → Run | Evaluator flags ungrounded numbers, retries on large, score recovers |
| 5:45 | Human review tab | Why each item is here; approve one, edit one, reject one; approved text becomes `past_insight` |
| 6:45 | Cost & latency tab | USD per agent and tier, critical-path latency by stage, OpenJev decisions |
| 7:30 | Compare inputs tab | Cost grows with input size when an LLM labels notes; flat with OpenJev; the four levers |
| 8:15 | Terminal: `--transport a2a` + curl of an Agent Card | Agents as independent A2A services |
| 8:45 | Close | What I would change next |
