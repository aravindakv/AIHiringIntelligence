# Reflection: design choices and why (one page)

*Draft for you to rewrite in your own words; the brief grades your reasoning, not mine.*

**Compute first, interpret second.** The biggest decision was to keep counting out of the LLM. Pandas produces every number; the model reads small feature tables and recommends. That one rule made grounding checkable (every evidence number can be found in the features), shrank prompts to a few thousand tokens regardless of input size, and made the rule-based "alternative" honest: it really is the same numbers without narrative.

**LangGraph over n8n.** I needed parallel fan-out, a conditional retry edge and a pause for human approval. LangGraph gives all three in code I can test with pytest. n8n would have been faster to wire but harder to version and test.

**Three tiers of decision-maker.** Rules decide what rules can (thresholds, routing scores). OpenJev decides closed-set questions (labels, yes/no) at $0 API cost. The LLM handles interpretation and wording. The benchmark showed note labelling was the largest cost driver; moving it to OpenJev cut run cost by 81% and made cost flat as input grows.

**A second AI checking the first, plus a human.** The Evaluation Agent catches invented numbers mechanically and retries on a larger model; the fault-injection test proves it. The HITL gate sends anything touching pay, headcount or workload to a person. Trust comes from those two together, not from a better prompt.

**What the golden set taught me.** It caught a compression bug I would have shipped: trimming the panel table changed the median and hid overload. It also showed the agents raise false positives on small inputs. Both would have reached an EM as confident-looking cards.

**What I would change with more time.** Validate the judge against human scores; tune HITL to reduce review fatigue; measure real OpenJev accuracy on real notes; move from SQLite to Postgres and add tracing.

**Would I act on these insights?** On medium and large inputs, yes for sourcing, offers and pipeline; I would still review rejection root causes, because "JD vs interview" is inferred, not tested.
