from hiring_intel.rag.retriever import build_index, embedding_fn, retrieve

print("indexed", build_index(), "chunks with", embedding_fn()[1], "embeddings")
for q, agent, role in [("offer decline compensation band expected CTC acceptance", "offers", None),
                       ("rejection history stage reason JD mismatch fundamentals diagnosis playbook", "rejection", "Mobile"),
                       ("interviewer load policy interviews per week rebalance shadows", "panel_load", None)]:
    print(agent, [cid for cid, _ in retrieve(q, agent, role)])