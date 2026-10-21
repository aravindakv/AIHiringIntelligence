# 06 — RAG: knowledge base and retriever

**Goal:** a small, tagged knowledge base in Chroma that each agent queries with its own filters, so benchmarks are retrieved rather than invented.

**Time:** 40 minutes.

## Concepts

- **What goes in.** Sourcing benchmarks per role, the SLA policy, the interviewer load policy, past rejection and offer history, playbooks, and past insights. Approved insights from human review are added automatically (chapter 11), so the system learns what people accepted.
- **Every benchmark here is illustrative.** Replace them with your organisation's history before you trust a recommendation that cites one.
- **Per-agent metadata filters.** Each chunk carries `agent_<name>: true` flags and a `role`. The Offers agent never sees sourcing benchmarks.
- **Embeddings.** `EMBEDDINGS=minilm` uses sentence-transformers all-MiniLM-L6-v2 (one download). `hashing` is an offline vectoriser with no download. `auto` tries MiniLM and falls back. The index records which embedding built it. If that changes (you installed `requirements-ml.txt` after building offline), it rebuilds on first use and keeps human-approved insights. Before this check, every query failed with "expecting embedding with dimension of 512, got 384" and every agent showed `degraded`.
- **Distance cutoff.** Chunks beyond a cosine-distance threshold are dropped (0.6 for MiniLM, 0.9 for hashing). Tune it on 10 hand-labelled queries; the learning path has the exercise.
- **Cache.** `lru_cache` on `retrieve()`: repeated queries skip embedding and search.
- **Traceability.** Agents cite chunks as `source: "rag"` evidence with the chunk id as `ref`, and the evaluator checks that the id was really retrieved.

## Step 6.1 — Write the knowledge base

**File:** `scripts/write_knowledge_base.py`
```python
"""Write the RAG knowledge base (markdown + front matter) into data/knowledge/.

All benchmark figures here are ILLUSTRATIVE, authored for this capstone. Replace them with your
organisation's real history before trusting any recommendation that cites them.
"""
from pathlib import Path

DOCS = {
    "bench_sourcing_backend": ("benchmark", ["sourcing"], "Backend",
        "Sourcing benchmark (illustrative) for Backend roles. Referral-to-hire conversion is typically 8-12%; "
        "job boards convert at 2-4%. Referral candidates reach onsite about twice as often. Agency fees run ~15% "
        "of first-year CTC, so agencies should be reserved for niche or senior searches."),
    "bench_sourcing_data": ("benchmark", ["sourcing", "rejection"], "Data",
        "Sourcing benchmark (illustrative) for Data roles. Agency candidates historically fail SQL and statistics "
        "fundamentals more often than referrals. Require a take-home or SQL screen before agency candidates reach tech_1."),
    "bench_sourcing_mobile": ("benchmark", ["sourcing"], "Mobile",
        "Sourcing benchmark (illustrative) for Mobile roles. LinkedIn and referrals give the best native "
        "Android/iOS pipelines; careers-page applicants are often web developers applying to mobile postings."),
    "bench_sourcing_sre": ("benchmark", ["sourcing"], "SRE",
        "Sourcing benchmark (illustrative) for SRE roles. Expect 60-90 day time-to-fill. Referral and community "
        "channels (meetups, open source) outperform job boards for on-call experienced SREs."),
    "bench_sourcing_all": ("benchmark", ["sourcing"], "all",
        "General sourcing guidance (illustrative). Cut a channel when its hire rate is under half of the best channel "
        "for the same role over at least 50 applicants; re-test after one quarter. Shift recruiter hours, not just spend."),
    "sla_policy": ("sla", ["pipeline", "panel_load"], "all",
        "Stage SLA policy: applied 6 days, screen 6, tech_1 7, tech_2 7, onsite 7, offer 5. A stage is in breach when "
        "more than 30% of candidates exceed its SLA. Requisitions open over 45 days (60 for senior) need an EM review."),
    "load_policy": ("policy", ["panel_load"], "all",
        "Interviewer load policy: no more than 4 interviews per person per week, and no one above 2x the team median "
        "for two weeks running. Rebalance within the same team and level; pair new interviewers with shadows first."),
    "rej_hist_q1": ("rejection_history", ["rejection"], "Mobile",
        "Rejection history Q1 2026: Mobile screen rejection was 35%, mostly 'experience gap'. A JD rewrite in 2025 "
        "that listed must-have native skills cut screen rejections by a third within one quarter."),
    "rej_hist_data": ("rejection_history", ["rejection"], "Data",
        "Rejection history Q1 2026: Data tech_1 failures concentrated in candidates from two agencies; adding a SQL "
        "pre-screen reduced tech_1 failure from 62% to 38%."),
    "rej_playbook": ("playbook", ["rejection"], "all",
        "Diagnosis playbook: high screen rejection with 'JD mismatch' means fix the JD or sourcing filters; high "
        "technical rejection in one source means fix that source; high technical rejection across all sources means "
        "recalibrate the bar or the interviewers (run a calibration session with scored recordings)."),
    "offer_hist": ("offer_history", ["offers"], "all",
        "Offer outcome history (illustrative): when the offer is below 90% of the candidate's expected CTC, acceptance "
        "falls to roughly one in four. Offers at or above 95% of expectation are accepted about 80% of the time."),
    "comp_bands": ("offer_history", ["offers"], "all",
        "Compensation band guidance: approve up to 10% above band midpoint for senior engineers with a competing offer. "
        "Offer approvals should complete within 48 hours; every extra week of delay lowers acceptance."),
    "offer_timing": ("offer_history", ["offers", "pipeline"], "all",
        "Offer timing: candidates who wait more than 7 days for an offer decision are twice as likely to decline for "
        "'timing' or a competing offer. Pre-approve bands before the onsite."),
    "pipeline_ttf": ("benchmark", ["pipeline"], "all",
        "Time-to-fill benchmark (illustrative): Backend/Frontend 35-45 days, Data 45 days, Mobile 40 days, SRE 60-90 days. "
        "Onsite scheduling is the most common bottleneck; block recurring onsite slots per week."),
    "past_insight_panel": ("past_insight", ["panel_load"], "all",
        "Past insight (accepted): two interviewers carried 35% of loops in Q1. Moving 3 slots/week each to trained "
        "shadows cut their load to policy within 3 weeks and reduced onsite scheduling delay."),
    "past_insight_sourcing": ("past_insight", ["sourcing"], "Backend",
        "Past insight (accepted): Backend referral bonus raised in Q1; referral share rose from 12% to 20% of applicants."),
    "interview_calibration": ("playbook", ["rejection", "panel_load"], "all",
        "Interviewer calibration: run a monthly calibration with 3 scored recordings per stage; flag interviewers whose "
        "pass rate deviates more than 20 points from the panel average."),
}


def main() -> None:
    out = Path("data/knowledge")
    out.mkdir(parents=True, exist_ok=True)
    for name, (doc_type, agents, role, body) in DOCS.items():
        (out / f"{name}.md").write_text(
            f"---\ndoc_type: {doc_type}\nagents: [{', '.join(agents)}]\nrole: {role}\nperiod: 2026-Q1\n---\n"
            f"# {name.replace('_', ' ').title()}\n{body}\n")
    print(f"wrote {len(DOCS)} docs to {out}")


if __name__ == "__main__":
    main()
```

## Step 6.2 — The retriever

**File:** `hiring_intel/rag/retriever.py`
```python
"""RAG: Chroma vector store + per-agent metadata filters + similarity threshold + cache.

Embeddings (env EMBEDDINGS): 'minilm' = sentence-transformers all-MiniLM-L6-v2 (download once),
'hashing' = sklearn HashingVectorizer (offline, no download). 'auto' tries MiniLM then falls back.
"""
from __future__ import annotations

import logging
import os
import pathlib
import threading
from functools import lru_cache

import chromadb
import numpy as np
import yaml
from chromadb import Documents, EmbeddingFunction, Embeddings

from hiring_intel.llm import ROOT

_LOCK = threading.RLock()       # Chroma client is not safe for concurrent first use across threads
KB_DIR = ROOT / "data/knowledge"
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "data/chroma"))
COLLECTION = "hiring_kb"
log = logging.getLogger("hiring_intel.rag")
# cosine-distance cutoff per embedding backend; tune on labelled queries (scripts/tune_rag.py)
DEFAULT_MAX_DISTANCE = {"minilm": 0.6, "hashing": 0.9}


class HashingEmbedding(EmbeddingFunction):
    """Stateless, offline embedding: word + bigram hashing, L2 normalised, 512 dims."""

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.v = HashingVectorizer(n_features=512, ngram_range=(1, 2), alternate_sign=False,
                                   norm="l2", stop_words="english")

    def __call__(self, input: Documents) -> Embeddings:
        return [np.asarray(r, dtype=np.float32) for r in self.v.transform(list(input)).toarray()]

    @staticmethod
    def name() -> str:
        return "hashing-512"


@lru_cache(maxsize=1)
def embedding_fn():
    mode = os.getenv("EMBEDDINGS", "auto")
    if mode in ("minilm", "auto"):
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            ef(["warmup"])
            return ef, "minilm"
        except Exception:
            if mode == "minilm":
                raise
    return HashingEmbedding(), "hashing"


def _client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_index(extra_docs: list[tuple[str, str, dict]] | None = None) -> int:
    ef, ef_name = embedding_fn()
    client = _client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    # the embedding name is stored so a later switch (hashing -> MiniLM) triggers a rebuild, not a
    # dimension error on every query
    col = client.create_collection(COLLECTION, embedding_function=ef,
                                   metadata={"hnsw:space": "cosine", "embedding": ef_name})
    ids, docs, metas = [], [], []
    for f in sorted(pathlib.Path(KB_DIR).glob("*.md")):
        _, fm, body = f.read_text().split("---", 2)
        meta = yaml.safe_load(fm)
        ids.append(f.stem)
        docs.append(body.strip())
        metas.append(_meta(meta))
    for doc_id, text, meta in extra_docs or []:
        if doc_id not in ids:                      # knowledge-base files win over a preserved copy
            ids.append(doc_id); docs.append(text); metas.append(_meta(meta))
    col.add(ids=ids, documents=docs, metadatas=metas)
    retrieve.cache_clear()
    return len(ids)


def _meta(meta: dict) -> dict:
    agents = meta.get("agents", [])
    return {"doc_type": meta.get("doc_type", "note"), "role": meta.get("role", "all"),
            "agents": ",".join(agents), **{f"agent_{a}": True for a in agents}}


def _past_insights(client) -> list[tuple[str, str, dict]]:
    """Human-approved insights live only in the index; keep them across a rebuild."""
    try:
        got = client.get_collection(COLLECTION).get(where={"doc_type": "past_insight"})
    except Exception:
        return []
    return [(i, d, {"doc_type": "past_insight", "role": m.get("role", "all"),
                    "agents": [a for a in m.get("agents", "").split(",") if a]})
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]


@lru_cache(maxsize=1)
def _collection():
    ef, ef_name = embedding_fn()
    client = _client()
    names = [c.name for c in client.list_collections()]
    if COLLECTION not in names:                                   # first start on an empty volume
        build_index()
    else:
        built_with = (client.get_collection(COLLECTION).metadata or {}).get("embedding")
        if built_with != ef_name:
            log.warning("RAG index was built with %s embeddings, now using %s: rebuilding", built_with, ef_name)
            build_index(_past_insights(client))
    return client.get_collection(COLLECTION, embedding_function=ef)


def add_past_insight(doc_id: str, text: str, agent: str, role: str = "all") -> None:
    """Human-approved insights are written back so the next run can cite them."""
    with _LOCK:
        _collection().upsert(ids=[doc_id], documents=[text],
                             metadatas=[_meta({"doc_type": "past_insight", "agents": [agent], "role": role})])
    retrieve.cache_clear()


@lru_cache(maxsize=512)    # optimization lever: repeated queries skip embedding + search
def retrieve(query: str, agent: str, role: str | None = None, k: int = 4) -> tuple[tuple[str, str], ...]:
    where: dict = {f"agent_{agent}": True}
    if role:
        where = {"$and": [where, {"role": {"$in": [role, "all"]}}]}
    try:
        with _LOCK:
            res = _collection().query(query_texts=[query], n_results=k, where=where)
    except Exception as e:                       # agents carry on as "degraded"; say why in the log
        log.warning("RAG query failed for %s: %s: %s", agent, type(e).__name__, e)
        return tuple()
    cutoff = float(os.getenv("RAG_MAX_DISTANCE", DEFAULT_MAX_DISTANCE[embedding_fn()[1]]))
    return tuple((i, d) for i, d, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])
                 if dist <= cutoff)
```

## Verify

<!-- verify:06 -->
```bash
python scripts/write_knowledge_base.py
python - <<'EOF'
from hiring_intel.rag.retriever import build_index, retrieve, embedding_fn
print("indexed", build_index(), "chunks with", embedding_fn()[1], "embeddings")
for q, agent, role in [("offer decline compensation band expected CTC acceptance", "offers", None),
                       ("rejection history stage reason JD mismatch fundamentals diagnosis playbook", "rejection", "Mobile"),
                       ("interviewer load policy interviews per week rebalance shadows", "panel_load", None)]:
    print(agent, [cid for cid, _ in retrieve(q, agent, role)])
EOF
```

Expected output (captured on a fresh extract, mock backend):

```text
wrote 17 docs to data/knowledge
indexed 17 chunks with hashing embeddings
offers ['comp_bands', 'offer_hist', 'offer_timing']
rejection ['rej_playbook', 'rej_hist_q1', 'interview_calibration']
panel_load ['load_policy', 'past_insight_panel', 'sla_policy']
```

With MiniLM the ids may differ in order; the filters guarantee that each agent only sees its own documents.

## Checkpoint

- [ ] 17 chunks indexed
- [ ] each agent retrieves only documents tagged for it
