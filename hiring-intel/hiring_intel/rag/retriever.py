"""RAG: Chroma vector store + per-agent metadata filters + similarity threshold + cache.

Embeddings (env EMBEDDINGS): 'minilm' = sentence-transformers all-MiniLM-L6-v2 (download once),
'hashing' = sklearn HashingVectorizer (offline, no download). 'auto' tries MiniLM then falls back.
"""
from __future__ import annotations

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
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
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
    ef, _ = embedding_fn()
    client = _client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for f in sorted(pathlib.Path(KB_DIR).glob("*.md")):
        _, fm, body = f.read_text().split("---", 2)
        meta = yaml.safe_load(fm)
        ids.append(f.stem)
        docs.append(body.strip())
        metas.append(_meta(meta))
    for doc_id, text, meta in extra_docs or []:
        ids.append(doc_id); docs.append(text); metas.append(_meta(meta))
    col.add(ids=ids, documents=docs, metadatas=metas)
    retrieve.cache_clear()
    return len(ids)


def _meta(meta: dict) -> dict:
    agents = meta.get("agents", [])
    return {"doc_type": meta.get("doc_type", "note"), "role": meta.get("role", "all"),
            "agents": ",".join(agents), **{f"agent_{a}": True for a in agents}}


@lru_cache(maxsize=1)
def _collection():
    ef, _ = embedding_fn()
    client = _client()
    if COLLECTION not in [c.name for c in client.list_collections()]:   # first start on an empty volume
        build_index()
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
    except Exception:
        return tuple()
    cutoff = float(os.getenv("RAG_MAX_DISTANCE", DEFAULT_MAX_DISTANCE[embedding_fn()[1]]))
    return tuple((i, d) for i, d, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])
                 if dist <= cutoff)