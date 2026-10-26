import numpy as np
from chromadb import EmbeddingFunction

from hiring_intel.rag import retriever as R


class Fake384(EmbeddingFunction):
    """Stands in for MiniLM (384 dims) without a download."""
    def __init__(self):
        pass

    def __call__(self, input):
        return [np.ones(384, dtype=np.float32) / 20 for _ in input]

    @staticmethod
    def name():
        return "fake-384"


def test_index_rebuilds_when_embeddings_change(tmp_path, monkeypatch):
    """Index built offline with hashing (512 dims), then MiniLM installed (384 dims): every query used to
    fail with a dimension error and every agent showed 'degraded'. Approved insights must survive."""
    monkeypatch.setattr(R, "CHROMA_DIR", str(tmp_path))
    monkeypatch.setattr(R, "embedding_fn", lambda: (R.HashingEmbedding(), "hashing"))
    R._collection.cache_clear(); R.retrieve.cache_clear()
    R.build_index()
    R.add_past_insight("approved_1", "Referral hires converted well last quarter", "sourcing")

    monkeypatch.setattr(R, "embedding_fn", lambda: (Fake384(), "minilm"))
    monkeypatch.setitem(R.DEFAULT_MAX_DISTANCE, "minilm", 2.0)
    R._collection.cache_clear(); R.retrieve.cache_clear()
    ids = [c for c, _ in R.retrieve("referral hire rate", "sourcing", None, 10)]
    assert ids and "approved_1" in ids
    R._collection.cache_clear(); R.retrieve.cache_clear()