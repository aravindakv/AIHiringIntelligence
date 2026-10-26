import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("EMBEDDINGS", "hashing")
os.environ.setdefault("OPENJEV_BACKEND", "fallback")
os.environ["HI_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
if not (ROOT / "data/raw/baseline_small").exists():
    from scripts.generate_mock_data import generate
    for sc in ["baseline", "healthy"]:
        generate(sc, "small")
