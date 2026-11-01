"""Shared config and helpers for the pgvector vs Qdrant hybrid search benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "docs"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
INGEST_STATS_PATH = ARTIFACTS_DIR / "ingest_stats.json"
INDEX_STATS_PATH = ARTIFACTS_DIR / "index_stats.json"
CATEGORY_MANIFEST = DOCS_DIR / "categories.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

PG_DSN = os.getenv("PG_DSN", "postgresql://rag:rag@localhost:5432/rag")
PG_TABLE = "policy_chunks"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "1") == "1"
# Set QDRANT_LOCAL_PATH to run qdrant-client's in-process mode instead of a server.
# Only useful for smoke-testing the code path; its timings are NOT comparable.
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH")
QDRANT_COLLECTION = "policy_chunks"

# HNSW parameters kept identical on both engines.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 64

# Offline smoke-test escape hatch: EMBED_BACKEND=hash uses feature hashing instead
# of MiniLM. Never use it for reported numbers.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "minilm")


class _HashEmbedder:
    """Deterministic bag-of-words feature hashing into 384 dims (smoke tests only)."""

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % EMBED_DIM] += 1.0 if (h >> 32) & 1 else -1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out /= np.where(norms == 0, 1, norms)
        return out


def load_embedder():
    if EMBED_BACKEND == "hash":
        print("WARNING: EMBED_BACKEND=hash - smoke-test embeddings, not MiniLM.")
        return _HashEmbedder()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=os.getenv("EMBED_DEVICE", "cpu"))


def get_qdrant_client():
    from qdrant_client import QdrantClient

    if QDRANT_LOCAL_PATH:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=QDRANT_PREFER_GRPC,
        timeout=120,
    )


def pg_connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(PG_DSN, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load_chunks() -> tuple[list[dict], np.ndarray]:
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        raise SystemExit("No ingested chunks found - run `python ingest.py` first.")
    with CHUNKS_PATH.open() as f:
        chunks = [json.loads(line) for line in f]
    emb = np.load(EMBEDDINGS_PATH)
    assert len(chunks) == emb.shape[0], "chunks/embeddings out of sync - re-run ingest.py"
    return chunks, emb


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

# --------------------------------------------------------------------------- hybrid / scale
# Named vectors in the Qdrant collection.
QDRANT_DENSE = "dense"
QDRANT_SPARSE = "bm25"

# BM25 parameters for Qdrant's sparse branch (Lucene / Elasticsearch defaults).
BM25_K1 = float(os.getenv("BM25_K1", "1.2"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# pgvector's HNSW build slows down sharply once the graph no longer fits in
# maintenance_work_mem (default 64MB), which happens around ~40k x 384-d rows.
PG_MAINTENANCE_WORK_MEM = os.getenv("PG_MAINTENANCE_WORK_MEM", "1GB")

# Bump when the on-disk index layout or replica scheme changes so benchmark.py reindexes.
INDEX_SCHEMA = 2


def corpus_fingerprint() -> str:
    """Identifies the ingested artifacts an index was built from (size + mtime)."""
    parts = []
    for p in (CHUNKS_PATH, EMBEDDINGS_PATH):
        st = p.stat()
        parts.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=8).hexdigest()


def scale_corpus(chunks: list[dict], emb: np.ndarray, factor: int, noise: float = 0.01,
                 seed: int = 0) -> tuple[list[dict], np.ndarray]:
    """Multiply the corpus `factor` times with synthetic chunks.

    Replica 0 is the original corpus, unchanged. Each replica r >= 1 of chunk i is
    blended with a random partner j from the SAME category, with weight w ~ U(0.5, 0.9):
      embedding = normalise(w*e_i + (1-w)*e_j + N(0, noise))
      text      = first w of i's words + the rest from j's words
    It keeps i's metadata (category, filename, page) and gets a new id/doc_id, so the
    category filter keeps the same ~10% selectivity at every scale.

    Why blends and not noisy copies: N near-identical copies of each chunk fill each
    HNSW node's neighbour list (2*m = 32 slots) with its own siblings, so every chunk
    becomes an island the graph search cannot leave - filtered queries then return
    0 rows even with iterative scans. Blends spread the new points through the space
    between real chunks, which is closer to what a genuinely larger corpus looks like.
    """
    if factor <= 1:
        return chunks, emb
    rng = np.random.default_rng(seed)
    stride = max(c["id"] for c in chunks) + 1
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_cat.setdefault(c["category"], []).append(i)
    pools = {k: np.asarray(v) for k, v in by_cat.items()}
    cats = [c["category"] for c in chunks]
    words = [c["content"].split() for c in chunks]
    out_chunks, out_emb = list(chunks), [emb]
    for r in range(1, factor):
        partner = np.array([rng.choice(pools[k]) for k in cats])
        w = rng.uniform(0.5, 0.9, size=(len(chunks), 1)).astype(np.float32)
        e = w * emb + (1 - w) * emb[partner] + rng.normal(0.0, noise, emb.shape).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        out_emb.append(e.astype(np.float32))
        for i, c in enumerate(chunks):
            a, b = words[i], words[partner[i]]
            k = int(round(len(a) * float(w[i, 0])))
            text = " ".join(a[:k] + b[int(round(len(b) * float(w[i, 0]))):]) or c["content"]
            out_chunks.append({**c, "id": c["id"] + r * stride,
                               "doc_id": f"{c['doc_id']}-r{r}", "content": text})
    return out_chunks, np.concatenate(out_emb)
