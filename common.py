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
