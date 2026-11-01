"""Parse ./docs/*.pdf -> 500-char chunks (100 overlap) -> 384-dim MiniLM embeddings.

Writes artifacts/chunks.jsonl (text + metadata), artifacts/embeddings.npy and
artifacts/ingest_stats.json (timings). Both databases are loaded from these
artifacts by index_db.py, so they index byte-identical data.

Metadata per chunk: doc_id, filename, page_number (1-based), category.
Category comes from docs/categories.json if present, else the filename prefix
before "_policy", else "general".
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from common import (
    ARTIFACTS_DIR, CATEGORY_MANIFEST, CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, DOCS_DIR,
    EMBED_BACKEND, EMBEDDINGS_PATH, INGEST_STATS_PATH, MODEL_NAME, load_embedder, write_json,
)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding character window that prefers to break on whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start, step_min = [], 0, size - overlap
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + step_min, end)  # don't split mid-word
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # align the overlap start to a word boundary
        sp = text.find(" ", start, end)
        if sp != -1 and sp - start < 20:
            start = sp + 1
    return [c for c in chunks if c]


def category_for(fname: str, manifest: dict[str, str]) -> str:
    if fname in manifest:
        return manifest[fname]
    m = re.match(r"(.+?)_policy", fname)
    return m.group(1) if m else "general"


def main() -> None:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {DOCS_DIR} - run `python generate_docs.py` or add your own.")
    manifest = json.loads(CATEGORY_MANIFEST.read_text()) if CATEGORY_MANIFEST.exists() else {}

    t0 = time.perf_counter()
    chunks: list[dict] = []
    pages = 0
    for d_idx, pdf in enumerate(pdfs, start=1):
        doc_id = f"DOC-{d_idx:03d}"
        cat = category_for(pdf.name, manifest)
        for p_idx, page in enumerate(PdfReader(pdf).pages, start=1):
            pages += 1
            for piece in chunk_text(page.extract_text() or ""):
                chunks.append({
                    "id": len(chunks) + 1,
                    "doc_id": doc_id,
                    "filename": pdf.name,
                    "page_number": p_idx,
                    "category": cat,
                    "content": piece,
                })
    t_parse = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = load_embedder()
    t_model = time.perf_counter() - t1

    t2 = time.perf_counter()
    emb = model.encode([c["content"] for c in chunks], batch_size=64,
                       normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    t_embed = time.perf_counter() - t2
    assert emb.shape[1] == 384, f"expected 384 dims, got {emb.shape[1]}"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    np.save(EMBEDDINGS_PATH, emb)

    stats = {
        "documents": len(pdfs),
        "pages": pages,
        "chunks": len(chunks),
        "embed_backend": EMBED_BACKEND if EMBED_BACKEND == "hash" else MODEL_NAME,
        "parse_chunk_s": round(t_parse, 3),
        "model_load_s": round(t_model, 3),
        "embed_s": round(t_embed, 3),
        "preprocess_total_s": round(t_parse + t_embed, 3),  # excludes one-off model load
    }
    write_json(INGEST_STATS_PATH, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
