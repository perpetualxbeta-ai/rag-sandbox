"""Run 100 hybrid queries against pgvector and Qdrant and print a Markdown summary.

    python benchmark.py --rebuild      # ingest + index + query (full head-to-head)
    python benchmark.py                # query only, reuse artifacts/*_stats.json

Query latency is measured client-side around the database call only; query
embeddings are computed once up front (same cost for both engines) and reported
separately. Each engine gets warm-up queries before the timed run.
Writes results/benchmark.md and results/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time

import numpy as np

from common import (
    EMBED_BACKEND, INDEX_STATS_PATH, INGEST_STATS_PATH, MODEL_NAME, QDRANT_LOCAL_PATH,
    QDRANT_PREFER_GRPC, RESULTS_DIR, get_qdrant_client, load_chunks, load_embedder, pg_connect,
    read_json, write_json,
)
from hybrid_query import TOP_K, pg_hybrid, pg_session_setup, qdrant_hybrid

QUESTION_TEMPLATES = [
    "What is the policy on {topic}?",
    "How far in advance do I need to request {topic}?",
    "Who approves exceptions for {topic}?",
    "What happens if I breach the {topic} rules?",
    "How long are {topic} records kept?",
    "Is training on {topic} mandatory and how often?",
    "Who should I contact with questions about {topic}?",
    "What is the maximum limit for {topic} per year?",
]
GENERIC_TOPICS = ["approvals", "exceptions", "record retention", "training", "escalation", "compliance"]


def build_queries(categories: list[str], n: int, seed: int) -> list[dict]:
    try:
        from generate_docs import CATEGORIES as SPEC
    except Exception:  # reportlab missing etc.
        SPEC = {}
    rng = random.Random(seed)
    queries = []
    for i in range(n):
        primary = categories[i % len(categories)]
        topics = SPEC.get(primary, {}).get("topics", GENERIC_TOPICS)
        filt = [primary]
        if rng.random() < 0.3 and len(categories) > 1:  # ~30% filter on two categories
            filt.append(rng.choice([c for c in categories if c != primary]))
        queries.append({"text": rng.choice(QUESTION_TEMPLATES).format(topic=rng.choice(topics)),
                        "categories": filt})
    return queries


def pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


def run_engine(name, fn, queries, qvecs, warmup):
    for i in range(warmup):
        fn(queries[i % len(queries)], qvecs[i % len(queries)])
    lat, hits, ids = [], [], []
    for q, v in zip(queries, qvecs):
        t = time.perf_counter()
        rows = fn(q, v)
        lat.append((time.perf_counter() - t) * 1000)
        hits.append(len(rows))
        ids.append(rows)
    print(f"{name}: p50={pct(lat, 50):.2f} ms  p95={pct(lat, 95):.2f} ms  avg hits={np.mean(hits):.1f}")
    return lat, hits, ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="run ingest.py and index_db.py first")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.rebuild:
        import index_db
        import ingest
        ingest.main()
        chunks, emb = load_chunks()
        stats = {"pgvector": index_db.index_pgvector(chunks, emb),
                 "qdrant": index_db.index_qdrant(chunks, emb)}
        write_json(INDEX_STATS_PATH, stats)

    ing, idx = read_json(INGEST_STATS_PATH), read_json(INDEX_STATS_PATH)
    if not ing or "pgvector" not in idx or "qdrant" not in idx:
        raise SystemExit("Missing stats - run with --rebuild (or ingest.py + index_db.py) first.")

    chunks, _ = load_chunks()
    categories = sorted({c["category"] for c in chunks})
    queries = build_queries(categories, args.queries, args.seed)

    model = load_embedder()
    t = time.perf_counter()
    qvecs = np.asarray(model.encode([q["text"] for q in queries], batch_size=64,
                                    normalize_embeddings=True), dtype=np.float32)
    q_embed_ms = (time.perf_counter() - t) * 1000 / len(queries)

    conn = pg_connect()
    pg_session_setup(conn)
    qc = get_qdrant_client()

    pg_lat, pg_hits, pg_rows = run_engine(
        "pgvector", lambda q, v: pg_hybrid(conn, v, q["text"], q["categories"]), queries, qvecs, args.warmup)
    qd_lat, qd_hits, qd_rows = run_engine(
        "qdrant", lambda q, v: qdrant_hybrid(qc, v, q["categories"]), queries, qvecs, args.warmup)

    # Sanity: every result must honour the metadata filter.
    for q, prow, qrow in zip(queries, pg_rows, qd_rows):
        assert all(r[4] in q["categories"] for r in prow), "pgvector filter violated"
        assert all(p.payload["category"] in q["categories"] for p in qrow), "qdrant filter violated"
    overlap = np.mean([len({r[0] for r in p} & {x.id for x in d}) / max(1, len(d))
                       for p, d in zip(pg_rows, qd_rows)])
    conn.close()
    qc.close()

    pre = ing["preprocess_total_s"]
    pg, qd = idx["pgvector"], idx["qdrant"]
    row = lambda label, a, b: f"| {label} | {a} | {b} |"
    md = "\n".join([
        f"# Hybrid search benchmark: pgvector vs Qdrant",
        "",
        f"Corpus: {ing['documents']} PDFs, {ing['pages']} pages, {ing['chunks']} chunks "
        f"(500 chars / 100 overlap), embeddings: `{ing['embed_backend']}` (384-d). "
        f"{len(queries)} queries, top-{TOP_K}, {args.warmup} warm-up queries per engine.",
        "",
        "| Metric | pgvector (RRF: HNSW cosine + GIN ts_rank_cd, WHERE filter) | Qdrant (HNSW cosine + payload query_filter) |",
        "|---|---:|---:|",
        row("Shared preprocessing: parse + chunk + embed (s)", f"{pre:.2f}", f"{pre:.2f}"),
        row("DB load / upsert (s)", f"{pg['load_s']:.2f}", f"{qd['load_s']:.2f}"),
        row("**Total ingestion time (s)**", f"**{pre + pg['load_s']:.2f}**", f"**{pre + qd['load_s']:.2f}**"),
        row("**Index creation overhead (s)**", f"**{pg['index_s']:.2f}**", f"**{qd['index_s']:.2f}**"),
        row("**Query latency P50 (ms)**", f"**{pct(pg_lat, 50):.2f}**", f"**{pct(qd_lat, 50):.2f}**"),
        row("**Query latency P95 (ms)**", f"**{pct(pg_lat, 95):.2f}**", f"**{pct(qd_lat, 95):.2f}**"),
        row("Query latency P99 (ms)", f"{pct(pg_lat, 99):.2f}", f"{pct(qd_lat, 99):.2f}"),
        row("Query latency mean (ms)", f"{np.mean(pg_lat):.2f}", f"{np.mean(qd_lat):.2f}"),
        row("Avg results returned", f"{np.mean(pg_hits):.1f}", f"{np.mean(qd_hits):.1f}"),
        "",
        f"- Index breakdown (s): pgvector {pg['index_breakdown_s']}; Qdrant {qd['index_breakdown_s']}",
        f"- Query embedding ({ing['embed_backend']}, shared, not in latency above): {q_embed_ms:.2f} ms/query",
        f"- Top-{TOP_K} overlap between engines: {overlap:.0%} (pgvector adds keyword fusion, so <100% is expected)",
        f"- Qdrant transport: {'in-process local mode (NOT comparable)' if QDRANT_LOCAL_PATH else ('gRPC' if QDRANT_PREFER_GRPC else 'REST')}; "
        f"pgvector {pg.get('pgvector_version')}; host {platform.platform()}",
    ])
    if EMBED_BACKEND == "hash" or QDRANT_LOCAL_PATH:
        md += "\n\n> **Smoke-test run** (hash embeddings and/or Qdrant local mode) - do not publish these numbers."

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.md").write_text(md + "\n")
    write_json(RESULTS_DIR / "benchmark.json", {
        "ingest": ing, "index": idx, "queries": queries, "embed_model": MODEL_NAME,
        "pgvector_latency_ms": pg_lat, "qdrant_latency_ms": qd_lat, "overlap_at_k": overlap,
    })
    print("\n" + md)


if __name__ == "__main__":
    main()
