"""Run hybrid queries against pgvector and Qdrant and print a Markdown summary.

    python benchmark.py --rebuild                  # ingest + index + query, base corpus
    python benchmark.py --scale 100 --runs 5       # 100x corpus (~113k chunks), 5 timed runs
    python benchmark.py --scale 100 --index-runs 3 # also rebuild the indexes 3x for load/index spread
    python benchmark.py                            # query only, reuse the existing indexes

Both engines run true hybrid search: dense HNSW + a keyword branch (Postgres full-text
on the pgvector side, BM25 sparse vectors on the Qdrant side), fused with RRF, under the
same category filter.

--scale N multiplies the ingested corpus N times with synthetic chunks (same-category
blends of two real chunks; see common.scale_corpus) so the HNSW indexes are actually
used. Indexes are rebuilt automatically when the scale or the ingested data changes.

An untimed check then runs each engine's dense branch on its own and reports rows
returned and recall@10 against brute force, so a degraded filtered-HNSW path can't
hide behind the keyword branch filling the fused top-10.

--runs R repeats the timed query phase R times (query order reshuffled each run,
engine order alternated, warm-up before every run). The table reports the median
across runs with the min-max range in parentheses; per-run numbers go to the JSON.

Query latency is client-side time around the database call; dense query embeddings
are computed once up front (same cost for both engines) and reported separately.
Writes results/benchmark.md and results/benchmark.json.
"""

from __future__ import annotations

import argparse
import platform
import random
import time

import numpy as np

from common import (
    EMBED_BACKEND, HNSW_EF_SEARCH, INDEX_SCHEMA, INDEX_STATS_PATH, INGEST_STATS_PATH, MODEL_NAME,
    QDRANT_LOCAL_PATH, QDRANT_PREFER_GRPC, RESULTS_DIR, corpus_fingerprint, get_qdrant_client,
    load_chunks, load_embedder, pg_connect, read_json, scale_corpus, write_json,
)
from hybrid_query import (
    CANDIDATES, RRF_K, TOP_K, pg_dense_ids, pg_hybrid, pg_plan_indexes, pg_session_setup,
    qdrant_dense_ids, qdrant_expected_plan, qdrant_fusion, qdrant_hybrid,
)

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
ENGINES = ("pgvector", "qdrant")


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


def lat_stats(lat) -> dict:
    return {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99), "mean": float(np.mean(lat))}


def spread(values) -> dict:
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "stdev": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "cv": float(a.std(ddof=1) / mean) if len(a) > 1 and mean else 0.0, "n": len(a)}


def fmt(s: dict, digits: int = 2, bold: bool = False) -> str:
    med = f"{s['median']:.{digits}f}"
    if bold:
        med = f"**{med}**"
    if s["n"] == 1:
        return med
    return f"{med} ({s['min']:.{digits}f}–{s['max']:.{digits}f})"


def timed_pass(fn, order, queries, qvecs, warmup):
    for i in range(warmup):
        j = order[i % len(order)]
        fn(queries[j], qvecs[j])
    lat, rows = [0.0] * len(queries), [None] * len(queries)
    for j in order:
        t = time.perf_counter()
        r = fn(queries[j], qvecs[j])
        lat[j] = (time.perf_counter() - t) * 1000
        rows[j] = r
    return lat, rows


def dense_branch_check(conn, qc, queries, qvecs, base_chunks, base_emb, idx) -> dict:
    """Untimed: run each engine's dense branch alone and compare with exact search.

    The fused result always fills top-k from the keyword branch, so a broken ANN
    path (e.g. filtered HNSW returning too few rows) would not show up in
    "avg results returned". This reports, per engine, how many of the CANDIDATES
    rows the filtered dense branch returns and its recall@TOP_K against brute force.
    """
    chunks, emb = scale_corpus(base_chunks, base_emb, idx["scale"], noise=idx.get("scale_noise", 0.01))
    ids = np.array([c["id"] for c in chunks])
    cats = np.array([c["category"] for c in chunks])
    out = {e: {"rows": [], "recall": []} for e in ENGINES}
    for q, v in zip(queries, qvecs):
        mask = np.isin(cats, q["categories"])
        sims = emb[mask] @ v
        exact = set(ids[mask][np.argsort(-sims)[:TOP_K]].tolist())
        for e, got in (("pgvector", pg_dense_ids(conn, v, q["categories"])),
                       ("qdrant", qdrant_dense_ids(qc, v, q["categories"]))):
            out[e]["rows"].append(len(got))
            out[e]["recall"].append(len(exact & set(got[:TOP_K])) / TOP_K)
    return {e: {"rows_mean": float(np.mean(d["rows"])), "rows_min": int(np.min(d["rows"])),
                "queries_short": int(sum(r < CANDIDATES for r in d["rows"])),
                "recall_at_k": float(np.mean(d["recall"]))} for e, d in out.items()}


def ensure_indexes(args, base_chunks, base_emb) -> dict:
    import index_db

    idx = read_json(INDEX_STATS_PATH)
    stale = (idx.get("schema") != INDEX_SCHEMA or idx.get("scale") != args.scale
             or idx.get("corpus") != corpus_fingerprint() or any(e not in idx for e in ENGINES))
    if not (args.rebuild or args.reindex or args.index_runs > 1 or stale):
        return idx
    if stale and not (args.rebuild or args.reindex):
        print(f"Index missing or built from different data/scale -> reindexing at scale {args.scale}.")
    chunks, emb = scale_corpus(base_chunks, base_emb, args.scale, noise=args.scale_noise)
    print(f"Indexing {len(chunks):,} chunks ({args.scale}x), {args.index_runs} build(s) per engine ...")
    runs = {e: [] for e in ENGINES}
    for r in range(args.index_runs):
        for e, stats in index_db.build_indexes(chunks, emb).items():
            runs[e].append(stats)
    idx = {"schema": INDEX_SCHEMA, "corpus": corpus_fingerprint(), "scale": args.scale,
           "scale_noise": args.scale_noise, "rows": len(chunks), **{e: {"runs": runs[e]} for e in ENGINES}}
    write_json(INDEX_STATS_PATH, idx)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="run ingest.py, then rebuild both indexes")
    ap.add_argument("--reindex", action="store_true", help="rebuild both indexes from existing artifacts")
    ap.add_argument("--scale", type=int, default=1, help="multiply the corpus N times (e.g. 50-100)")
    ap.add_argument("--scale-noise", type=float, default=0.01, help="embedding noise std for replicas")
    ap.add_argument("--runs", type=int, default=5, help="timed query runs per engine")
    ap.add_argument("--index-runs", type=int, default=1, help="index builds per engine (forces reindex if >1)")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.rebuild:
        import ingest
        ingest.main()
    ing = read_json(INGEST_STATS_PATH)
    if not ing:
        raise SystemExit("Missing ingest stats - run with --rebuild (or python ingest.py) first.")
    base_chunks, base_emb = load_chunks()
    idx = ensure_indexes(args, base_chunks, base_emb)

    categories = sorted({c["category"] for c in base_chunks})
    queries = build_queries(categories, args.queries, args.seed)

    model = load_embedder()
    t = time.perf_counter()
    qvecs = np.asarray(model.encode([q["text"] for q in queries], batch_size=64,
                                    normalize_embeddings=True), dtype=np.float32)
    q_embed_ms = (time.perf_counter() - t) * 1000 / len(queries)

    conn = pg_connect()
    pg_session_setup(conn)
    qc = get_qdrant_client()
    fusion_label = qdrant_fusion(qc)

    fns = {
        "pgvector": lambda q, v: pg_hybrid(conn, v, q["text"], q["categories"]),
        "qdrant": lambda q, v: qdrant_hybrid(qc, v, q["text"], q["categories"]),
    }
    per_run = {e: [] for e in ENGINES}
    pooled = {e: [] for e in ENGINES}
    first_rows = {}
    for r in range(args.runs):
        order = list(range(len(queries)))
        random.Random(args.seed * 1000 + r).shuffle(order)
        engine_order = ENGINES if r % 2 == 0 else ENGINES[::-1]
        line = []
        for e in engine_order:
            lat, rows = timed_pass(fns[e], order, queries, qvecs, args.warmup)
            per_run[e].append({**lat_stats(lat), "hits_mean": float(np.mean([len(x) for x in rows]))})
            pooled[e].extend(lat)
            first_rows.setdefault(e, rows)
            line.append(f"{e} p50={per_run[e][-1]['p50']:.2f} p95={per_run[e][-1]['p95']:.2f}")
        print(f"run {r + 1}/{args.runs}: " + "  ".join(line))

    # Sanity: every result must honour the metadata filter.
    pg_rows, qd_rows = first_rows["pgvector"], first_rows["qdrant"]
    for q, prow, qrow in zip(queries, pg_rows, qd_rows):
        assert all(x[4] in q["categories"] for x in prow), "pgvector filter violated"
        assert all(p.payload["category"] in q["categories"] for p in qrow), "qdrant filter violated"
    overlap = float(np.mean([len({x[0] for x in p} & {y.id for y in d}) / max(1, len(d))
                             for p, d in zip(pg_rows, qd_rows)]))

    # Which access paths are actually in play at this scale?
    one = next(i for i, q in enumerate(queries) if len(q["categories"]) == 1)
    two = next((i for i, q in enumerate(queries) if len(q["categories"]) == 2), one)
    plans = {
        "pgvector_1cat": pg_plan_indexes(conn, qvecs[one], queries[one]["text"], queries[one]["categories"]),
        "pgvector_2cat": pg_plan_indexes(conn, qvecs[two], queries[two]["text"], queries[two]["categories"]),
        "qdrant_1cat": None if QDRANT_LOCAL_PATH else qdrant_expected_plan(qc, queries[one]["categories"]),
        "qdrant_2cat": None if QDRANT_LOCAL_PATH else qdrant_expected_plan(qc, queries[two]["categories"]),
    }
    dense = dense_branch_check(conn, qc, queries, qvecs, base_chunks, base_emb, idx)
    conn.close()
    qc.close()

    # ---------------------------------------------------------------- report
    pre = ing["preprocess_total_s"]
    pg_runs, qd_runs = idx["pgvector"]["runs"], idx["qdrant"]["runs"]
    lat = {e: {m: spread([x[m] for x in per_run[e]]) for m in ("p50", "p95", "p99", "mean", "hits_mean")}
           for e in ENGINES}
    pool = {e: lat_stats(pooled[e]) for e in ENGINES}
    ix = {e: {k: spread([x[k] for x in runs]) for k in ("load_s", "index_s")}
          for e, runs in (("pgvector", pg_runs), ("qdrant", qd_runs))}
    total = {e: spread([pre + x["load_s"] for x in runs])
             for e, runs in (("pgvector", pg_runs), ("qdrant", qd_runs))}

    def breakdown(runs, key):
        keys = runs[0][key].keys()
        return {k: round(float(np.median([r[key][k] for r in runs])), 3) for k in keys}

    def hnsw_used(plan):
        return any("hnsw" in p for p in plan)

    row = lambda label, a, b: f"| {label} | {a} | {b} |"
    n_idx = len(pg_runs)
    md = "\n".join([
        "# Hybrid search benchmark: pgvector vs Qdrant",
        "",
        f"Corpus: {ing['documents']} PDFs, {ing['pages']} pages, {ing['chunks']:,} chunks "
        f"(500 chars / 100 overlap) x **scale {idx['scale']}** = **{idx['rows']:,} chunks** indexed"
        + (f" (replicas: same-category blends of two chunks, noise σ={idx['scale_noise']})" if idx["scale"] > 1 else "")
        + f". Embeddings: `{ing['embed_backend']}` (384-d). {len(queries)} queries, top-{TOP_K}, "
        f"{CANDIDATES} candidates per branch, {args.warmup} warm-up queries per engine per run.",
        "",
        f"Latency: median of **{args.runs} runs** (min–max across runs). "
        f"Load/index: median of **{n_idx} build(s)**" + (" (min–max)." if n_idx > 1 else "."),
        "",
        "| Metric | pgvector (RRF: HNSW cosine + GIN ts_rank_cd, WHERE filter) "
        f"| Qdrant ({fusion_label}: HNSW cosine + BM25 sparse, payload filter) |",
        "|---|---:|---:|",
        row("Shared preprocessing: parse + chunk + embed, base corpus (s)", f"{pre:.2f}", f"{pre:.2f}"),
        row("DB load / upsert (s)", fmt(ix["pgvector"]["load_s"]), fmt(ix["qdrant"]["load_s"])),
        row("**Total ingestion time (s)**", fmt(total["pgvector"], bold=True), fmt(total["qdrant"], bold=True)),
        row("**Index creation overhead (s)**", fmt(ix["pgvector"]["index_s"], bold=True),
            fmt(ix["qdrant"]["index_s"], bold=True)),
        row("**Query latency P50 (ms)**", fmt(lat["pgvector"]["p50"], bold=True), fmt(lat["qdrant"]["p50"], bold=True)),
        row("**Query latency P95 (ms)**", fmt(lat["pgvector"]["p95"], bold=True), fmt(lat["qdrant"]["p95"], bold=True)),
        row("Query latency P99 (ms)", fmt(lat["pgvector"]["p99"]), fmt(lat["qdrant"]["p99"])),
        row("Query latency mean (ms)", fmt(lat["pgvector"]["mean"]), fmt(lat["qdrant"]["mean"])),
        row(f"Pooled P50 / P95 / P99 over all {args.runs * len(queries)} queries (ms)",
            f"{pool['pgvector']['p50']:.2f} / {pool['pgvector']['p95']:.2f} / {pool['pgvector']['p99']:.2f}",
            f"{pool['qdrant']['p50']:.2f} / {pool['qdrant']['p95']:.2f} / {pool['qdrant']['p99']:.2f}"),
        row("Run-to-run CV of P50", f"{lat['pgvector']['p50']['cv']:.1%}", f"{lat['qdrant']['p50']['cv']:.1%}"),
        row("Avg results returned", f"{lat['pgvector']['hits_mean']['median']:.1f}",
            f"{lat['qdrant']['hits_mean']['median']:.1f}"),
        row(f"Dense branch alone: rows returned of {CANDIDATES} (mean / min)",
            f"{dense['pgvector']['rows_mean']:.1f} / {dense['pgvector']['rows_min']}",
            f"{dense['qdrant']['rows_mean']:.1f} / {dense['qdrant']['rows_min']}"),
        row(f"Dense branch alone: recall@{TOP_K} vs exact (ef_search={HNSW_EF_SEARCH})",
            f"{dense['pgvector']['recall_at_k']:.1%}", f"{dense['qdrant']['recall_at_k']:.1%}"),
        "",
        f"- Load breakdown (s, median): pgvector {breakdown(pg_runs, 'load_breakdown_s')}; "
        f"Qdrant {breakdown(qd_runs, 'load_breakdown_s')}",
        f"- Index breakdown (s, median): pgvector {breakdown(pg_runs, 'index_breakdown_s')}; "
        f"Qdrant {breakdown(qd_runs, 'index_breakdown_s')}",
        f"- Access path, 1-category filter: pgvector plan uses {plans['pgvector_1cat']} "
        f"(HNSW {'used' if hnsw_used(plans['pgvector_1cat']) else 'NOT used'}); "
        f"Qdrant expected {plans['qdrant_1cat']}",
        f"- Access path, 2-category filter: pgvector plan uses {plans['pgvector_2cat']} "
        f"(HNSW {'used' if hnsw_used(plans['pgvector_2cat']) else 'NOT used'}); "
        f"Qdrant expected {plans['qdrant_2cat']}",
        f"- Query embedding ({ing['embed_backend']}, shared, not in latency above): {q_embed_ms:.2f} ms/query. "
        "Keyword query parsing is inside the timed call on both sides (to_tsquery / BM25 query encoding).",
        f"- Top-{TOP_K} overlap between engines (run 1): {overlap:.0%}. Both are hybrid now; the remaining gap "
        "is mostly the keyword scorer (ts_rank_cd vs BM25) and ANN approximation.",
        f"- Qdrant transport: {'in-process local mode (NOT comparable)' if QDRANT_LOCAL_PATH else ('gRPC' if QDRANT_PREFER_GRPC else 'REST')}"
        f", server {qd_runs[0].get('server_version')}; pgvector {pg_runs[0].get('pgvector_version')} "
        f"(parallel maintenance workers: {pg_runs[0].get('parallel_maintenance_workers')}); "
        f"host {platform.platform()}",
    ])
    if EMBED_BACKEND == "hash" or QDRANT_LOCAL_PATH:
        md += "\n\n> **Smoke-test run** (hash embeddings and/or Qdrant local mode) - do not publish these numbers."

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.md").write_text(md + "\n")
    write_json(RESULTS_DIR / "benchmark.json", {
        "args": vars(args), "ingest": ing, "index": idx, "queries": queries, "embed_model": MODEL_NAME,
        "rrf": {"pgvector_k": RRF_K, "qdrant": fusion_label, "candidates": CANDIDATES, "top_k": TOP_K},
        "latency_per_run_ms": per_run, "latency_summary_ms": lat, "latency_pooled_ms": pool,
        "plans": plans, "dense_branch_check": dense, "overlap_at_k": overlap,
    })
    print("\n" + md)


if __name__ == "__main__":
    main()
