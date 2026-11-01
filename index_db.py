"""Load the ingested chunks into pgvector and Qdrant, timing load vs index build.

Both engines follow the same two-phase shape so the numbers are comparable:
  1. load   - bulk-write rows/points with no ANN index in place
  2. index  - build HNSW (m=16, ef_construction=64) + the filter/keyword indexes

pgvector: table with vector(384), a STORED generated tsvector column,
          HNSW (vector_cosine_ops), GIN on the tsvector, btree on category.
          Bulk load is binary COPY; HNSW is built with maintenance_work_mem
          = PG_MAINTENANCE_WORK_MEM (default 1GB) so the graph fits in memory.
Qdrant:   collection with a `dense` Cosine vector and a `bm25` sparse vector
          (modifier=IDF), HNSW indexing deferred during upload, then enabled,
          plus a keyword payload index on `category`; we wait for status GREEN.
          BM25 document weights are computed client-side and counted in Qdrant's
          load time (Postgres computes its tsvector inside COPY, also in load).

    python index_db.py                 # both, base corpus
    python index_db.py --scale 50      # both, corpus multiplied 50x
    python index_db.py --only pgvector
"""

from __future__ import annotations

import argparse
import json
import time

from common import (
    EMBED_DIM, HNSW_EF_CONSTRUCTION, HNSW_M, INDEX_SCHEMA, INDEX_STATS_PATH, PG_MAINTENANCE_WORK_MEM,
    PG_TABLE, QDRANT_COLLECTION, QDRANT_DENSE, QDRANT_LOCAL_PATH, QDRANT_SPARSE, corpus_fingerprint,
    get_qdrant_client, load_chunks, pg_connect, read_json, scale_corpus, write_json,
)

UPSERT_BATCH = 512


# --------------------------------------------------------------------------- pgvector
def index_pgvector(chunks, emb) -> dict:
    conn = pg_connect()
    conn.execute(f"DROP TABLE IF EXISTS {PG_TABLE}")
    conn.execute(f"""
        CREATE TABLE {PG_TABLE} (
            id          BIGINT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            filename    TEXT NOT NULL,
            page_number INT  NOT NULL,
            category    TEXT NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({EMBED_DIM}) NOT NULL,
            tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        )""")

    # Phase 1: binary COPY (tsvector is computed by Postgres as rows land).
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        with cur.copy(f"COPY {PG_TABLE} (id, doc_id, filename, page_number, category, content, embedding) "
                      "FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["bigint", "text", "text", "integer", "text", "text", "vector"])
            for c, v in zip(chunks, emb):
                cp.write_row((c["id"], c["doc_id"], c["filename"], c["page_number"], c["category"],
                              c["content"], v))
    load_s = time.perf_counter() - t0

    # Phase 2: indexes.
    conn.execute(f"SET maintenance_work_mem = '{PG_MAINTENANCE_WORK_MEM}'")
    timings = {}
    for name, ddl in [
        ("hnsw", f"CREATE INDEX {PG_TABLE}_hnsw ON {PG_TABLE} USING hnsw (embedding vector_cosine_ops) "
                 f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"),
        ("gin_tsv", f"CREATE INDEX {PG_TABLE}_tsv_gin ON {PG_TABLE} USING gin (tsv)"),
        ("btree_category", f"CREATE INDEX {PG_TABLE}_category ON {PG_TABLE} (category)"),
    ]:
        t = time.perf_counter()
        conn.execute(ddl)
        timings[name] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    conn.execute(f"ANALYZE {PG_TABLE}")
    timings["analyze"] = round(time.perf_counter() - t, 3)

    n = conn.execute(f"SELECT count(*) FROM {PG_TABLE}").fetchone()[0]
    ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
    workers = conn.execute("SHOW max_parallel_maintenance_workers").fetchone()[0]
    conn.close()
    return {"rows": n, "pgvector_version": ver, "parallel_maintenance_workers": int(workers),
            "load_s": round(load_s, 3), "index_s": round(sum(timings.values()), 3),
            "load_breakdown_s": {"copy": round(load_s, 3)}, "index_breakdown_s": timings}


# --------------------------------------------------------------------------- Qdrant
def index_qdrant(chunks, emb) -> dict:
    from qdrant_client import models as qm

    from sparse import encode_docs

    client = get_qdrant_client()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config={QDRANT_DENSE: qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE)},
        sparse_vectors_config={QDRANT_SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        hnsw_config=qm.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCTION),
        # indexing_threshold=0 disables HNSW building during upload (phase 1 = pure load).
        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=0),
    )

    # Phase 1: BM25 weights + upload points.
    t0 = time.perf_counter()
    sparse, avgdl = encode_docs([c["content"] for c in chunks])
    bm25_s = time.perf_counter() - t0
    t = time.perf_counter()
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i:i + UPSERT_BATCH]
        client.upsert(
            QDRANT_COLLECTION,
            points=[qm.PointStruct(
                id=c["id"],
                vector={QDRANT_DENSE: emb[i + j].tolist(),
                        QDRANT_SPARSE: qm.SparseVector(indices=sparse[i + j][0], values=sparse[i + j][1])},
                payload={k: c[k] for k in ("doc_id", "filename", "page_number", "category", "content")})
                for j, c in enumerate(batch)],
            wait=True,
        )
    upsert_s = time.perf_counter() - t
    load_s = time.perf_counter() - t0

    # Phase 2: payload index on `category` + enable HNSW and wait until it is built.
    timings = {}
    t = time.perf_counter()
    client.create_payload_index(QDRANT_COLLECTION, field_name="category",
                                field_schema=qm.PayloadSchemaType.KEYWORD, wait=True)
    timings["payload_index_category"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    indexed = None
    if not QDRANT_LOCAL_PATH:  # in-process mode has no HNSW/optimizer
        # threshold is in KB; 1 KB forces every segment to get an HNSW index.
        client.update_collection(QDRANT_COLLECTION,
                                 optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=1))
        deadline = time.time() + 1800
        while time.time() < deadline:
            info = client.get_collection(QDRANT_COLLECTION)
            indexed = info.indexed_vectors_count or 0
            # indexed_vectors_count covers dense + sparse vectors, so wait for both.
            if info.status == qm.CollectionStatus.GREEN and indexed >= 2 * len(chunks):
                break
            time.sleep(0.05)
        else:
            raise SystemExit("Qdrant HNSW build did not finish within 1800 s")
    timings["hnsw"] = round(time.perf_counter() - t, 3)

    n = client.count(QDRANT_COLLECTION, exact=True).count
    version = None
    try:
        version = client.info().version
    except Exception:
        pass
    client.close()
    return {"points": n, "indexed_vectors": indexed, "server_version": version,
            "mode": "local" if QDRANT_LOCAL_PATH else "server", "bm25_avgdl": round(avgdl, 1),
            "load_s": round(load_s, 3), "index_s": round(sum(timings.values()), 3),
            "load_breakdown_s": {"bm25_encode": round(bm25_s, 3), "upsert": round(upsert_s, 3)},
            "index_breakdown_s": timings}


def build_indexes(chunks, emb, engines=("pgvector", "qdrant")) -> dict:
    out = {}
    if "pgvector" in engines:
        out["pgvector"] = index_pgvector(chunks, emb)
        print("pgvector:", json.dumps({k: out["pgvector"][k] for k in ("rows", "load_s", "index_s")}))
    if "qdrant" in engines:
        out["qdrant"] = index_qdrant(chunks, emb)
        print("qdrant:  ", json.dumps({k: out["qdrant"][k] for k in ("points", "load_s", "index_s")}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["pgvector", "qdrant"])
    ap.add_argument("--scale", type=int, default=1, help="multiply the corpus N times (synthetic replicas)")
    ap.add_argument("--scale-noise", type=float, default=0.01)
    args = ap.parse_args()

    chunks, emb = load_chunks()
    chunks, emb = scale_corpus(chunks, emb, args.scale, noise=args.scale_noise)
    engines = [args.only] if args.only else ["pgvector", "qdrant"]
    run = build_indexes(chunks, emb, engines)
    stats = read_json(INDEX_STATS_PATH)
    if (stats.get("schema"), stats.get("scale"), stats.get("corpus")) != \
            (INDEX_SCHEMA, args.scale, corpus_fingerprint()):
        stats = {}
    stats.update({"schema": INDEX_SCHEMA, "corpus": corpus_fingerprint(), "scale": args.scale,
                  "scale_noise": args.scale_noise, "rows": len(chunks)})
    for e in engines:
        stats[e] = {"runs": [run[e]]}
    write_json(INDEX_STATS_PATH, stats)


if __name__ == "__main__":
    main()
