"""Hybrid query implementations for both engines.

Both sides now do true hybrid search - a dense branch and a keyword branch under
the same metadata filter, each cut to CANDIDATES rows, fused with Reciprocal Rank
Fusion score = sum(1 / (RRF_K + rank)):

pgvector: one SQL statement. Dense list ordered by cosine distance (<=>, HNSW),
          keyword list ordered by ts_rank_cd (GIN on tsvector), both under
          WHERE category = ANY(...), fused in SQL.
Qdrant:   one Query API call. Two `prefetch` branches - the `dense` named vector
          (HNSW) and the `bm25` sparse vector (inverted index, IDF applied
          server-side) - each with the payload filter, fused with RRF.

The keyword scorers differ (Postgres ts_rank_cd vs real BM25 in Qdrant); RRF only
uses ranks, so the fusion step is the same on both sides.
"""

from __future__ import annotations

import re

from common import HNSW_EF_SEARCH, PG_TABLE, QDRANT_COLLECTION, QDRANT_DENSE, QDRANT_SPARSE

RRF_K = 60          # standard RRF constant, used by both engines when the server supports it
CANDIDATES = 40     # per-branch candidate depth before fusion
TOP_K = 10
PAYLOAD_FIELDS = ["doc_id", "filename", "page_number", "category"]

PG_RRF_SQL = f"""
WITH semantic AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> %(qvec)s) AS rank
    FROM {PG_TABLE}
    WHERE category = ANY(%(cats)s)
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(cand)s
),
keyword AS (
    SELECT id, RANK() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rank
    FROM {PG_TABLE}, to_tsquery('english', %(tsq)s) AS q
    WHERE tsv @@ q AND category = ANY(%(cats)s)
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT %(cand)s
)
SELECT c.id, c.doc_id, c.filename, c.page_number, c.category,
       COALESCE(1.0 / (%(rrf_k)s + s.rank), 0.0)
     + COALESCE(1.0 / (%(rrf_k)s + k.rank), 0.0) AS rrf_score
FROM semantic s
FULL OUTER JOIN keyword k ON s.id = k.id
JOIN {PG_TABLE} c ON c.id = COALESCE(s.id, k.id)
ORDER BY rrf_score DESC, c.id
LIMIT %(top_k)s
"""


def to_or_tsquery(text: str) -> str:
    """'How do I claim mileage?' -> 'how | do | i | claim | mileage' (OR semantics).

    to_tsquery drops English stopwords itself; tokens are restricted to [a-z0-9]
    so user text can never inject tsquery operators.
    """
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return " | ".join(toks) if toks else "''"


def pg_session_setup(conn) -> None:
    conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    # pgvector >= 0.8: keep scanning the HNSW graph until enough rows pass the WHERE filter.
    try:
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except Exception:
        pass  # older pgvector; filtered HNSW may return < LIMIT rows for rare categories


def _pg_params(qvec, qtext, categories, top_k):
    return {"qvec": qvec, "tsq": to_or_tsquery(qtext), "cats": categories,
            "cand": CANDIDATES, "rrf_k": RRF_K, "top_k": top_k}


def pg_hybrid(conn, qvec, qtext: str, categories: list[str], top_k: int = TOP_K):
    return conn.execute(PG_RRF_SQL, _pg_params(qvec, qtext, categories, top_k)).fetchall()


def pg_dense_ids(conn, qvec, categories: list[str], k: int = CANDIDATES) -> list[int]:
    """The dense branch on its own (same filter/settings as inside PG_RRF_SQL)."""
    return [r[0] for r in conn.execute(
        f"SELECT id FROM {PG_TABLE} WHERE category = ANY(%s) ORDER BY embedding <=> %s LIMIT %s",
        (categories, qvec, k)).fetchall()]


def pg_plan_indexes(conn, qvec, qtext: str, categories: list[str]) -> list[str]:
    """Names of the indexes the planner picks for this hybrid query (EXPLAIN, not ANALYZE)."""
    plan = conn.execute("EXPLAIN (FORMAT JSON) " + PG_RRF_SQL,
                        _pg_params(qvec, qtext, categories, TOP_K)).fetchone()[0]
    found = []

    def walk(node):
        name = node.get("Index Name")
        if name:
            found.append(f"{node['Node Type']}: {name}")
        elif node.get("Node Type") == "Seq Scan":
            found.append(f"Seq Scan: {node.get('Relation Name')}")
        for child in node.get("Plans", []):
            walk(child)

    walk(plan[0]["Plan"])
    return sorted(set(found))


# --------------------------------------------------------------------------- Qdrant
_FUSION = None   # resolved once by qdrant_fusion()


def qdrant_fusion(client) -> str:
    """Pick RRF with k=RRF_K when client+server support it, else Qdrant's default RRF.

    Returns a label for the report. Parameterised RRF arrived in Qdrant 1.16; older
    servers/clients fall back to Fusion.RRF, whose k is Qdrant's built-in default.
    """
    global _FUSION
    from qdrant_client import models as qm

    if hasattr(qm, "RrfQuery") and hasattr(qm, "Rrf"):
        candidate = qm.RrfQuery(rrf=qm.Rrf(k=RRF_K))
        try:
            # Probe: a fusion query with one trivial prefetch; old servers reject Rrf(k).
            client.query_points(
                QDRANT_COLLECTION,
                prefetch=[qm.Prefetch(query=qm.SparseVector(indices=[0], values=[1.0]),
                                      using=QDRANT_SPARSE, limit=1)],
                query=candidate, limit=1)
            _FUSION = candidate
            return f"RRF k={RRF_K}"
        except Exception:
            pass
    _FUSION = qm.FusionQuery(fusion=qm.Fusion.RRF)
    return "RRF (Qdrant default k - server/client too old for Rrf(k=...))"


def qdrant_hybrid(client, qvec, qtext: str, categories: list[str], top_k: int = TOP_K):
    from qdrant_client import models as qm

    from sparse import encode_query

    if _FUSION is None:
        qdrant_fusion(client)
    flt = qm.Filter(must=[qm.FieldCondition(key="category", match=qm.MatchAny(any=categories))])
    dense = qvec.tolist() if hasattr(qvec, "tolist") else qvec
    prefetch = [qm.Prefetch(query=dense, using=QDRANT_DENSE, filter=flt, limit=CANDIDATES,
                            params=qm.SearchParams(hnsw_ef=HNSW_EF_SEARCH))]
    idx, val = encode_query(qtext)   # timed with the call, like to_tsquery on the pg side
    if idx:                          # all-stopword query -> dense branch only (pg does the same)
        prefetch.append(qm.Prefetch(query=qm.SparseVector(indices=idx, values=val),
                                    using=QDRANT_SPARSE, filter=flt, limit=CANDIDATES))
    res = client.query_points(QDRANT_COLLECTION, prefetch=prefetch, query=_FUSION,
                              query_filter=flt, limit=top_k, with_payload=PAYLOAD_FIELDS)
    return res.points


def qdrant_dense_ids(client, qvec, categories: list[str], k: int = CANDIDATES) -> list[int]:
    """The dense prefetch branch on its own."""
    from qdrant_client import models as qm

    flt = qm.Filter(must=[qm.FieldCondition(key="category", match=qm.MatchAny(any=categories))])
    res = client.query_points(QDRANT_COLLECTION, query=qvec.tolist(), using=QDRANT_DENSE,
                              query_filter=flt, limit=k, with_payload=False,
                              search_params=qm.SearchParams(hnsw_ef=HNSW_EF_SEARCH))
    return [p.id for p in res.points]


def qdrant_expected_plan(client, categories: list[str]) -> str:
    """Qdrant's planner rule: filtered HNSW if the filter's matching vectors exceed
    hnsw_config.full_scan_threshold (KB), otherwise an exact scan of the matches."""
    from qdrant_client import models as qm

    from common import EMBED_DIM

    info = client.get_collection(QDRANT_COLLECTION)
    thr_kb = info.config.hnsw_config.full_scan_threshold
    flt = qm.Filter(must=[qm.FieldCondition(key="category", match=qm.MatchAny(any=categories))])
    n = client.count(QDRANT_COLLECTION, count_filter=flt, exact=True).count
    kb = n * EMBED_DIM * 4 / 1024
    how = "filtered HNSW" if kb > thr_kb else "exact scan of filtered points"
    return f"{how} ({n:,} matching vectors = {kb:,.0f} KB vs full_scan_threshold {thr_kb:,} KB)"
