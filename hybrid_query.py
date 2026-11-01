"""Hybrid query implementations for both engines.

pgvector: Reciprocal Rank Fusion in one SQL statement - a dense candidate list
          ordered by cosine distance (<=>, served by HNSW) and a keyword list
          ordered by ts_rank_cd (served by GIN), both under the same metadata
          WHERE filter, fused with score = sum(1 / (k + rank)).
Qdrant:   dense ANN query with a structured payload `query_filter` on the same
          metadata (category IN (...)), served by HNSW + the category index.
"""

from __future__ import annotations

import re

from common import HNSW_EF_SEARCH, PG_TABLE, QDRANT_COLLECTION

RRF_K = 60          # standard RRF constant
CANDIDATES = 40     # per-branch candidate depth before fusion
TOP_K = 10

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


def pg_hybrid(conn, qvec, qtext: str, categories: list[str], top_k: int = TOP_K):
    return conn.execute(PG_RRF_SQL, {
        "qvec": qvec, "tsq": to_or_tsquery(qtext), "cats": categories,
        "cand": CANDIDATES, "rrf_k": RRF_K, "top_k": top_k,
    }).fetchall()


def qdrant_hybrid(client, qvec, categories: list[str], top_k: int = TOP_K):
    from qdrant_client import models as qm

    flt = qm.Filter(must=[qm.FieldCondition(key="category", match=qm.MatchAny(any=categories))])
    res = client.query_points(
        QDRANT_COLLECTION,
        query=qvec.tolist() if hasattr(qvec, "tolist") else qvec,
        query_filter=flt,
        limit=top_k,
        search_params=qm.SearchParams(hnsw_ef=HNSW_EF_SEARCH),
        with_payload=["doc_id", "filename", "page_number", "category"],
    )
    return res.points
