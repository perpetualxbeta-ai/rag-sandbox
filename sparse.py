"""BM25 sparse vectors for Qdrant's keyword branch.

Documents get per-term BM25 term-frequency weights

    w(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

and queries get 1.0 per unique term. The Qdrant sparse vector is created with
`modifier=IDF`, so Qdrant multiplies in IDF server-side from live collection
statistics; the sparse dot product is then exactly the BM25 score.

Tokenisation mirrors Postgres' `english` text-search config used by the
pgvector side: lowercase alphanumeric tokens, Snowball English stopwords
dropped, Snowball English stemming. Terms are hashed to u32 ids with CRC32.
"""

from __future__ import annotations

import re
import zlib
from functools import lru_cache

from common import BM25_B, BM25_K1

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Snowball English stopword list (the same list Postgres ships as english.stop).
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she
her hers herself it its itself they them their theirs themselves what which who whom this that
these those am is are was were be been being have has had having do does did doing a an the and
but if or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here
there when where why how all any both each few more most other some such no nor not only own
same so than too very s t can will just don should now
""".split())

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("english")
except ImportError as e:  # pragma: no cover
    raise SystemExit("BM25 needs `snowballstemmer` - pip install -r requirements-bench.txt") from e


@lru_cache(maxsize=200_000)
def _term_id(token: str) -> int | None:
    if token in STOPWORDS:
        return None
    return zlib.crc32(_STEMMER.stemWord(token).encode())


def term_ids(text: str) -> list[int]:
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tid = _term_id(tok)
        if tid is not None:
            out.append(tid)
    return out


def encode_query(text: str) -> tuple[list[int], list[float]]:
    ids = sorted(set(term_ids(text)))
    return ids, [1.0] * len(ids)


def encode_docs(texts: list[str], k1: float = BM25_K1, b: float = BM25_B):
    """Return (list of (indices, values), avgdl) for a corpus."""
    toks = [term_ids(t) for t in texts]
    avgdl = sum(map(len, toks)) / max(1, len(toks))
    out = []
    for ids in toks:
        tf: dict[int, int] = {}
        for i in ids:
            tf[i] = tf.get(i, 0) + 1
        norm = k1 * (1 - b + b * len(ids) / avgdl) if avgdl else k1
        idx = sorted(tf)
        out.append((idx, [tf[i] * (k1 + 1) / (tf[i] + norm) for i in idx]))
    return out, avgdl
