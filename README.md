# rag-sandbox

This repo holds two independent experiments:

- **[RAG proof of concept](#rag-proof-of-concept)** over the *Wacky Acme Company Policy* handbook, built with LangChain, Chroma and OpenAI.
- **[Hybrid search benchmark](#hybrid-search-benchmark-pgvector-vs-qdrant)** comparing pgvector and Qdrant on filtered dense + keyword search. Fully local, no API keys. [Latest results](#latest-results).

## RAG proof of concept

```
data/Wacky Acme Company Policy.pdf
  -> PyPDFLoader (one Document per page)
  -> RecursiveCharacterTextSplitter (chunk_size=600, chunk_overlap=100)
  -> OpenAIEmbeddings -> Chroma (persisted to ./chroma_db)
  -> retriever (top 4) -> create_stuff_documents_chain(ChatOpenAI gpt-4o-mini, temperature=0)
  -> create_retrieval_chain
```

### Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then put your real OPENAI_API_KEY in .env
```

### Run

```bash
python src/main.py            # builds the index on first run, reuses it after that
python src/main.py --rebuild  # re-embed the PDF (e.g. after changing chunking)
```

For each question the CLI prints the answer, followed by the chunks it retrieved from ChromaDB with their page numbers. Type `exit` or `quit` to leave.

> In LangChain 1.x, `create_retrieval_chain` and `create_stuff_documents_chain` moved to the `langchain-classic` package, which is why it appears in `requirements.txt`.

### Testing

See [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for the full plan. In short:

```bash
pip install -r requirements-dev.txt
pytest                         # offline tests, no API key needed
pytest -m live                 # golden questions against real OpenAI models
python evals/run_eval.py       # metrics, quality gates and report in evals/results/
```

## Hybrid search benchmark: pgvector vs Qdrant

A fully local experiment (no API keys) that indexes the same chunks in both engines and races
them on **hybrid search**: a dense-vector branch and a keyword branch, each under the same
category filter, fused with Reciprocal Rank Fusion (RRF).

| | pgvector | Qdrant |
|---|---|---|
| Dense branch | HNSW `vector_cosine_ops`, `<=>` | HNSW on the `dense` named vector |
| Keyword branch | Postgres full-text (`tsvector` + GIN), ranked by `ts_rank_cd` | BM25 on the `bm25` sparse vector (IDF applied server-side) |
| Filter | `WHERE category = ANY(...)` in both branches | payload filter on both `prefetch` branches |
| Fusion | RRF in SQL, k=60, 40 candidates per branch | Query API `RrfQuery`, k=60, 40 candidates per branch |

### Latest results

113,000 chunks (`--scale 100`), 100 queries x 5 runs, MiniLM embeddings, pgvector 0.8.7 vs
Qdrant 1.20.0 in Docker on an Apple-silicon Mac. Values are the median across runs, with the
min–max range in parentheses. Full report: [`benchmarks/2026-11-01_scale100.md`](benchmarks/2026-11-01_scale100.md).

| Metric | pgvector | Qdrant |
|---|---:|---:|
| Query latency P50 (ms) | 36.05 (33.60–37.68) | 3.33 (2.74–9.21) |
| Query latency P95 (ms) | 68.02 (61.84–69.12) | 4.04 (3.94–26.69) |
| DB load (s) | 7.27 | 15.62 |
| Index build (s) | 12.42 | 11.02 |
| Dense branch recall@10 vs exact search | 95.3% | 97.2% |

- Both engines use filtered HNSW at this size and return all 40 dense candidates, with similar
  recall, so the latency gap is not from one side cutting accuracy.
- pgvector's latency is most likely dominated by its keyword branch (see the notes below), not HNSW.
- Qdrant loads more slowly because of the sparse-vector upserts.
- Qdrant had one slow run (P50 run-to-run variation 62.6%). The likely cause is background segment
  merging right after the index build; a query-only rerun with `--runs 10` should confirm it.
- The extra 111,870 chunks are synthetic (see `--scale` below), and this is a single index build.

### Files

| File | Role |
|---|---|
| `docker-compose.yml` | `pgvector/pgvector:pg16` on `${PG_HOST_PORT:-5432}` (`shm_size: 1g` for parallel HNSW builds), `qdrant/qdrant:latest` on 6333 (REST) / 6334 (gRPC) |
| `generate_docs.py` | Builds 50 synthetic policy PDFs (10 categories x 5) + `docs/categories.json` |
| `ingest.py` | PDF -> 500-char chunks (100 overlap) -> `all-MiniLM-L6-v2` 384-d embeddings + metadata (`doc_id`, `filename`, `page_number`, `category`) |
| `index_db.py` | pgvector: `vector(384)`, generated `tsvector`, binary `COPY`, then HNSW, GIN on the tsvector, btree on `category`. Qdrant: `dense` (Cosine) + `bm25` sparse vector (`modifier=IDF`), keyword payload index on `category` |
| `sparse.py` | BM25 encoding for Qdrant: Snowball stemming and stopwords matching Postgres `english`, term ids hashed with CRC32 |
| `hybrid_query.py` | The hybrid query for each engine, plus the diagnostics (EXPLAIN plan, dense branch on its own) |
| `benchmark.py` | Runs everything and writes `results/benchmark.md` + `results/benchmark.json` (gitignored) |
| `common.py` | Shared settings, `scale_corpus()` |
| `benchmarks/` | Dated copies of results worth keeping, committed by hand |

### Run it

```bash
docker compose up -d
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-bench.txt
python generate_docs.py
python benchmark.py --rebuild
python benchmark.py --scale 100
```

The first run downloads MiniLM (~90 MB) from Hugging Face. Skip `generate_docs.py`
if you put your own PDFs in `./docs/`. `--rebuild` ingests the PDFs and builds both indexes;
after that, runs reuse the indexes unless the scale or the ingested data changes. Step by step:
`python ingest.py`, `python index_db.py [--scale N]`, `python benchmark.py`.

If you already run Postgres locally on 5432 (you'll see `role "rag" does not exist`), put the
container on another port and point the scripts at it. The `export` lasts for the current
terminal only.

```bash
PG_HOST_PORT=5433 docker compose up -d
export PG_DSN=postgresql://rag:rag@localhost:5433/rag
```

> zsh treats `#` as a literal, not a comment, when you paste commands, so paste the lines above
> without trailing comments.

### `benchmark.py` options

| Option | Default | What it does |
|---|---|---|
| `--rebuild` | off | Run `ingest.py`, then rebuild both indexes |
| `--reindex` | off | Rebuild both indexes from the existing `artifacts/` |
| `--scale N` | 1 | Multiply the corpus N times (use 50–100 to exercise HNSW); reindexes automatically when it changes |
| `--scale-noise` | 0.01 | Embedding noise added to synthetic chunks |
| `--runs R` | 5 | Timed query runs per engine |
| `--index-runs K` | 1 | Index builds per engine, to get a spread on load/index times (forces a reindex if > 1) |
| `--queries` / `--warmup` / `--seed` | 100 / 10 / 7 | Query count, warm-up queries before each run, query generator seed |

Environment variables: `PG_DSN`, `PG_HOST_PORT`, `PG_MAINTENANCE_WORK_MEM` (default `1GB`),
`BM25_K1` / `BM25_B` (1.2 / 0.75), `QDRANT_HOST`, `QDRANT_PREFER_GRPC` (default `1`).

### How it's measured

- Both engines use HNSW `m=16, ef_construction=64, ef_search=64`.
- **Load and index are timed separately.** pgvector bulk-loads with binary `COPY`, then runs
  `CREATE INDEX` with `maintenance_work_mem` raised so the HNSW graph fits in memory. Qdrant uploads
  with HNSW indexing disabled, then enables it and waits until the dense and sparse indexes are
  built and the collection is green. Qdrant's load includes client-side BM25 encoding, since
  Postgres computes its `tsvector` inside `COPY`.
- **Query latency** is client-side time around the database call. Query embeddings are computed once
  up front and reported separately. Keyword query parsing is inside the timed call on both sides.
- **Repeats:** each run shuffles the query order, alternates which engine goes first and does 10
  warm-up queries. The report shows the median across runs with the min–max range and the
  run-to-run variation of P50; every run's numbers are in the JSON.
- **Checks in every report:** all results must honour the category filter; the "Access path" lines
  show whether pgvector's plan uses the HNSW index and whether Qdrant's filter matches more vectors
  than its `full_scan_threshold`; the "Dense branch alone" rows give each engine's rows returned and
  recall@10 against brute force.

### Things to know when reading the numbers

- **The keyword scorers differ.** Postgres `ts_rank_cd` vs real BM25 in Qdrant, so the engines'
  top-10 lists don't fully overlap (58% at scale 100).
- **`ts_rank_cd` has no top-k pruning.** pgvector's keyword branch scores every row in the filtered
  category that matches any query word (OR semantics), and words like "policy" match nearly all of
  them. At `--scale 50+` this is likely the biggest part of pgvector's latency. Qdrant's sparse
  index doesn't pay this cost.
- **At ~1.1k chunks neither engine uses HNSW.** A single-category filter keeps ~10% of rows, and both
  planners scan that subset exactly. Use `--scale 50`–`100`.
- **`--scale` chunks are synthetic.** Each extra chunk blends two real chunks from the same category
  (embedding and text), keeping the category metadata, so filter selectivity stays ~10%. Plain
  noisy copies were tried first: they clump together, split the HNSW graph into islands, and
  filtered searches return nothing.
- **Check the "Dense branch alone" rows.** The keyword branch always fills the fused top-10, so a weak
  vector search is otherwise invisible in "avg results returned".
- **Smoke-testing without Docker or Hugging Face:** `EMBED_BACKEND=hash QDRANT_LOCAL_PATH=./qlocal`
  swaps in hash embeddings and Qdrant's in-process mode. The output is flagged and is not a benchmark.
