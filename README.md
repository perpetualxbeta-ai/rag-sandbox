# rag-sandbox

A RAG proof of concept over the *Wacky Acme Company Policy* handbook, built with LangChain, Chroma and OpenAI.

```
data/Wacky Acme Company Policy.pdf
  -> PyPDFLoader (one Document per page)
  -> RecursiveCharacterTextSplitter (chunk_size=600, chunk_overlap=100)
  -> OpenAIEmbeddings -> Chroma (persisted to ./chroma_db)
  -> retriever (top 4) -> create_stuff_documents_chain(ChatOpenAI gpt-4o-mini, temperature=0)
  -> create_retrieval_chain
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then put your real OPENAI_API_KEY in .env
```

## Run

```bash
python src/main.py            # builds the index on first run, reuses it after that
python src/main.py --rebuild  # re-embed the PDF (e.g. after changing chunking)
```

For each question the CLI prints the answer, followed by the chunks it retrieved from ChromaDB with their page numbers. Type `exit` or `quit` to leave.

> In LangChain 1.x, `create_retrieval_chain` and `create_stuff_documents_chain` moved to the `langchain-classic` package, which is why it appears in `requirements.txt`.

## Testing

See [docs/TEST_PLAN.md](docs/TEST_PLAN.md) for the full plan. In short:

```bash
pip install -r requirements-dev.txt
pytest                         # offline tests, no API key needed
pytest -m live                 # golden questions against real OpenAI models
python evals/run_eval.py       # metrics, quality gates and report in evals/results/
```

## Hybrid search benchmark: pgvector vs Qdrant

A separate, fully local experiment (no API keys) that indexes the same chunks in
both engines and races them.

| File | Role |
|---|---|
| `docker-compose.yml` | `pgvector/pgvector:pg16` on 5432, `qdrant/qdrant:latest` on 6333 (REST) / 6334 (gRPC) |
| `generate_docs.py` | Builds 50 synthetic policy PDFs (10 categories x 5) + `docs/categories.json` |
| `ingest.py` | PDF -> 500-char chunks (100 overlap) -> `all-MiniLM-L6-v2` 384-d embeddings + metadata (`doc_id`, `filename`, `page_number`, `category`) |
| `index_db.py` | pgvector: `vector(384)`, generated `tsvector`, HNSW `vector_cosine_ops`, GIN on tsvector. Qdrant: Cosine collection + keyword payload index on `category` |
| `sparse.py` | BM25 sparse vectors for Qdrant (Snowball stemming + stopwords like Postgres `english`; IDF applied server-side) |
| `hybrid_query.py` | pgvector RRF SQL (`<=>` + `ts_rank_cd`, `WHERE category = ANY(...)`); Qdrant `prefetch` dense + BM25 sparse, fused with RRF (k=60), payload filter on both branches |
| `benchmark.py` | 100 queries x `--runs` repeats, optional `--scale N`; writes a Markdown table with median (min–max) to `results/benchmark.md` |

```bash
docker compose up -d
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-bench.txt     # first run downloads MiniLM (~90 MB) from Hugging Face
python generate_docs.py                   # skip if you put your own 50 PDFs in ./docs/
python benchmark.py --rebuild             # ingest + index + 100 queries x 5 runs
python benchmark.py --scale 100           # ~113k chunks: reindexes, then 5 timed runs
python benchmark.py --scale 100 --runs 10 --index-runs 3   # more runs, spread on load/index too
```

Or step by step: `python ingest.py`, `python index_db.py`, `python benchmark.py`.

If you already run Postgres locally on 5432 (you'll see `role "rag" does not exist`), put the
container on another port and point the scripts at it:

```bash
PG_HOST_PORT=5433 docker compose up -d
export PG_DSN=postgresql://rag:rag@localhost:5433/rag
python benchmark.py --rebuild
```

**How it's measured.** Both engines use HNSW `m=16, ef_construction=64, ef_search=64`.
Loading and indexing are timed separately: pgvector bulk-loads with `COPY` and then runs
`CREATE INDEX`; Qdrant uploads with HNSW indexing disabled, then enables it and waits for
the collection to go green. Shared preprocessing (parse/chunk/embed) is counted once and
added to each engine's load time for "total ingestion". Query latency is client-side
time around the DB call only (query embeddings precomputed), after 10 warm-up queries.

**Things to know when reading the numbers.**
- Both sides are true hybrid: dense HNSW + a keyword branch, RRF-fused with k=60 and 40 candidates
  per branch, filter applied inside each branch. The keyword *scorers* still differ: Postgres
  `ts_rank_cd` vs real BM25 in Qdrant, so top-10 overlap stays below 100%.
- `ts_rank_cd` has no top-k pruning: the keyword branch scores every row in the filtered category
  that matches any query term (OR semantics). At `--scale 50+` that is usually the biggest part of
  pgvector's latency, not HNSW. Qdrant's sparse inverted index does not have this cost.
- At ~1.1k chunks, both planners skip HNSW for a single-category filter (~10% of rows) and scan the
  filtered subset exactly. Use `--scale 50`-`100`: the report's "Access path" lines show whether
  pgvector's plan uses the HNSW index and whether Qdrant's filter cardinality is above its
  `full_scan_threshold` (10 MB by default, so ~6.8k x 384-d vectors).
- `--scale N` replicas are same-category blends of two real chunks (embedding and text), not noisy
  copies. Copies form tight clusters that fragment the HNSW graph into islands, and filtered
  searches then return nothing.
- "Dense branch alone" rows show, untimed, how many of the 40 candidates each engine's filtered
  HNSW returns and its recall@10 against brute force. Check them: the keyword branch always fills
  the fused top-10, so a weak ANN path is otherwise invisible.
- Qdrant's load time includes client-side BM25 encoding and sparse-vector upserts, which grow
  faster than linearly with collection size; the load breakdown splits them out.
- Smoke-testing without Docker/Hugging Face: `EMBED_BACKEND=hash QDRANT_LOCAL_PATH=./qlocal`
  swaps in hash embeddings and Qdrant's in-process mode. The output is flagged and is not a benchmark.
