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
| `hybrid_query.py` | pgvector RRF SQL (`<=>` + `ts_rank_cd`, `WHERE category = ANY(...)`); Qdrant dense query with payload `query_filter` |
| `benchmark.py` | 100 queries, prints/writes a Markdown table to `results/benchmark.md` |

```bash
docker compose up -d
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-bench.txt     # first run downloads MiniLM (~90 MB) from Hugging Face
python generate_docs.py                   # skip if you put your own 50 PDFs in ./docs/
python benchmark.py --rebuild             # ingest + index + 100 queries
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
- The comparison is asymmetric by design (per spec): pgvector fuses dense + full-text via RRF;
  Qdrant runs dense + payload filter only. Qdrant's sparse vectors + `prefetch`/`FusionQuery`
  would be the like-for-like hybrid if you want it.
- At ~1.1k chunks, both planners often skip HNSW for a single-category filter (~10% of rows) and
  do an exact scan of the filtered subset (Postgres via the category btree, Qdrant below its
  `full_scan_threshold`). That's the correct choice at this size; scale the corpus up to
  exercise the ANN paths.
- Smoke-testing without Docker/Hugging Face: `EMBED_BACKEND=hash QDRANT_LOCAL_PATH=./qlocal`
  swaps in hash embeddings and Qdrant's in-process mode. The output is flagged and is not a benchmark.
