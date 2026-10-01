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
