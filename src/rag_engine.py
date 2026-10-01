"""RAG engine for the Wacky Acme Company Policy handbook.

Pipeline: PyPDFLoader -> RecursiveCharacterTextSplitter -> OpenAIEmbeddings
-> local Chroma vector store -> retriever -> stuff-documents chain over
ChatOpenAI(gpt-4o-mini, temperature=0).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# langchain-community is in maintenance mode and warns on import; keep the CLI quiet.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Chroma.*deprecated.*")

from langchain_community.document_loaders import PyPDFLoader  # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

# LangChain >= 1.0 moved the legacy chain constructors into `langchain-classic`.
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:  # LangChain 0.2 / 0.3
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "Wacky Acme Company Policy.pdf"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "wacky_acme_policy"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 4

SYSTEM_PROMPT = (
    "You are an HR assistant for WackyAcme Corporation. Answer the employee's "
    "question using ONLY the handbook excerpts provided below. Quote specific "
    "rules, times and consequences where they appear. If the excerpts do not "
    "contain the answer, say you don't know rather than guessing.\n\n"
    "Handbook excerpts:\n{context}"
)


class RAGEngine:
    """Loads the policy PDF, indexes it in Chroma and answers questions about it."""

    def __init__(
        self,
        pdf_path: Path | str = DEFAULT_PDF_PATH,
        persist_dir: Path | str = DEFAULT_PERSIST_DIR,
        rebuild: bool = False,
        embeddings=None,
        llm=None,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        if (embeddings is None or llm is None) and not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to the .env file in the project root."
            )

        self.pdf_path = Path(pdf_path)
        self.persist_dir = Path(persist_dir)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Policy PDF not found at {self.pdf_path}")

        self.embeddings = embeddings or OpenAIEmbeddings()
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.vectorstore = self._build_or_load_vectorstore(rebuild)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        self.chain = self._build_chain()

    # ------------------------------------------------------------------ indexing
    def load_and_split(self):
        """Load the PDF (one Document per page) and split it into chunks."""
        pages = PyPDFLoader(str(self.pdf_path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(pages)
        self.num_pages, self.num_chunks = len(pages), len(chunks)
        return chunks

    def _build_or_load_vectorstore(self, rebuild: bool) -> Chroma:
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        # Reuse an existing index unless asked to rebuild (saves embedding calls).
        if not rebuild and store._collection.count() > 0:
            self.num_pages, self.num_chunks = None, store._collection.count()
            self.loaded_from_disk = True
            return store

        # Clear through Chroma rather than deleting files: chromadb caches clients
        # per path, so removing the folder under a live client corrupts later writes.
        store.delete_collection()
        chunks = self.load_and_split()
        self.loaded_from_disk = False
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(self.persist_dir),
        )

    # --------------------------------------------------------------------- chain
    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{input}")]
        )
        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(self.retriever, qa_chain)

    def ask(self, question: str) -> dict:
        """Return {'answer': str, 'context': [Document, ...]} for a question."""
        result = self.chain.invoke({"input": question})
        return {"answer": result["answer"], "context": result["context"]}
