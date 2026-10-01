import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "evals"))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fakes import BagOfWordsEmbeddings, RecordingLLM  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402

PDF_PATH = ROOT / "data" / "Wacky Acme Company Policy.pdf"


@pytest.fixture(scope="session")
def offline_engine(tmp_path_factory):
    """Real loader/splitter/Chroma/chain wiring with fake embeddings and LLM."""
    llm = RecordingLLM()
    engine = RAGEngine(
        pdf_path=PDF_PATH,
        persist_dir=tmp_path_factory.mktemp("chroma"),
        rebuild=True,
        embeddings=BagOfWordsEmbeddings(),
        llm=llm,
    )
    engine.fake_llm = llm
    return engine


@pytest.fixture(scope="session")
def live_engine(tmp_path_factory):
    """Real OpenAI engine; tests using it are skipped without an API key."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping live tests")
    return RAGEngine(persist_dir=tmp_path_factory.mktemp("chroma_live"), rebuild=True)
