"""TP-RET / TP-CHN: retrieval wiring and chain contract (offline)."""

import pytest

import rag_engine
from fakes import BagOfWordsEmbeddings, RecordingLLM

TEST_QUERY = "What is the policy on time loop tardiness and Friday refrigerator purges?"


def pages_for(engine, question):
    return {d.metadata["page"] + 1 for d in engine.retriever.invoke(question)}


def test_ret_01_returns_top_k(offline_engine):
    assert len(offline_engine.retriever.invoke(TEST_QUERY)) == rag_engine.TOP_K


def test_ret_02_reference_query_hits_both_sections(offline_engine):
    docs = offline_engine.retriever.invoke(TEST_QUERY)
    joined = " ".join(d.page_content for d in docs)
    assert "Time Loop Protocol" in joined
    assert "Friday Purge" in joined
    assert {2, 3} <= {d.metadata["page"] + 1 for d in docs}


@pytest.mark.parametrize(
    "question, expected_page",
    [
        ("What happens if the fire is singing show tunes?", 5),
        ("What is the coffee machine Clarence?", 4),
        ("Which pets are prohibited, dragons or dire wolves?", 3),
        ("Rock-Paper-Scissors-Lizard-Spock dispute resolution", 4),
    ],
)
def test_ret_03_lexical_questions_hit_expected_page(offline_engine, question, expected_page):
    assert expected_page in pages_for(offline_engine, question)


def test_chn_01_ask_returns_answer_and_context(offline_engine):
    result = offline_engine.ask(TEST_QUERY)
    assert set(result) == {"answer", "context"}
    assert result["answer"] == "FAKE ANSWER"
    assert len(result["context"]) == rag_engine.TOP_K


def test_chn_02_prompt_contains_context_and_question(offline_engine):
    offline_engine.fake_llm.calls.clear()
    result = offline_engine.ask(TEST_QUERY)
    system, human = offline_engine.fake_llm.calls[-1]
    assert human.content == TEST_QUERY
    for doc in result["context"]:
        assert doc.page_content in system.content, "retrieved chunk not stuffed into prompt"
    assert "ONLY" in system.content and "don't know" in system.content


def test_chn_03_default_model_config():
    llm = rag_engine.ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="sk-test")
    assert llm.model_name == "gpt-4o-mini" and llm.temperature == 0


def test_chn_04_index_is_reused_without_rebuild(tmp_path):
    kw = dict(persist_dir=tmp_path, embeddings=BagOfWordsEmbeddings(), llm=RecordingLLM())
    first = rag_engine.RAGEngine(rebuild=True, **kw)
    second = rag_engine.RAGEngine(**kw)
    assert first.loaded_from_disk is False
    assert second.loaded_from_disk is True
    assert second.vectorstore._collection.count() == first.num_chunks


def test_chn_05_rebuild_does_not_duplicate_vectors(tmp_path):
    kw = dict(persist_dir=tmp_path, embeddings=BagOfWordsEmbeddings(), llm=RecordingLLM())
    a = rag_engine.RAGEngine(rebuild=True, **kw)
    b = rag_engine.RAGEngine(rebuild=True, **kw)
    assert b.vectorstore._collection.count() == a.num_chunks


def test_chn_06_missing_api_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(rag_engine, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(EnvironmentError):
        rag_engine.RAGEngine(persist_dir=tmp_path)


def test_chn_07_missing_pdf_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rag_engine.RAGEngine(pdf_path=tmp_path / "nope.pdf", persist_dir=tmp_path,
                             embeddings=BagOfWordsEmbeddings(), llm=RecordingLLM())
