"""TP-ING: PDF loading and chunking (offline)."""

import unicodedata

import rag_engine


def test_ing_01_pdf_loads_all_pages(offline_engine):
    pages = rag_engine.PyPDFLoader(str(offline_engine.pdf_path)).load()
    assert len(pages) == 5
    assert all(p.page_content.strip() for p in pages), "a page extracted as empty text"


def test_ing_02_chunk_size_respected(offline_engine):
    chunks = offline_engine.load_and_split()
    assert chunks, "no chunks produced"
    assert max(len(c.page_content) for c in chunks) <= rag_engine.CHUNK_SIZE


def test_ing_03_chunks_overlap(offline_engine):
    chunks = offline_engine.load_and_split()
    same_page_pairs = [
        (a, b) for a, b in zip(chunks, chunks[1:]) if a.metadata["page"] == b.metadata["page"]
    ]
    overlapping = [b for a, b in same_page_pairs if b.page_content[:40] in a.page_content]
    assert overlapping, "adjacent chunks on the same page never share text"


def test_ing_04_every_chunk_has_page_metadata(offline_engine):
    chunks = offline_engine.load_and_split()
    pages = {c.metadata.get("page") for c in chunks}
    assert pages == {0, 1, 2, 3, 4}
    assert all(c.metadata.get("source") for c in chunks)


def test_ing_05_all_sections_indexed(offline_engine):
    text = unicodedata.normalize(
        "NFKC", " ".join(c.page_content for c in offline_engine.load_and_split())
    )
    for heading in ["1.0 Welcome", "3.1 Tardiness", "5.1 Refrigerator", "7.1 Password",
                    "9.0 Dispute", "10.3 Spontaneous", "11.2 Accidental"]:
        assert heading in text, f"section '{heading}' missing from chunks"


def test_ing_06_vector_count_matches_chunks(offline_engine):
    assert offline_engine.vectorstore._collection.count() == offline_engine.num_chunks
