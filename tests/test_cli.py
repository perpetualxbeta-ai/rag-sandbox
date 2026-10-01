"""TP-CLI: interactive loop behaviour (offline)."""

import builtins
import sys

import pytest

import main as cli


class StubEngine:
    loaded_from_disk = False
    num_pages, num_chunks = 5, 22

    def __init__(self, *a, fail=False, **k):
        self.fail = fail
        self.questions = []

    def ask(self, q):
        self.questions.append(q)
        if self.fail:
            raise RuntimeError("API down")
        from langchain_core.documents import Document

        return {
            "answer": "The answer.",
            "context": [Document(page_content="Chunk   text\n here", metadata={"page": 1, "page_label": "2"}),
                        Document(page_content="No label", metadata={"page": 4})],
        }


def run_cli(monkeypatch, inputs, engine_factory=StubEngine):
    feed = iter(inputs)

    def fake_input(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    created = []

    def factory(*a, **k):
        created.append(engine_factory(*a, **k))
        return created[-1]

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(cli, "RAGEngine", factory)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    return cli.main(), created


@pytest.mark.parametrize("word", ["exit", "quit", "EXIT", "  Quit  "])
def test_cli_01_exit_words(monkeypatch, capsys, word):
    code, engines = run_cli(monkeypatch, [word, "should never be asked"])
    assert code == 0
    assert engines[0].questions == []
    assert "Goodbye" in capsys.readouterr().out


def test_cli_02_eof_exits_cleanly(monkeypatch, capsys):
    code, _ = run_cli(monkeypatch, [])
    assert code == 0 and "Goodbye" in capsys.readouterr().out


def test_cli_03_blank_input_ignored(monkeypatch):
    _, engines = run_cli(monkeypatch, ["", "   ", "q1", "exit"])
    assert engines[0].questions == ["q1"]


def test_cli_04_prints_answer_and_sources_with_pages(monkeypatch, capsys):
    run_cli(monkeypatch, ["q1", "exit"])
    out = capsys.readouterr().out
    assert "Answer:\nThe answer." in out
    assert "Sources retrieved from ChromaDB:" in out
    assert "[1] Page 2" in out           # page_label used
    assert "[2] Page 5" in out           # 0-based page converted to 1-based
    assert "Chunk text here" in out      # whitespace collapsed


def test_cli_05_loop_survives_answer_errors(monkeypatch, capsys):
    code, engines = run_cli(monkeypatch, ["q1", "q2", "quit"],
                            engine_factory=lambda *a, **k: StubEngine(fail=True))
    assert code == 0
    assert engines[0].questions == ["q1", "q2"]
    assert capsys.readouterr().out.count("Error while answering") == 2


def test_cli_06_startup_error_returns_1(monkeypatch, capsys):
    def boom(*a, **k):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    code, _ = run_cli(monkeypatch, ["exit"], engine_factory=boom)
    assert code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
