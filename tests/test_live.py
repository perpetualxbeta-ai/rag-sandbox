"""TP-LIVE: end-to-end checks against real OpenAI models.

Skipped automatically when OPENAI_API_KEY is not set. Run with:
    pytest -m live
"""

import json
from pathlib import Path

import pytest

from run_eval import is_refusal, score_case  # noqa: E402  (evals/ added to path below)

pytestmark = pytest.mark.live

CASES = json.loads((Path(__file__).resolve().parent.parent / "evals" / "golden_qa.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_live_golden_case(live_engine, case):
    result = live_engine.ask(case["question"])
    pages = [d.metadata["page"] + 1 for d in result["context"]]
    s = score_case(case, result["answer"], pages)
    assert s["retrieval_ok"] is not False, f"expected pages {case['expected_pages']}, got {pages}"
    assert s["refusal_ok"], f"refusal expected={case['expect_refusal']}: {result['answer']}"
    assert s["keywords_ok"], f"missing {s['missing_keywords']}: {result['answer']}"
    assert not s["trap_hits"], f"hallucination trap {s['trap_hits']}: {result['answer']}"


def test_live_answers_are_deterministic(live_engine):
    q = CASES[0]["question"]
    a1, a2 = live_engine.ask(q)["answer"], live_engine.ask(q)["answer"]
    assert a1 == a2 or not (is_refusal(a1) ^ is_refusal(a2)), "temperature=0 answers diverged materially"
