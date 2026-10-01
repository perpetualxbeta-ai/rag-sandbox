"""Score the RAG engine against the golden question set (evals/golden_qa.json).

    python evals/run_eval.py                 # live run against OpenAI (needs OPENAI_API_KEY)
    python evals/run_eval.py --judge         # also grade faithfulness with an LLM judge
    python evals/run_eval.py --offline       # harness smoke test with fake models, no key needed
    python evals/run_eval.py --only GQ-01    # run a subset

Writes evals/results/<timestamp>.json and .md and exits non-zero if any
quality gate in docs/TEST_PLAN.md fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore")

from rag_engine import RAGEngine  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_qa.json"
RESULTS_DIR = ROOT / "evals" / "results"

# Quality gates (keep in sync with docs/TEST_PLAN.md section 6).
GATES = {
    "retrieval_hit_rate": 0.90,   # answerable cases whose expected pages were all retrieved
    "answer_accuracy": 0.85,      # answerable cases whose answer contains every keyword group
    "refusal_accuracy": 1.00,     # cases correctly refused / correctly answered
    "hallucination_traps": 1.00,  # cases with no must_not_include string in the answer
    "faithfulness_avg": 4.0,      # 1-5 LLM-judge score (only with --judge)
}

REFUSAL_MARKERS = [
    "don't know", "do not know", "not sure", "does not mention", "doesn't mention",
    "does not specify", "doesn't specify", "not specified", "does not contain",
    "doesn't contain", "no information", "not provided", "not covered", "does not cover",
    "doesn't cover", "does not say", "doesn't say", "not mentioned", "does not include",
    "doesn't include", "not stated", "does not state", "doesn't state", "unable to find",
    "not addressed", "does not address", "doesn't address", "no mention",
]

JUDGE_PROMPT = """You are grading a RAG system. Given handbook EXCERPTS, a QUESTION and an ANSWER,
rate how faithful the ANSWER is to the EXCERPTS on a 1-5 scale:
5 = every claim is supported by the excerpts; 3 = mostly supported with minor unsupported details;
1 = major claims are unsupported or contradict the excerpts. Saying the excerpts don't cover the
question is faithful if that is true. Reply with JSON only: {{"score": <int>, "reason": "<short>"}}

EXCERPTS:
{context}

QUESTION: {question}

ANSWER: {answer}"""


def norm(text: str) -> str:
    # NFKC folds PDF ligatures (e.g. U+FB01 "fi"); collapse line breaks from PDF extraction.
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def is_refusal(answer: str) -> bool:
    a = norm(answer).replace("’", "'")
    return any(m in a for m in REFUSAL_MARKERS)


def score_case(case: dict, answer: str, pages: list[int]) -> dict:
    a = norm(answer)
    retrieved = set(pages)
    page_groups = case["expected_pages"]
    retrieval_ok = all(any(p in retrieved for p in group) for group in page_groups)
    missing = [g for g in case["must_include"] if not any(norm(v) in a for v in g)]
    traps = [s for s in case["must_not_include"] if norm(s) in a]
    refused = is_refusal(answer)
    return {
        "retrieval_ok": retrieval_ok if page_groups else None,
        "keywords_ok": not missing,
        "missing_keywords": missing,
        "trap_hits": traps,
        "refused": refused,
        "refusal_ok": refused == case["expect_refusal"],
    }


def judge(llm, question: str, answer: str, docs) -> dict:
    context = "\n---\n".join(d.page_content for d in docs)
    raw = llm.invoke(JUDGE_PROMPT.format(context=context, question=question, answer=answer)).content
    try:
        return json.loads(raw.strip().strip("`").removeprefix("json"))
    except json.JSONDecodeError:
        return {"score": None, "reason": f"unparseable judge output: {raw[:120]}"}


def build_engine(offline: bool) -> RAGEngine:
    persist = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    if offline:
        from fakes import BagOfWordsEmbeddings, ExtractiveLLM

        return RAGEngine(persist_dir=persist, rebuild=True,
                         embeddings=BagOfWordsEmbeddings(), llm=ExtractiveLLM())
    return RAGEngine(persist_dir=persist, rebuild=True)


def summarise(rows: list[dict], use_judge: bool) -> dict:
    answerable = [r for r in rows if not r["expect_refusal"]]
    with_pages = [r for r in rows if r["retrieval_ok"] is not None]
    pct = lambda xs, k: (sum(1 for x in xs if x[k]) / len(xs)) if xs else 1.0  # noqa: E731
    metrics = {
        "retrieval_hit_rate": pct(with_pages, "retrieval_ok"),
        "answer_accuracy": pct(answerable, "keywords_ok"),
        "refusal_accuracy": pct(rows, "refusal_ok"),
        "hallucination_traps": sum(1 for r in rows if not r["trap_hits"]) / len(rows),
        "avg_latency_s": sum(r["latency_s"] for r in rows) / len(rows),
    }
    if use_judge:
        scores = [r["judge"]["score"] for r in rows if r.get("judge", {}).get("score")]
        metrics["faithfulness_avg"] = sum(scores) / len(scores) if scores else 0.0
    gates = {k: metrics[k] >= v for k, v in GATES.items() if k in metrics}
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def write_report(stamp: str, mode: str, rows: list[dict], summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = RESULTS_DIR / f"{stamp}_{mode}"
    base.with_suffix(".json").write_text(json.dumps({"mode": mode, **summary, "cases": rows}, indent=2))

    tick = lambda ok: "n/a" if ok is None else ("PASS" if ok else "FAIL")  # noqa: E731
    lines = [f"# RAG eval - {stamp} ({mode})", "",
             f"**Overall: {'PASS' if summary['passed'] else 'FAIL'}**", "",
             "| Metric | Value | Gate | Result |", "|---|---|---|---|"]
    for k, v in summary["metrics"].items():
        gate = GATES.get(k)
        lines.append(f"| {k} | {v:.2f} | {'>= %.2f' % gate if gate else '-'} | "
                     f"{tick(summary['gates'].get(k)) if gate else '-'} |")
    lines += ["", "| ID | Category | Retrieval | Keywords | Refusal | Traps | Pages | Notes |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        notes = []
        if r["missing_keywords"]:
            notes.append("missing " + "; ".join("/".join(g) for g in r["missing_keywords"]))
        if r["trap_hits"]:
            notes.append("trap " + ", ".join(r["trap_hits"]))
        if r.get("judge"):
            notes.append(f"judge {r['judge'].get('score')}")
        lines.append(f"| {r['id']} | {r['category']} | {tick(r['retrieval_ok'])} | "
                     f"{tick(r['keywords_ok'])} | {tick(r['refusal_ok'])} | {tick(not r['trap_hits'])} | "
                     f"{r['retrieved_pages']} | {' / '.join(notes)} |")
    lines += ["", "## Answers", ""]
    for r in rows:
        lines += [f"**{r['id']}** {r['question']}", "", "> " + r["answer"].replace("\n", "\n> "), ""]
    md = base.with_suffix(".md")
    md.write_text("\n".join(lines))
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="fake models; smoke-tests the harness only")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness scoring")
    ap.add_argument("--only", nargs="*", help="case IDs to run")
    args = ap.parse_args()

    cases = json.loads(GOLDEN.read_text())["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    engine = build_engine(args.offline)
    judge_llm = None
    if args.judge and not args.offline:
        from langchain_openai import ChatOpenAI
        judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rows = []
    for case in cases:
        t0 = time.perf_counter()
        result = engine.ask(case["question"])
        latency = time.perf_counter() - t0
        pages = [d.metadata["page"] + 1 for d in result["context"]]
        row = {**case, "answer": result["answer"], "retrieved_pages": pages,
               "latency_s": round(latency, 3), **score_case(case, result["answer"], pages)}
        if judge_llm:
            row["judge"] = judge(judge_llm, case["question"], result["answer"], result["context"])
        rows.append(row)
        flag = "ok " if row["keywords_ok"] and row["refusal_ok"] and row["retrieval_ok"] is not False else "BAD"
        print(f"[{flag}] {case['id']} pages={pages} {latency:.1f}s")

    summary = summarise(rows, use_judge=bool(judge_llm))
    mode = "offline" if args.offline else "live"
    report = write_report(datetime.now().strftime("%Y%m%d-%H%M%S"), mode, rows, summary)
    print("\n" + "\n".join(f"{k}: {v:.2f}" for k, v in summary["metrics"].items()))
    print(f"\nOverall: {'PASS' if summary['passed'] else 'FAIL'}  ->  {report.relative_to(ROOT)}")
    if args.offline:
        print("(offline mode uses fake models - only retrieval numbers are meaningful)")
        return 0
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
