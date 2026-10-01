"""Interactive CLI for asking questions about the Wacky Acme Company Policy.

Usage:
    python src/main.py            # reuse the existing Chroma index if present
    python src/main.py --rebuild  # re-embed the PDF from scratch
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from rag_engine import RAGEngine

DIVIDER = "-" * 72


def format_source(doc, index: int) -> str:
    # PyPDFLoader stores a 0-based page index; show it 1-based like a PDF viewer.
    page = doc.metadata.get("page")
    page_label = doc.metadata.get("page_label") or (page + 1 if page is not None else "?")
    body = textwrap.indent(textwrap.fill(" ".join(doc.page_content.split()), 88), "    ")
    return f"  [{index}] Page {page_label}\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about the Wacky Acme policy.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the Chroma index.")
    args = parser.parse_args()

    print("Initialising RAG engine...")
    try:
        engine = RAGEngine(rebuild=args.rebuild)
    except (EnvironmentError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if engine.loaded_from_disk:
        print(f"Loaded existing Chroma index ({engine.num_chunks} chunks).")
    else:
        print(f"Indexed {engine.num_pages} pages into {engine.num_chunks} chunks.")
    print("Ask a question about the Wacky Acme policy. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = engine.ask(question)
        except Exception as exc:  # network / API errors shouldn't kill the session
            print(f"Error while answering: {exc}\n")
            continue

        print(f"\n{DIVIDER}\nAnswer:\n{result['answer']}\n")
        print("Sources retrieved from ChromaDB:")
        for i, doc in enumerate(result["context"], start=1):
            print(format_source(doc, i))
        print(f"{DIVIDER}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
