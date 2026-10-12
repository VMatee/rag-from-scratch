import argparse
from pathlib import Path
from . import BM25Index, load_documents, format_context, evidence_answer
from .ollama import generate_local


def main():
    parser = argparse.ArgumentParser(description="BM25 RAG: inspect retrieval or optionally generate via local Ollama")
    parser.add_argument("question")
    parser.add_argument("--documents", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "documents.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model", default="llama3.2:1b")
    args = parser.parse_args()
    hits = BM25Index(load_documents(args.documents)).search(args.question, args.top_k)
    if not args.generate:
        print(evidence_answer(hits))
        return
    if not hits:
        print(evidence_answer(hits))
        return
    print("Evidence:\n" + format_context(hits))
    print("\nLocal model response (verify citations):\n" + generate_local(args.question, format_context(hits), args.model))


if __name__ == "__main__":
    main()
