"""Small transparent retrieval hit@k demo, NOT an external benchmark."""
from pathlib import Path
from . import BM25Index, load_documents


EXAMPLES = [
    ("How do I record a change in Git history?", "git-commit"),
    ("What does git clone do?", "git-clone"),
    ("How do I submit code for review on GitHub?", "pull-request"),
    ("How can I navigate folders in the Linux terminal?", "linux-cd"),
]


def run() -> float:
    path = Path(__file__).resolve().parents[2] / "data" / "documents.jsonl"
    index = BM25Index(load_documents(path))
    correct = sum(expected in [h.doc.id for h in index.search(query, 3)] for query, expected in EXAMPLES)
    value = correct / len(EXAMPLES)
    print(f"Hit@3 on {len(EXAMPLES)} illustrative examples: {correct}/{len(EXAMPLES)} ({value:.1%})")
    return value


if __name__ == "__main__":
    run()
