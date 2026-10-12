"""Source-aware context assembly. Evidence display is not text generation."""
import json
from pathlib import Path
from typing import Sequence

from .bm25 import Document, SearchHit


def load_documents(path: str | Path) -> list[Document]:
    documents = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
            documents.append(Document(**{key: item[key] for key in ("id", "title", "text", "source")}))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid JSONL document at line {number}") from exc
    return documents


def format_context(hits: Sequence[SearchHit]) -> str:
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(
        f"[DOC:{hit.doc.id}] {hit.doc.title}\nSource: {hit.doc.source}\n{hit.doc.text}"
        for hit in hits
    )


def evidence_answer(hits: Sequence[SearchHit]) -> str:
    if not hits:
        return "I cannot find grounded evidence in this small local corpus."
    return "Retrieved evidence (NOT a generated answer):\n" + format_context(hits)
