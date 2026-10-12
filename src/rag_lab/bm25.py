"""A hand-written BM25 inverted index with source provenance.

Only Python's standard library is used; there is no vector DB, Lucene,
sentence-transformer, or external retrieval framework.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Sequence


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class SearchHit:
    doc: Document
    score: float


class BM25Index:
    """Cache term frequencies and document-length norms at index-build time."""

    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and 0 <= b <= 1")
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("Document ids must be unique")
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.counts = [Counter(tokenize(f"{doc.title} {doc.text}")) for doc in self.documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.doc_freq = Counter(term for counts in self.counts for term in counts)
        self._lookup = {doc.id: index for index, doc in enumerate(self.documents)}
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_index, frequencies in enumerate(self.counts):
            for term, tf in frequencies.items():
                self._postings[term].append((doc_index, tf))
        self._normalizers = [
            k1 * (1 - b + b * length / self.avg_len) if self.avg_len else 0.0
            for length in self.lengths
        ]
        total = len(self.documents)
        self._idf = {
            term: math.log1p((total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in self.doc_freq.items()
        }

    def _term_score(self, term: str, doc_index: int, tf: int) -> float:
        return self._idf[term] * (tf * (self.k1 + 1)) / (tf + self._normalizers[doc_index])

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.documents or not self.avg_len:
            return []
        # Only inspect documents containing query terms, never scan all documents.
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            for doc_index, tf in self._postings.get(term, ()):
                scores[doc_index] += self._term_score(term, doc_index, tf)
        indices = sorted(
            (i for i, score in scores.items() if score > 0),
            key=lambda i: (-scores[i], self.documents[i].id),
        )[:top_k]
        return [SearchHit(self.documents[i], scores[i]) for i in indices]

    def explain(self, query: str, document_id: str) -> dict[str, float]:
        """Return a reproducible BM25 score contribution for each query term."""
        if document_id not in self._lookup:
            raise KeyError(f"Unknown document: {document_id}")
        i = self._lookup[document_id]
        result = {}
        for term in sorted(set(tokenize(query))):
            tf = self.counts[i].get(term, 0)
            if tf:
                result[term] = self._term_score(term, i, tf)
        return result
