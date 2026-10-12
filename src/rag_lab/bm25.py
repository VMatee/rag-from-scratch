"""Small, deterministic BM25 implementation with document provenance."""
from collections import Counter
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
    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require k1 > 0 and 0 <= b <= 1")
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("Document ids must be unique")
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.counts = [Counter(tokenize(f"{doc.title} {doc.text}")) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.doc_freq = Counter(term for counts in self.counts for term in counts)

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = set(tokenize(query))
        if not terms or not self.documents or not self.avg_len:
            return []
        number_docs = len(self.documents)
        results = []
        for doc, counts, length in zip(self.documents, self.counts, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                frequency = self.doc_freq[term]
                idf = math.log1p((number_docs - frequency + 0.5) / (frequency + 0.5))
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += idf * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                results.append(SearchHit(doc, score))
        return sorted(results, key=lambda hit: (-hit.score, hit.doc.id))[:top_k]
