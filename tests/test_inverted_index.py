"""Check numerical equivalence to full-scan BM25, plus index behavior."""
import math
import random
import pytest
from rag_lab import BM25Index, Document
from rag_lab.bm25 import tokenize


def naive_bm25(documents, query, k1=1.5, b=0.75):
    vocab = [tokenize(f"{doc.title} {doc.text}") for doc in documents]
    length = [len(row) for row in vocab]
    average = sum(length) / len(length) if length else 0
    pairs = []
    for doc, words in zip(documents, vocab):
        if not words:
            continue
        score = 0.0
        for term in sorted(set(tokenize(query))):
            tf = words.count(term)
            if tf == 0:
                continue
            df = sum(term in row for row in vocab)
            idf = math.log1p((len(documents) - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * len(words) / average)
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            pairs.append((doc.id, score))
    return sorted(pairs, key=lambda item: (-item[1], item[0]))


def test_inverted_index_matches_full_scan_scores():
    random.seed(19)
    corpus = ["branch", "git", "linux", "review", "commit", "merge", "code"]
    documents = [
        Document(str(i), " ".join(random.choices(corpus, k=4)),
                 " ".join(random.choices(corpus, k=15)), f"sample:{i}")
        for i in range(130)
    ]
    index = BM25Index(documents)
    for query in ("git merge", "commit commit branch", "linux review", "not-in-corpus"):
        actual = index.search(query, top_k=len(documents))
        expected = naive_bm25(documents, query)
        assert [h.doc.id for h in actual] == [i for i, _ in expected]
        assert [h.score for h in actual] == pytest.approx([v for _, v in expected], rel=1e-12, abs=1e-12)


def test_explain_scores_sum_to_hit_and_reject_unknown():
    ix = BM25Index([Document("1", "Git commit", "commit histories", "origin")])
    hit = ix.search("git commit")[0]
    explain = ix.explain("git commit", "1")
    assert sum(explain.values()) == pytest.approx(hit.score)
    assert explain["commit"] > 0
    assert ix.explain("irrelevant", "1") == {}
    with pytest.raises(KeyError):
        ix.explain("git", "missing")


def test_empty_index_and_parameter_validation():
    assert BM25Index([]).search("anything") == []
    with pytest.raises(ValueError):
        BM25Index([], k1=float("nan"))
    with pytest.raises(ValueError):
        BM25Index([], b=float("inf"))
