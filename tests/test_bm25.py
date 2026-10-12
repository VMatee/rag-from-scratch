import pytest
from rag_lab import BM25Index, Document, evidence_answer, format_context
from rag_lab.evaluate import run


def fixture_index():
    return BM25Index([
        Document("a", "Linux permissions", "chmod changes permissions", "demo://a"),
        Document("b", "Git commit", "commit records version history", "demo://b"),
        Document("c", "Git branch", "merge integrates branch", "demo://c"),
    ])


def test_returns_relevant_document_and_source():
    hits = fixture_index().search("chmod permissions", top_k=2)
    assert hits[0].doc.id == "a" and hits[0].score > 0
    assert "[DOC:a]" in format_context(hits)
    assert "demo://a" in evidence_answer(hits)


def test_no_match_is_not_invented():
    assert fixture_index().search("elephant zebras") == []
    assert "cannot find grounded" in evidence_answer([])


def test_validates_input():
    with pytest.raises(ValueError):
        fixture_index().search("git", 0)
    with pytest.raises(ValueError):
        BM25Index([Document("x", "", "", ""), Document("x", "", "", "")])


def test_tied_scores_are_stably_sorted():
    ix = BM25Index([Document("z", "a", "cat", "src"), Document("a", "a", "cat", "src")])
    assert [h.doc.id for h in ix.search("cat")] == ["a", "z"]


def test_illustrative_retrieval_eval():
    assert 0 <= run() <= 1
