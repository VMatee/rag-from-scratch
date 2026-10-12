# RAG from Scratch — BM25, Evidence and Optional Local Generation

[![Tests](https://github.com/VMatee/rag-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/VMatee/rag-from-scratch/actions/workflows/tests.yml)

Build the **retrieval** and **context assembly** parts of Retrieval-Augmented
Generation (RAG) with just the Python standard library. Learn tokenization,
BM25 lexical ranking, top-k evidence selection and source tracking.

An optional local Ollama HTTP backend handles generation; we do **not** claim
to implement or train an LLM here. The default mode prints retrieved evidence
only and works without Ollama, internet access, or private APIs.

## Run

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m rag_lab "How do I combine changes from another Git branch?"
python -m rag_lab "What is a pull request?" --generate --model llama3.2:1b
python -m rag_lab.evaluate
```

The `--generate` mode requires an **already-running local Ollama** instance and
an installed model. The model may hallucinate, ignore evidence, or cite sources
incorrectly; always inspect its answer alongside the retrieved passages.

## Process

```text
JSONL documents -- tokenize and index --> BM25 corpus
                                             |
Question ------------ tokenize -------------+--> scores --> top-k contexts
                                                        |             \
                                            evidence-only display   Ollama (optional)
```

Key files: `src/rag_lab/bm25.py`, `src/rag_lab/pipeline.py`,
`src/rag_lab/ollama.py`, `src/rag_lab/evaluate.py`, `tests/`.

## Scope and limitations

- Uses lexical retrieval (BM25), **not embeddings** or a vector database.
- Documents are short, already segmented, English sample facts written for this project.
- BM25 does not understand synonyms and will miss some paraphrases.
- Local generation sends selected documents to your Ollama process. Only use
  documents you are permitted to process.
- Document text is untrusted: never let retrieved text dictate system actions.
- Tests check ranking invariants, citations, empty queries, and evidence formatting.

**Reading:** Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond*;
Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.

## Interactive notebook

Open [the step-by-step notebook](notebooks/01_walkthrough.ipynb) in Jupyter after installing this repository with `python -m pip install -e ".[dev]"`.

## Architecture walk-through

See [the implementation map](docs/architecture.md) for a component-by-component overview.

## Related educational reading

Sebastian Raschka maintains a useful [independent educational reference](https://github.com/rasbt/RAGs). This repository uses its own small code examples and does not reproduce his book content.
