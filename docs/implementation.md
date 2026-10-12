# BM25 from scratch: indexing, scoring, and provenance

Unlike libraries that hide lexical retrieval behind a search API, this package exposes the complete algorithm in src/rag_lab/bm25.py and uses the Python standard library only.

## Index construction

Each document is lowercased and segmented with a regular expression tokenizer. A Counter records term frequencies for each document. An inverted index maps every vocabulary term to a list of (document index, frequency) postings. Document frequency and average document length are calculated once; inverse document frequency and length normalization are cached.

## Query processing

1. Normalize the query into unique lexical terms.
2. Read the inverted posting lists for only those terms.
3. For each posting, calculate BM25 IDF times the saturated term-frequency component.
4. Accumulate scores by document index and sort by decreasing score, breaking ties by document ID.
5. Return a document and a numeric score; keep source metadata unchanged.

## Scoring equation

IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).

score(q,d) = sum over unique query terms t of IDF(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1 * (1-b + b*len(d)/avg_len)).

Run BM25Index.explain(query, doc_id) to get each term's contribution. The new reference test computes the same formula with an independent whole-corpus scan and checks ranking and numerical agreement.

The postings index prevents scanning every document for most queries, but performance depends on corpus size, query term frequencies, and Python overhead. This is not a claim of a production search engine.

## Demo

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m rag_lab "What is a pull request?"
python -m rag_lab.evaluate
~~~

The optional Ollama backend is separate and performs generation using a locally running model. Neither embedding models nor a hosted API are required for the default retrieval path.
