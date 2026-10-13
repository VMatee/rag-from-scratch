# The mathematics of BM25 retrieval and evidence assembly

*From a text query to a ranked, auditable set of documents—without an external search framework.*

Retrieval-Augmented Generation begins with a practical problem: a text generator is more useful when it can consult relevant source material. But before asking any model to write, we must decide **which documents are relevant**. This repository implements that retrieval step directly using BM25, a classical lexical-ranking method, and exposes both the score of every retrieved document and the words that contributed to that score.

The default pipeline stops at retrieved evidence. Optional text generation uses a separate local Ollama process; it is not part of the BM25 implementation.

## 1. Represent each document as countable terms

A document contains an ID, title, text and source label. The tokenizer in [bm25.py](../src/rag_lab/bm25.py) lowercases text and extracts sequences of ASCII letters and digits using a regular expression. Thus, the phrase **Git branch and git commit** is converted into terms such as **git**, **branch**, **and**, **git**, **commit**.

Let $D=\{d_1,\ldots,d_N\}$ be the collection of indexed documents. Define the term frequency of token $t$ in document $d$ as

```math
\mathrm{tf}(t,d)
=\sum_{w\in d}\mathbf{1}[w=t],
```

where $\mathbf{1}[\cdot]$ is 1 when the condition holds and 0 otherwise. The document frequency is different: it counts documents that contain the term at least once,

```math
\mathrm{df}(t)
=\sum_{d\in D}\mathbf{1}[\mathrm{tf}(t,d)>0].
```

The code uses Python Counters to build these values. Both document titles and document bodies contribute to the indexed text; the source label is retained for provenance but is not scored as text.

### Why term frequency and document frequency differ

If **git** occurs five times in one document, then $\mathrm{tf}(\text{git},d)=5$. But that document contributes only **one** to $\mathrm{df}(\text{git})$. A term that appears in nearly every document provides less evidence that a particular document is distinctive than a term that appears rarely.

## 2. Derive the inverse document frequency

BM25 uses a smoothed inverse document frequency:

```math
\boxed{
\mathrm{IDF}(t)
=\ln\left(1+\frac{N-\mathrm{df}(t)+0.5}
{\mathrm{df}(t)+0.5}\right).
}
```

The added 1 keeps the value positive, including for very common terms. The implementation follows this exact equation with **math.log1p**.

Consider a tiny collection of $N=3$ documents. If only one document contains a search term, its document frequency is 1 and

```math
\mathrm{IDF}(t)
=\ln\left(1+\frac{3-1+0.5}{1+0.5}\right)
=\ln\left(\frac{8}{3}\right)
\approx0.98083.
```

If all three documents contain the term, the IDF is lower. That is intentional: the term provides less power to distinguish documents.

## 3. Account for repetition without letting long documents dominate

Counting matches alone would favor long documents, because they have more opportunities to repeat a query term. BM25 introduces a saturating term-frequency factor and length normalization.

Let $|d|$ be the number of indexed terms in document $d$, and let $\overline{L}$ be the collection's average document length. Define

```math
K(d)
=k_1\left(1-b+b\frac{|d|}{\overline{L}}\right),
```

where $k_1>0$ sets term-frequency saturation and $b\in[0,1]$ sets the strength of length normalization. The repository defaults to $k_1=1.5$ and $b=0.75$.

For a term $t$, its contribution to a document's relevance is

```math
s(t,d)=\mathrm{IDF}(t)
\frac{\mathrm{tf}(t,d)(k_1+1)}
{\mathrm{tf}(t,d)+K(d)}.
```

When a word repeats, the numerator grows, but the denominator grows too. The score therefore increases more slowly than a raw term count.

### One fully worked BM25 calculation

Suppose a document is 100 tokens long, the average length is also 100, a query term occurs twice, and that term has IDF 0.98083. With $k_1=1.5$ and $b=0.75$:

```math
K(d)=1.5\left(1-0.75+0.75\frac{100}{100}\right)=1.5.
```

The term contributes

```math
s(t,d)=0.98083
\frac{2(1.5+1)}{2+1.5}
=0.98083\frac{5}{3.5}
\approx\boxed{1.40118}.
```

This is a numerical illustration of the formula, not a score measured from the repository's small example dataset.

## 4. Combine contributions into a document score

For a query $q$, the total BM25 score is the sum of the contributions of its *distinct* normalized terms:

```math
\boxed{
\mathrm{BM25}(q,d)
=\sum_{t\in\mathrm{unique}(q)}s(t,d).
}
```

The word **unique** matters here. This repository converts the query tokens into a set, so writing the same query term twice does not double its contribution. That is an implementation decision, not a universal property of every BM25 variant.

A document receives a hit only when at least one query term occurs in it and the score is positive. Hits are ordered from highest to lowest score. Exact score ties are resolved by document ID for deterministic results.

The **BM25Index.explain** method exposes the individual terms' scores, making it possible to answer a useful engineering question: *Did this document rank highly for the concept I intended, or just because an unrelated word happened to match?*

## 5. Build an inverted index rather than scan every document

Relevance scoring would be wasteful if the code searched every document for every query. Instead, [BM25Index](../src/rag_lab/bm25.py) creates an **inverted index**.

For each vocabulary term $t$, it stores a postings list

```math
\mathcal{P}(t)
=\{(i,\mathrm{tf}(t,d_i)):\mathrm{tf}(t,d_i)>0\}.
```

For a query with terms $t_1,\ldots,t_m$, the code reads only the corresponding lists $\mathcal{P}(t_1),\ldots,\mathcal{P}(t_m)$. It accumulates scores by document index and sorts the matched candidates.

The number of term-document pairs inspected for scoring is related to

```math
\sum_{t\in\mathrm{unique}(q)}|\mathcal{P}(t)|
```

rather than necessarily $N$ full document scans. However, search time still depends on posting-list sizes, sorting, Python overhead and the query distribution. There are no published latency benchmarks in this repository.

A separate [test implementation](../tests/test_inverted_index.py) calculates BM25 by explicitly scanning a synthetic corpus and checks that the inverted-index search produces numerically matching scores and the same ranking.

## 6. Turn ranked hits into attributed evidence

A retrieved item is not only a numeric score. It also carries the original document ID, title, body and source. The [pipeline](../src/rag_lab/pipeline.py) formats each hit into an evidence block tagged by its document ID, such as **[DOC:git-commit]**.

Conceptually, the evidence passed downstream is

```math
\mathcal{C}(q)
=\mathrm{concat}\bigl(\mathrm{top}\text{-}k\ \text{documents ranked by BM25}(q,\cdot)\bigr).
```

This produces *context*, not a generated answer. The default command therefore prints retrieved evidence and makes no claim that a language model has summarized it.

When the optional **--generate** mode is enabled, [ollama.py](../src/rag_lab/ollama.py) sends the question and evidence to a local Ollama model. The request asks the generator to cite document IDs and report insufficient evidence, but this is a prompt instruction—not a guarantee of factual or citation accuracy. The retrieved content must still be treated as untrusted input.

## 7. Evaluate the retriever separately from the generator

The bundled [evaluation](../src/rag_lab/evaluate.py) contains a few handcrafted query/document-ID pairs. For $Q$ evaluation queries, hit-at-$k$ is

```math
\mathrm{Hit@}k
=\frac{1}{Q}
\sum_{i=1}^{Q}
\mathbf{1}\left[d_i^\star\in\mathrm{Top}k(q_i)\right],
```

where $d_i^\star$ is the expected relevant document for query $q_i$. A score of 1.0 means every expected ID appeared in the top-$k$ for *that small set*—not that the retriever will handle unknown documents or real-world paraphrases.

The system currently uses **lexical matching**, not semantic embeddings. Queries containing a synonym absent from the documents can fail. For serious evaluation, use a larger independently labeled corpus and report retrieval metrics, query categories, latency and error analysis.

### Reproduce the pipeline

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m rag_lab "What is a pull request?"
python -m rag_lab.evaluate
~~~

The optional local-generation command requires a separate Ollama installation and an available local model:

~~~bash
python -m rag_lab "What is a pull request?" --generate --model llama3.2:1b
~~~

**Source crosswalk:** [tokenizer, postings and BM25](../src/rag_lab/bm25.py) · [evidence assembly](../src/rag_lab/pipeline.py) · [optional local generation](../src/rag_lab/ollama.py) · [score parity tests](../tests/test_inverted_index.py).

**Interpretation.** This implementation demonstrates the retrieval mathematics, search data structures and provenance chain of a small RAG pipeline. It does not train a language model, and neither its toy evaluation nor its optional generation command establishes production retrieval accuracy.
