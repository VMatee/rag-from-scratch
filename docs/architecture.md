# Architecture note — Retrieval augmented answer construction

```text
local documents -> tokens -> BM25; question -> BM25 -> ranked evidence -> optional local generator
```

## Fidelity and assumptions

Retrieval is implemented from scratch; generation delegates to optional Ollama.

Read the source implementation and the tests together. New features should
add controlled experiments or unit tests before using benchmark claims.
