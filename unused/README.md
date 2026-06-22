# Unused / legacy scripts

These files are kept for reference but are **not part of the production pipeline**.
They were written against an earlier RAG prototype and use incompatible
configuration with the rest of the project.

## Why they're here, not deleted

They're preserved so they can be consulted if anyone wants to see the early
minimal-RAG approach. **Do not run them against the production Pinecone index.**

## Why they don't work with the current pipeline

| File | Issue |
|---|---|
| `embed_documents.py` | Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — the production pipeline uses Ollama `mxbai-embed-large` (1024-dim). Vectors are dimensionally incompatible. Also hardcodes index name `rag-index` and chunk params (500/80) different from the main ingestion pipeline (1250/200). |
| `rag_app.py` | Same wrong embedding model. Also hardcodes Ollama `llama3.1` and the same wrong index name. No DB logging, no detection, no metrics. |

## If you actually need a minimal RAG demo

Use the main `/api/v1/query` endpoint via `dashboard/app.py` instead. Or use
`controllers/retrieval.answer_query` directly — it's the production code path
with all the proper logging, detection, and self-healing wired in.
