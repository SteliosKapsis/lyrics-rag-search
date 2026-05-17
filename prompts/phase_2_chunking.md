# Phase 2 — Chunking

You are an expert in NLP and retrieval-augmented generation (RAG). I'm building a lyrics RAG pipeline for my thesis.

I have a JSON file where each entry contains: title, artist, release_date, album, lyrics (as a single string).

Help me write a Python script to chunk the lyrics for embedding. Think through this step-by-step:

1. What chunking strategy makes most sense for lyrics specifically? (consider verse/chorus structure vs. fixed token windows — explain your reasoning)
2. Implement your recommended strategy
3. Each chunk should carry metadata: title, artist, release_date, album, chunk_index, total_chunks
4. Output a list of chunk objects saved to a JSONL file (one chunk per line)

Aim for chunks that preserve semantic meaning — avoid splitting mid-verse where possible. If you make any assumptions about typical lyric structure, state them clearly.
