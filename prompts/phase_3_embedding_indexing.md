# Phase 3 — Embedding & Indexing

You are an expert in vector databases and semantic search. I'm building a lyrics RAG pipeline for my thesis.

I have a JSONL file of lyric chunks, each with fields: text, title, artist, release_date, album, chunk_index, total_chunks.

Help me write a Python script to embed and index these chunks. Specifically:

1. Use sentence-transformers with one of these models (recommend which
   and explain why):
   - all-MiniLM-L6-v2 (384 dims, ~80MB, MTEB ~56 — lightweight baseline)
   - BAAI/bge-small-en-v1.5 (384 dims, ~50MB, MTEB ~59 — same speed,
     better quality, drop-in replacement for MiniLM)
   - all-mpnet-base-v2 (768 dims, ~420MB, MTEB ~58 — higher capacity
     but slower)
   The script must accept --model as a CLI flag so Phase 6 can sweep
   across all three models.

2. Before embedding, apply contextual enrichment to each chunk:
   prepend a short metadata header to the chunk text so the embedding
   captures which song, artist, album, and section it belongs to.
   Format:
       Song: '{title}' by {artist}. Album: {album}. Released: {release_date}. Section: {section_header}.
       {original chunk text}
   where section_header is extracted from the first line of the chunk
   if it matches [Section Name], otherwise "Unknown".
   This is based on Anthropic's "Contextual Retrieval" technique, which
   reported a 35% reduction in retrieval failure rate with contextual
   embeddings alone, and 67% when combined with BM25 + reranking.
   Store BOTH the contextualized text (for embedding) and the original
   text (for display) in metadata. The metadata schema should have:
   - "text": original chunk text (for display in UI and LLM context)
   - "text_for_embedding": contextualized text (only used at embed time)

3. Embed all chunks in batches to avoid memory issues
4. Store embeddings + metadata in a local FAISS index (I want to keep this free and local for now)
5. Save the FAISS index and a corresponding metadata lookup (JSON) to disk so it can be reloaded without re-embedding

## BM25 Keyword Index

In addition to the dense FAISS index, build a BM25 sparse keyword index
over the same chunks. This will be used alongside FAISS for hybrid
retrieval (dense + sparse) in Phase 4.

6. Use the `rank_bm25` library (BM25Okapi) to build a keyword index over
   chunk texts (use the ORIGINAL text, not the contextualized text —
   BM25 should match on actual lyrics keywords, not the metadata prefix)
7. Tokenize chunks by lowercasing and splitting on whitespace (keep it
   simple — lyrics don't need heavy NLP tokenization)
8. Save the BM25 index to disk using pickle so it can be reloaded without
   rebuilding. Save it alongside the FAISS outputs:
   - `data/processed/bm25.pkl` — serialized BM25Okapi object
9. The BM25 index must be positionally aligned with the FAISS index and
   metadata JSON (same chunk order), so that BM25 result indices map
   directly to metadata entries

If there are any gotchas with FAISS and metadata storage I should know about, flag them.

## Contextual Embedding Ablation

10. Add a `--skip-contextual` CLI flag (default False). When set, bypass
    `build_contextual_text()` and embed the raw chunk text directly
    (without the metadata header). This produces a non-contextual index
    for ablation comparison — measuring how much retrieval quality comes
    from the metadata enrichment vs. the lyrics content alone.

11. When `--skip-contextual` is set, suffix all output files to coexist
    with the contextual index:
    - `data/processed/faiss_noctx.index`
    - `data/processed/metadata_noctx.json`
    - `data/processed/bm25_noctx.pkl`
    (Plus the same `_noctx` suffix combined with model suffixes, e.g.,
    `faiss_bge_noctx.index` for the bge model.)

12. The BM25 index is unaffected by contextual headers (it already uses
    original text), but rebuild it anyway under the `_noctx` suffix so
    the non-contextual pipeline has its own complete, self-contained set
    of files.
