# Phase 3 — Embedding & Indexing

## Overview

Embeds lyric chunks into dense vectors using sentence-transformers and stores them in a local FAISS index for fast similarity search. Also builds a BM25 keyword index for hybrid retrieval. This is the core retrieval infrastructure of the RAG pipeline.

## Script

`pipeline/embedding.py`

## Usage

```bash
# From project root, with defaults (all-MiniLM-L6-v2):
.venv\Scripts\python pipeline\embedding.py

# With a different model:
.venv\Scripts\python pipeline\embedding.py --model all-mpnet-base-v2

# Adjust batch size for memory constraints:
.venv\Scripts\python pipeline\embedding.py --batch-size 32

# Build a non-contextual index (raw lyrics, no metadata header):
.venv\Scripts\python pipeline\embedding.py --skip-contextual
```

## Input

- **`data/processed/chunks.jsonl`** — output from Phase 2 (chunking)

## Output

| File | Location | Description |
|------|----------|-------------|
| `faiss.index` | `data/processed/` | FAISS binary index file containing all embedding vectors |
| `metadata.json` | `data/processed/` | JSON array of chunk metadata, positionally aligned with the FAISS index |
| `bm25.pkl` | `data/processed/` | Pickled BM25Okapi index for keyword-based sparse retrieval |
| `faiss_noctx.index` | `data/processed/` | Non-contextual FAISS index (only when `--skip-contextual` is used) |
| `metadata_noctx.json` | `data/processed/` | Non-contextual metadata (only when `--skip-contextual` is used) |
| `bm25_noctx.pkl` | `data/processed/` | Non-contextual BM25 index (only when `--skip-contextual` is used) |

### Metadata schema (each entry)

```json
{
  "text": "[Verse 1]\nThere ain't no gold in this river...",
  "title": "Easy On Me",
  "artist": "Adele",
  "release_date": "2021-10-15",
  "album": "30 (Target Exclusive)",
  "chunk_index": 0,
  "total_chunks": 6
}
```

The `text` field is included in metadata so the query pipeline (Phase 4) can display lyric excerpts without a separate lookup.

## Dependencies

- `sentence-transformers>=2.2.0` — embedding model loading and inference
- `faiss-cpu>=1.7.0` — vector similarity search index
- `rank_bm25>=0.2.2` — BM25Okapi sparse keyword retrieval

## Contextual Embeddings

Before embedding, each chunk's text is enriched with a metadata header:

```
Song: 'Easy On Me' by Adele. Album: 30 (Target Exclusive). Released: 2021-10-15. Section: [Chorus].
Go easy on me, baby / I was still a child...
```

This is based on Anthropic's **Contextual Retrieval** technique. The intuition: a verse like "she left me standing in the cold" gets a generic embedding without context, but when the embedding model also sees the song title, artist, and section type, the resulting vector is much more specific and retrievable.

Anthropic reported:
- 35% reduction in retrieval failure with contextual embeddings alone
- 67% reduction when combined with BM25 + reranking (both already in our pipeline)

**Implementation detail:** metadata stores two text fields:
- `text` — original chunk text (used for display in the UI and as LLM context)
- `text_for_embedding` — contextualized text (used only at embedding time)

The BM25 index uses the **original** text, not the contextualized text, because BM25 should match on actual lyrics keywords, not the metadata prefix.

### Contextual Embedding Ablation (`--skip-contextual`)

To measure the isolated contribution of contextual embeddings, the `--skip-contextual` flag bypasses `build_contextual_text()` and embeds the raw chunk text directly. This produces a separate non-contextual index (suffixed `_noctx`) that can be evaluated alongside the contextual index in Phase 6.

When `--skip-contextual` is set:
- The metadata header is **not** prepended — the embedding model sees only the raw lyrics text
- Output files are automatically suffixed: `faiss_noctx.index`, `metadata_noctx.json`, `bm25_noctx.pkl`
- The BM25 index is rebuilt under the `_noctx` suffix so the non-contextual pipeline has its own complete, self-contained file set
- Model-specific suffixes are applied first, then `_noctx`, so combining both flags produces `faiss_bge_noctx.index` automatically (no custom output paths needed)

The MRR delta between contextual and non-contextual indices directly measures how much retrieval quality comes from the metadata enrichment versus the lyrics content itself.

**Data integrity requirement:** Both contextual and non-contextual indices must be built from the same `chunks.jsonl` file. Do not re-run `pipeline/chunking.py` or `pipeline/ingestion.py` between builds — if the input chunks change, the comparison is invalid. Verify with a checksum before and after:

```bash
certutil -hashfile data\processed\chunks.jsonl MD5
```

Same hash = same data = valid ablation. Both runs read every chunk in file order with no filtering or randomness, so the only variable is the presence of the metadata header.

## Embedding Models

| Model | Dimensions | Size | MTEB Score | Notes |
|-------|-----------|------|------------|-------|
| `all-MiniLM-L6-v2` | 384 | ~80MB | ~56 | Lightweight baseline, fast iteration |
| `BAAI/bge-small-en-v1.5` | 384 | ~50MB | ~59 | Drop-in replacement for MiniLM, better quality at same speed |
| `all-mpnet-base-v2` | 768 | ~420MB | ~58 | Higher capacity, ~2x slower, larger index |

### Recommended: `BAAI/bge-small-en-v1.5`

Same 384 dimensions and comparable speed as MiniLM, but scores ~3 points higher on the MTEB benchmark. A zero-effort quality upgrade — the FAISS index structure is identical.

### Future comparison: OpenAI `text-embedding-3-small`

- Cloud-based, costs per token
- To be included in Phase 6 evaluation as a local vs. cloud comparison data point for the thesis

The model name is configurable via `--model` so Phase 6 can sweep across models.

## FAISS Index Design

### Why `IndexFlatIP` (Inner Product)?

- Embeddings are **L2-normalized** before insertion (`normalize_embeddings=True` in sentence-transformers). On normalized vectors, inner product equals cosine similarity.
- `IndexFlatIP` performs exact (brute-force) search — no approximation, no training step required.
- For our expected dataset size (~200k chunks from ~30k songs), exact search is fast enough (<100ms per query). No need for approximate indexes (IVF, HNSW).

### FAISS Gotchas

1. **FAISS stores only vectors, not metadata.** The index contains no information about which song a vector belongs to. The `metadata.json` file provides this mapping via positional alignment: FAISS index position `i` corresponds to `metadata[i]`.

2. **Positional mapping is fragile.** If you delete, reorder, or re-embed a subset of vectors, the metadata file becomes misaligned. Always regenerate both files together by re-running the full script.

3. **No incremental updates.** To add new songs, you must re-run the entire pipeline (chunking → embedding). For our use case this is fine — the dataset is static once ingested.

4. **Index file is platform-dependent.** FAISS index files written on one architecture (e.g., x86) may not load on another (e.g., ARM). Regenerate if moving between machines.

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `all-MiniLM-L6-v2` | Sentence-transformers model name from HuggingFace |
| `--batch-size` | 64 | Number of texts to embed at once. Lower if running out of memory. |
| `--input` | `data/processed/chunks.jsonl` | Input chunks file |
| `--index-output` | `data/processed/faiss.index` | Output FAISS index path |
| `--metadata-output` | `data/processed/metadata.json` | Output metadata JSON path |
| `--bm25-output` | `data/processed/bm25.pkl` | Output BM25 index path |
| `--skip-contextual` | False | Skip metadata header enrichment; embed raw lyrics only. Outputs are auto-suffixed with `_noctx` |
| `--model` (non-default) | — | Using any model other than `all-MiniLM-L6-v2` automatically applies a model suffix (`_bge`, `_mpnet`). Combined with `--skip-contextual` produces e.g. `faiss_bge_noctx.index` |

### Automatic Output Suffix Logic

The script applies suffixes in this order so indices never collide:

| Flags used | Output files |
|------------|-------------|
| *(defaults)* | `faiss.index`, `metadata.json`, `bm25.pkl` |
| `--skip-contextual` | `faiss_noctx.index`, `metadata_noctx.json`, `bm25_noctx.pkl` |
| `--model BAAI/bge-small-en-v1.5` | `faiss_bge.index`, `metadata_bge.json`, `bm25_bge.pkl` |
| `--model BAAI/bge-small-en-v1.5 --skip-contextual` | `faiss_bge_noctx.index`, `metadata_bge_noctx.json`, `bm25_bge_noctx.pkl` |
| `--openai` | `faiss_openai.index`, `metadata_openai.json`, `bm25_openai.pkl` |

The known model→suffix mapping is defined in `MODEL_SUFFIXES` inside `main()`:
```python
MODEL_SUFFIXES = {
    "all-MiniLM-L6-v2": "",           # default — no suffix
    "BAAI/bge-small-en-v1.5": "_bge",
    "all-mpnet-base-v2": "_mpnet",
}
```
For any model not in this dict, the script derives a slug automatically from the model name (basename after `/`, non-alphanumeric characters replaced with `_`). Custom `--index-output` / `--metadata-output` / `--bm25-output` paths bypass all auto-suffixing.

## Test Results (5-song sample, 37 chunks)

```
Model: all-MiniLM-L6-v2
Embedding time: ~2 seconds
Dimensions: 384
Index size: 37 vectors

Sanity check query: "sad song about letting someone go"
Top 3 results:
  1. [0.456] Easy On Me by Adele — [Chorus] Go easy on me, baby...
  2. [0.455] Easy On Me by Adele — [Chorus] Go easy on me, baby...
  3. [0.450] Easy On Me by Adele — [Chorus] Go easy on me, baby...
```

All top results correctly identify Adele's "Easy On Me" — the most semantically relevant song in the 5-song test set for that query. Multiple chorus chunks match because the same chorus repeats in the song (expected behavior — Phase 4's query pipeline should deduplicate by song).

## Design Decisions

- **Cosine similarity via normalized inner product** — standard for semantic search. Normalizing at embed time means we only pay the cost once, not at every query.
- **Flat index over approximate** — exact search is acceptable for our scale. Switching to `IndexIVFFlat` would only matter at millions of vectors.
- **Metadata includes full text** — eliminates a join between FAISS results and a separate text store. Minor memory cost but simplifies the query pipeline.
- **Model downloaded from HuggingFace on first run** — cached locally in `~/.cache/huggingface/` after first download. Subsequent runs are offline-capable.

## BM25 Index

### Purpose

The BM25 (Best Matching 25) index provides keyword-based sparse retrieval. While FAISS captures semantic meaning, BM25 excels at finding exact keyword matches. This is critical for direct lyric recall queries (e.g., "never gonna give you up").

### Tokenization

Simple lowercase + whitespace split. Lyrics don't benefit from stemming or lemmatization because users often search for exact words and phrases as they appear in the song.

### Alignment

The BM25 index is positionally aligned with FAISS and metadata — document `i` in BM25 corresponds to `metadata[i]` and FAISS vector `i`. All three must be regenerated together.
