# System Architecture

## Overview

The system is a Retrieval-Augmented Generation (RAG) pipeline for lyrics-based song search. Given a natural-language query, it retrieves relevant song chunks from a corpus of ~175,000 lyrics segments and synthesizes a structured answer with song identification, confidence scoring, and textual explanations.

The codebase is organized into seven sequential pipeline stages, a Streamlit frontend, and an optional RabbitMQ worker for asynchronous query dispatch.

---

## Pipeline Stages

```
[Stage 1] collection/ingest_lyrics.py
     ↓ data/raw/lyrics.json
[Stage 2] collection/clean_lyrics.py
     ↓ data/raw/cleaned_lyrics.json
[Stage 3] pipeline/chunking.py
     ↓ data/processed/chunks.jsonl
[Stage 4] pipeline/embedding.py
     ↓ data/processed/faiss*.index
     ↓ data/processed/metadata*.json
     ↓ data/processed/bm25*.pkl
[Stage 5–7] pipeline/query.py (at query time)
     ↓ Structured JSON response
[Frontend] app/app.py (Streamlit)
```

---

## Stage 1: Ingestion (`collection/ingest_lyrics.py`)

Fetches lyrics and metadata from the Genius API for every song in the Billboard Hot 100 CSV.

**Input**: `charts.csv` — columns: `date`, `rank`, `song`, `artist`, `last-week`, `peak-rank`, `weeks-on-board`

**Key behaviors**:
- Deduplicates by `(title.lower(), artist.lower())` before fetching
- Extracts primary artist from multi-artist Billboard strings (e.g., "Drake Featuring Future" → "Drake") using regex on separators `(&, Featuring, Feat., X, ,)`
- Exponential backoff on rate limits (up to 120s wait), hard stop on quota exhaustion
- Resume support: loads existing `lyrics.json` on startup and skips already-fetched songs
- Incremental saves every 10 songs (survives interruption)

**Output fields per song**: `title`, `artist`, `release_date`, `album`, `lyrics`, `genius_id`

**Failure handling**: Songs that cannot be fetched after max retries are logged to `data/failed/failed_fetches.csv` with the error reason.

---

## Stage 2: Cleaning (`collection/clean_lyrics.py`)

Removes Genius scraping artifacts and validates each entry.

**Cleaning steps applied in order**:
1. Remove `"Song Title Lyrics"` header injected on first line
2. Remove contributor/translation credit lines matching `^\d*\s*Contributor.*$`
3. Remove `"Embed"` / `"EmbedShare..."` footers
4. Remove `"You might also like"` ad injections
5. Normalize section headers to `[Title Case]` format (e.g., `[verse 1]` → `[Verse 1]`)
6. Collapse 3+ consecutive blank lines to a single blank line
7. Strip leading/trailing whitespace

**Validation flags** (entries are flagged but kept unless non-English):
- `short_lyrics`: lyrics under 100 characters after cleaning
- `non_english`: detected language is not English (these are **excluded entirely**)
- `artist_mismatch`: fuzzy artist name match below 60% (token sort ratio)
- `title_mismatch`: fuzzy title match below 60%

Language detection uses `langdetect` after stripping section headers (always English) and very short lines (ad-libs).

**Output**: `data/raw/cleaned_lyrics.json` + `data/failed/validation_report.csv`

---

## Stage 3: Chunking (`pipeline/chunking.py`)

Splits each song's lyrics into semantically coherent segments using a hybrid strategy that respects song structure.

**Why chunking matters**: FAISS and BM25 operate at the chunk level. Chunks that are too large dilute the embedding signal; chunks too small lose context. Section-aware splitting preserves semantic units (verse, chorus, bridge) rather than cutting arbitrarily at character boundaries.

**Algorithm**:

```
For each song:
  1. Split on section headers ([Verse 1], [Chorus], etc.)
  2. If no headers found → split on stanza boundaries (blank lines)
  3. For each section > max_chunk_size:
       → split on stanza boundaries
       → append " (cont.)" to header for continuation chunks
  4. Merge adjacent sections smaller than min_chunk_size
       → merge headers with " + " separator
  5. Prepend section header to chunk text: "[Chorus]\n<text>"
  6. If overlap_lines > 0:
       → prepend last N lines of previous chunk as context prefix
```

**Default parameters**: `max_chunk_size=400`, `min_chunk_size=80`, `overlap_lines=0`

**Output format** (JSONL — one JSON object per line):
```json
{
  "text": "[Chorus]\nWe found love in a hopeless place...",
  "title": "We Found Love",
  "artist": "Rihanna",
  "release_date": "2011-09-22",
  "album": "Talk That Talk",
  "chunk_index": 2,
  "total_chunks": 8
}
```

**Result**: 175,792 chunks from 14,997 songs (avg ~11.7 chunks/song)

---

## Stage 4: Embedding & Indexing (`pipeline/embedding.py`)

Encodes each chunk into a dense vector and builds three parallel indices: FAISS (dense), BM25 (sparse), and a metadata lookup.

### Contextual Embedding

Before encoding, each chunk is optionally prepended with a metadata header:

```
Song: 'We Found Love' by Rihanna. Album: Talk That Talk. Released: 2011-09-22. Section: [Chorus].
[Chorus]
We found love in a hopeless place...
```

This is the default behavior (`--skip-contextual` disables it). The metadata header enriches the embedding with song identity so FAISS can match queries like "that Rihanna song about hopeless love" without relying solely on lyric content. Evaluation showed contextual embeddings improve baseline MRR by ~21% over non-contextual.

### Embedding Models

| Model | Dimensions | Type | Index suffix |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Local (sentence-transformers) | *(none)* |
| `BAAI/bge-small-en-v1.5` | 384 | Local (sentence-transformers) | `_bge` |
| `all-mpnet-base-v2` | 768 | Local (sentence-transformers) | `_mpnet` |
| `text-embedding-3-small` | 1536 | OpenAI API | `_openai` |

All embeddings are **L2-normalized** before indexing. This makes inner product equivalent to cosine similarity, which is what `IndexFlatIP` computes.

### FAISS Index

- Type: `IndexFlatIP` — exact inner product search (no approximation)
- Rationale: The corpus size (~175K vectors) is small enough that exact search is fast (<50ms) and avoids the recall penalty of approximate methods (HNSW, IVF)
- FAISS stores **only vectors, no metadata**. Positional alignment is the critical invariant: vector at position `i` in FAISS corresponds to chunk at position `i` in `metadata.json` and position `i` in the BM25 corpus

### BM25 Index

- Implementation: `BM25Okapi` from `rank_bm25`
- Tokenization: lowercase + whitespace split (no stemming)
- Indexes the **original chunk text** (not the contextualized version)
- Saved as a pickle file; loaded at query time

### Index File Naming Convention

```
faiss.index             + metadata.json     + bm25.pkl          → all-MiniLM-L6-v2 (contextual)
faiss_bge.index         + metadata_bge.json + bm25_bge.pkl      → BAAI/bge-small-en-v1.5
faiss_openai.index      + metadata_openai.json + bm25_openai.pkl → OpenAI
faiss_noctx.index       + metadata_noctx.json  + bm25_noctx.pkl → MiniLM, no context (ablation)
faiss_bge_noctx.index   + ...                                    → BGE, no context
```

---

## Stages 5–7: Query Pipeline (`pipeline/query.py`)

At query time, the `QueryPipeline` class executes three stages: retrieval, grouping, and synthesis.

### Stage 5: Retrieval

Three retrieval strategies are available, controlled by `use_hybrid` and `use_hyde`:

#### Dense Retrieval (always active)
1. Embed the query with the same model used at index time
2. Call `faiss.index.search(q_embedding, fetch_k)` → returns (scores, chunk_indices)
3. Look up metadata by index position
4. Return chunks with their FAISS inner product scores

#### Sparse Retrieval (BM25)
1. Tokenize query: lowercase + whitespace split
2. Call `bm25.get_scores(tokens)` → score for every chunk in corpus
3. Sort by score, take top fetch_k
4. Return chunks with their BM25 scores

#### Hybrid Retrieval (RRF fusion)
When `use_hybrid=True`, both FAISS and BM25 results are merged using Reciprocal Rank Fusion:

```
For each unique chunk (identified by title + artist + chunk_index):
  rrf_score = Σ  1 / (rrf_k + rank_in_list)
              for each list it appears in (FAISS and/or BM25)
```

Lower `rrf_k` increases the weight on top-ranked items; the default is 60.

#### HyDE (Hypothetical Document Embeddings)
When `use_hyde=True`, the LLM first generates a synthetic lyric passage that *would match* the query, then that passage is embedded and used as the retrieval query instead of the original text. This bridges the semantic gap between a description-style query and document-space lyrics. In practice, for lyrics search this degrades performance because queries often already contain actual lyric text.

#### Cross-Encoder Re-ranking
When `use_reranker=True`, after initial retrieval the system fetches `top_k × fetch_k_multiplier` candidates, then scores each `(query, chunk_text)` pair with a cross-encoder model. The cross-encoder reads both texts jointly (unlike bi-encoder FAISS that encodes them separately), producing more nuanced relevance scores. Results are re-sorted by cross-encoder score and truncated to `top_k`.

```
Initial retrieval: fetch top_k × fetch_k_multiplier (e.g., 20 × 5 = 100 candidates)
Cross-encoder: score each (query, chunk) pair → float
Re-sort by cross_encoder_score → keep top_k (20)
```

### Stage 6: Grouping

Raw retrieval returns individual chunks (multiple per song). The grouping step deduplicates by `(title, artist)` and assembles a song-level result:

```json
{
  "title": "Shake It Off",
  "artist": "Taylor Swift",
  "album": "1989",
  "release_date": "2014-10-27",
  "best_score": 0.872,
  "chunks": [
    {
      "text": "[Chorus]\n'Cause the players gonna play...",
      "score": 0.872,
      "chunk_index": 3,
      "bm25_score": 12.4,
      "cross_encoder_score": 0.941
    }
  ]
}
```

Songs are sorted by `best_score` (maximum across all their chunks).

### Stage 7: LLM Synthesis

The grouped results are formatted into a context block and sent to the LLM with a system prompt instructing it to identify the song, explain the match, and express confidence.

**Anthropic (Claude Haiku)**:
- Uses the tool_use API with `tool_choice={"type": "tool", "name": "format_response"}`
- Forces structured JSON output matching the `LLMResponse` schema
- Streaming: `messages.stream()` with `content_block_delta` events yielding JSON tokens

**Ollama (Llama 3)**:
- Uses `/api/chat` REST endpoint with `format` parameter (JSON schema constraint)
- Streaming via line-delimited JSON (`stream=True`)

**LLMResponse schema**:
```python
class SongMatch(BaseModel):
    title: str
    artist: str
    album: str | None
    release_date: str | None
    relevant_lyric: str          # excerpt from retrieved chunk
    explanation: str             # why this song matches the query

class LLMResponse(BaseModel):
    matches: list[SongMatch]
    confidence: Literal["high", "medium", "low"]
    summary: str                 # 1-2 sentence answer
```

---

## Frontend (`app/app.py`)

The Streamlit app provides an interactive interface over the `QueryPipeline`. It supports two query execution paths:

### Direct Streaming Path (local dev)

```
User submits query
    → pipeline.query_stream(query, **settings)
    → First yield: {"retrieval_results": [...], "hyde_hypothesis": ...}
       (app displays retrieval immediately, no wait for LLM)
    → Subsequent yields: JSON token strings from LLM
    → App accumulates tokens, parses into LLMResponse at end
    → Displays answer + retrieved songs
```

### RabbitMQ Path (Docker deployment)

```
User submits query
    → app._query_via_rabbitmq(query, **settings)
    → Publishes to "rag.queries" queue with UUID correlation_id
    → Spins blocking event loop (120s timeout)
    → Worker processes query, publishes result to reply queue
    → App reads result, reconstructs LLMResponse
    → Displays answer + retrieved songs
```

The path is selected by whether `RABBITMQ_URL` environment variable is set. This means local dev always uses streaming without any configuration change.

### Model Detection

The app dynamically detects available FAISS indices in `data/processed/` and populates the embedding model selector. If `faiss_openai.index` exists, `text-embedding-3-small` is selected by default.

---

## Worker (`worker/worker.py`)

A standalone RabbitMQ consumer that wraps `QueryPipeline`. It processes one query at a time (`prefetch_count=1`) and pipelines the full RAG flow.

**Pipeline caching**: `QueryPipeline` initialization is expensive (loads FAISS index, sentence-transformer model). The worker caches instances by `(embedding_model, reranker_model)` so the cost is paid only once per model combination, not per query.

**Connection resilience**: On startup, the worker retries the RabbitMQ connection up to 15 times with 3-second delays. This handles the race condition where the worker container starts before RabbitMQ is healthy.

---

## Docker Architecture

```
┌──────────────────────────────────────────────────────┐
│ docker-compose                                        │
│                                                       │
│  ┌────────────┐    AMQP     ┌──────────────────────┐ │
│  │  rabbitmq  │◄────────────│      worker          │ │
│  │  :5672     │             │  worker/worker.py    │ │
│  │  :15672 UI │◄────────────│  QueryPipeline cache │ │
│  └────────────┘    AMQP     └──────────────────────┘ │
│         ▲                                             │
│         │ AMQP                                        │
│  ┌──────┴─────┐    HTTP     ┌──────────────────────┐ │
│  │    app     │◄────────────│       User           │ │
│  │  :8501     │             └──────────────────────┘ │
│  └────────────┘                                       │
│                                                       │
│  Volumes:                                             │
│    hf-cache         → HuggingFace model cache        │
│    data/processed/  → FAISS index, BM25, metadata    │
│    ollama-models/   → Llama 3 model (optional)       │
└──────────────────────────────────────────────────────┘
```

### Service startup order

1. `rabbitmq` starts first; healthcheck waits until `rabbitmq-diagnostics ping` succeeds
2. `worker` starts after `rabbitmq` is healthy; retries connection internally
3. `app` starts after `rabbitmq` is healthy and `worker` has started

### Image reuse

Both `app` and `worker` are built from the same `Dockerfile`. Docker Compose overrides the entrypoint for the worker:

```yaml
worker:
  build: .
  entrypoint: ["python", "worker/worker.py"]   # overrides Streamlit entrypoint
```

---

## Critical Data Invariant

The most important structural property of the system:

> **Position `i` in FAISS = Position `i` in `metadata.json` = Position `i` in the BM25 corpus.**

This invariant is established once at embedding time and must never be broken. Breaking it (e.g., by rebuilding only one index without the others, or filtering/reordering chunks between stages) produces silent retrieval errors where FAISS returns indices that map to wrong songs.

The build process protects this by:
1. Reading chunks in file order from `chunks.jsonl` with no filtering or shuffling
2. Saving all three indices (`faiss`, `metadata`, `bm25`) in the same script run from the same chunk list
3. Naming variant indices with consistent suffixes (`_bge`, `_openai`, `_noctx`) so the app always loads matched triplets

---

## Default Configuration (after evaluation)

The following values were set as defaults after Phase 6 evaluation identified the optimal configuration:

| Parameter | Default | Rationale |
|---|---|---|
| `embedding_model` | `text-embedding-3-small` | Best MRR (0.836), virtually tied with MiniLM (0.835) |
| `top_k` | 20 | Re-ranker benefit scales with k; 20 gives 88% Hit Rate |
| `use_hybrid` | True | Helps at 8x fetch multiplier; negligible cost |
| `use_reranker` | True | Single largest quality gain across all configurations |
| `fetch_k_multiplier` | 5 | Best accuracy/latency tradeoff (8x is marginally better but slower) |
| `use_hyde` | False | Consistently degrades MRR by 57–86% in lyrics domain |
| `rrf_k` | 60 | Dataset default; lower values help Hybrid+RR but not enough to change default |
