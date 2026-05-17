# Phase 4 — RAG Query Pipeline

## Overview

The core query pipeline that ties together retrieval and generation. Supports standard, hybrid (FAISS + BM25), and HyDE retrieval, optional cross-encoder re-ranking, streaming LLM responses, and two LLM backends (Anthropic Claude and local Ollama/Llama 3).

## Scripts

- `pipeline/query.py` — main query pipeline (QueryPipeline class + CLI)
- `pipeline/reranker.py` — cross-encoder reranker module

## Usage

### CLI

```bash
# Standard retrieval + Claude synthesis:
.venv\Scripts\python pipeline\query.py "that sad song about leaving home"

# With HyDE retrieval:
.venv\Scripts\python pipeline\query.py "upbeat dance song" --use-hyde

# With cross-encoder re-ranking:
.venv\Scripts\python pipeline\query.py "love song" --use-reranker

# Both HyDE and re-ranking:
.venv\Scripts\python pipeline\query.py "rock anthem" --use-hyde --use-reranker

# Using local Ollama/Llama 3:
.venv\Scripts\python pipeline\query.py "country song" --llm-backend ollama

# With a different re-ranker model:
.venv\Scripts\python pipeline\query.py "love song" --use-reranker --reranker-model BAAI/bge-reranker-base

# Custom embedding model (must match index-time model):
.venv\Scripts\python pipeline\query.py "ballad" --embedding-model all-mpnet-base-v2
```

### As a module (for Phase 5 Streamlit app and Phase 6 evaluation)

```python
from pipeline.query import QueryPipeline

pipeline = QueryPipeline(llm_backend="anthropic", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
result = pipeline.query(
    "sad song about letting someone go",
    top_k=5,
    use_hyde=True,
    use_hybrid=True,
    use_reranker=True,
    rrf_k=60,                # RRF fusion constant (default 60)
    fetch_k_multiplier=3,    # re-ranker candidate pool (default 3x)
)

# Streaming usage (for Streamlit):
stream = pipeline.query_stream("sad song about rain", top_k=5)
retrieval_data = next(stream)  # dict with retrieval_results + hyde_hypothesis
for token in stream:           # str tokens from LLM
    print(token, end="")

# result["llm_response"]        → LLM's synthesized answer
# result["retrieval_results"]    → grouped chunk results with scores
# result["query"]                → the original query string
# result["hyde_hypothesis"]      → generated hypothetical lyric (or None)
```

## Input

- **`data/processed/faiss.index`** — FAISS index from Phase 3
- **`data/processed/metadata.json`** — chunk metadata from Phase 3
- **`data/processed/bm25.pkl`** — BM25 keyword index from Phase 3 (optional; hybrid degrades gracefully if missing)
- **`.env`** — must contain `ANTHROPIC_API_KEY` (for Anthropic backend)

## Output

Returns a dict:

```python
{
    "query": "sad song about leaving home",
    "retrieval_results": [
        {
            "title": "Easy On Me",
            "artist": "Adele",
            "album": "30",
            "release_date": "2021-10-15",
            "best_score": 0.85,
            "chunks": [
                {
                    "text": "[Chorus]\nGo easy on me...",
                    "score": 0.85,                  # FAISS cosine similarity
                    "bm25_score": 12.3,              # present only if hybrid used
                    "rrf_score": 0.031,              # present only if hybrid used
                    "cross_encoder_score": 7.42,     # present only if reranker used
                    "chunk_index": 5,
                },
            ]
        }
    ],
    "llm_response": {                              # LLMResponse Pydantic object
        "matches": [
            {
                "title": "Easy On Me",
                "artist": "Adele",
                "album": "30",
                "release_date": "2021-10-15",
                "relevant_lyric": "Go easy on me, baby...",
                "explanation": "This matches the query about letting someone go."
            }
        ],
        "confidence": "high",
        "summary": "Based on the retrieved lyrics..."
    },
    "hyde_hypothesis": "Raindrops on the window pane...",  # or None
}
```

## Dependencies

- `anthropic>=0.30.0` — Claude API client
- `sentence-transformers>=2.2.0` — query embedding + cross-encoder reranker
- `faiss-cpu>=1.7.0` — vector similarity search
- `python-dotenv>=1.0.0` — loads API keys from `.env`
- `requests>=2.27.0` — Ollama REST API client
- `pydantic>=2.0.0` — structured output schema definition and validation

## Architecture

```
Query → [HyDE?] → [Embed] → [Hybrid Retrieve (FAISS+BM25)?] → [Rerank?] → [Group by Song] → [LLM Synthesize/Stream] → Response
```

### Step 1: HyDE (optional)

**HyDE (Hypothetical Document Embeddings)** generates a hypothetical lyric before retrieval.

**Intuition:** The user's query ("sad song about rain") lives in "question space", but the indexed chunks live in "document space" (actual lyrics). These spaces are linguistically different. By asking the LLM to generate what the answer might look like (a hypothetical lyric like "raindrops fall on empty streets / tears I cannot hide"), we get an embedding that's closer to how real lyrics are written, producing better similarity matches.

**Why this helps for lyrics:** Song lyrics use figurative language, repetition, rhyme, and slang that are very different from how users describe songs in natural language. HyDE bridges this gap.

The hypothesis is generated using the same LLM backend (Anthropic or Ollama) with a dedicated prompt that asks for 4-8 lines of plausible (not real) lyrics.

### Step 2: Retrieve (Standard or Hybrid)

When **hybrid mode** is enabled (default), retrieval combines two methods:

1. **FAISS dense retrieval** — embeds the query and finds semantically similar chunks. Good for descriptive queries like "sad song about rain".
2. **BM25 sparse retrieval** — keyword-based matching using term frequency. Good for direct lyric recall like "never gonna give you up".

Results from both methods are merged using **Reciprocal Rank Fusion (RRF)**:

```
score(d) = Σ 1 / (k + rank_i(d))    where k = 60
```

RRF is a simple, parameter-light fusion technique that doesn't require score normalization. The constant k=60 is the standard default from the original paper, but is now configurable via the `rrf_k` parameter (lower values weight the top-ranked method more heavily).

Both methods fetch `fetch_k` candidates before fusing, then the top_k fused results are kept. When re-ranking is enabled, `fetch_k = top_k × fetch_k_multiplier` (default 3x) to give the re-ranker a larger candidate pool. The `fetch_k_multiplier` parameter is configurable — Phase 6 evaluation showed that 8x yields significantly better results than the default 3x.

When hybrid is disabled, only FAISS dense retrieval is used (original behavior).

### Step 3: Re-rank (optional)

The **cross-encoder reranker** (`pipeline/reranker.py`) re-scores each (query, chunk) pair using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Why re-rank?** Bi-encoder retrieval (FAISS) embeds query and chunk independently — fast but approximate. A cross-encoder processes both together, capturing fine-grained interactions. It's too slow for full-index search but perfect for re-scoring top-k results.

**Important:** Re-ranking always uses the original query, even when HyDE is enabled. The HyDE hypothesis is good for retrieval (finding candidates) but the original query is what the user actually wants.

### Step 4: Group by Song

Chunks are grouped by (title, artist) to deduplicate repeated choruses. Songs are ranked by their best score (cross-encoder if available, else FAISS).

### Step 5: Synthesize (Structured, Standard or Streaming)

The grouped results are sent to the LLM with the system prompt and user query. Both backends return **structured JSON** conforming to a Pydantic schema, not free text.

#### Structured Output Schema

```python
class SongMatch(BaseModel):
    title: str
    artist: str
    album: str | None
    release_date: str | None
    relevant_lyric: str        # most relevant excerpt
    explanation: str            # why this matches the query

class LLMResponse(BaseModel):
    matches: list[SongMatch]    # ranked by relevance
    confidence: str             # "high", "medium", or "low"
    summary: str                # 1-2 sentence answer
```

**Why structured output?** Free-text LLM responses require fragile string parsing to extract song matches, confidence levels, and excerpts. Structured output guarantees the response shape, enables programmatic confidence scoring, and makes the Streamlit display cleaner.

**Anthropic implementation:** Uses tool_use — defines a tool whose `input_schema` matches the Pydantic model. Claude returns structured JSON as the tool call arguments.

**Ollama implementation:** Uses the `format` parameter with the Pydantic model's JSON schema (`LLMResponse.model_json_schema()`), which forces Ollama to return conforming JSON.

#### Modes

**Standard mode** (`query()`) waits for the full LLM response, parses it into the `LLMResponse` Pydantic model — used by the evaluation notebook.

**Streaming mode** (`query_stream()`) is a generator that:
1. First yields a dict with `retrieval_results` and `hyde_hypothesis` so the UI can display song results immediately
2. Then yields individual string tokens from the LLM as they arrive (raw JSON text)
3. The caller collects all tokens and parses the final JSON after stream completes

Streaming uses `client.messages.stream()` for Anthropic and `stream: true` for Ollama.

## LLM Backends

### Anthropic (Claude Haiku)

- Default backend, uses `claude-haiku-4-5-20251001`
- Requires `ANTHROPIC_API_KEY` in `.env`
- Fast, cost-efficient for the synthesis task

### Ollama (Llama 3 local)

- Runs entirely locally — no API key, no cost, no data leaves the machine
- Requires Ollama installed: https://ollama.com/download
- Pull the model: `ollama pull llama3`
- Ollama runs automatically after install, serving on `localhost:11434`
- Uses the `/api/chat` endpoint with `stream: false`

Both backends receive the same system prompt and produce the same output structure. The main tradeoff is quality vs. privacy/cost — Claude Haiku tends to produce more polished responses, while Llama 3 is free and local.

## LLM Prompt Design

### System prompt

Sets the LLM's role as a "lyrics search assistant" with explicit instructions to:
- Identify matching songs from the retrieved excerpts
- Show relevant lyric excerpts
- Provide metadata (title, artist, album, release date)
- Rank multiple matches by relevance
- Acknowledge uncertainty when scores are low (<0.3)
- Never fabricate lyrics or metadata not in the context

### HyDE prompt

Separate prompt that asks the LLM to generate 4-8 lines of hypothetical lyrics matching the user's description. Explicitly instructs not to use real lyrics from existing songs.

### Prompt design tradeoffs

1. **Scores included in context** — the LLM sees similarity scores and can calibrate confidence.
2. **Grouped by song** — reduces redundancy and gives clearer structure.
3. **Zero-shot** — no few-shot examples to avoid biasing response format.
4. **Same prompt for both backends** — ensures comparable behavior for Phase 6 evaluation.

## Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 5 | Number of song results to return |
| `--use-hyde` | False | Enable HyDE retrieval |
| `--no-hybrid` | False | Disable hybrid retrieval (use FAISS only) |
| `--use-reranker` | False | Enable cross-encoder re-ranking |
| `--reranker-model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for re-ranking |
| `--llm-backend` | `anthropic` | LLM backend: `anthropic` or `ollama` |
| `--llm-model` | auto | `claude-haiku-4-5-20251001` (anthropic) or `llama3` (ollama) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Must match index-time model |

### Programmatic-only parameters (used via `query()` / `query_stream()`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rrf_k` | 60 | RRF fusion constant. Lower values weight the top-ranked method more heavily. |
| `fetch_k_multiplier` | 3 | Candidate pool multiplier for re-ranking (`fetch_k = top_k × multiplier`). Higher values give the cross-encoder more candidates to re-score. |

These parameters are not exposed as CLI arguments but are available when using `QueryPipeline` as a module (e.g., from the evaluation notebook or Streamlit app).

## Reranker Details (`pipeline/reranker.py`)

### Supported Models

| Model | Params | Size | Notes |
|-------|--------|------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | ~80MB | Default. Lightweight, fast. |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | ~120MB | Same family, double depth. |
| `BAAI/bge-reranker-base` | 110M | ~440MB | Different architecture, highest capacity. |

The model is configurable via the `--reranker-model` CLI flag or the `reranker_model` constructor argument on `QueryPipeline`. The reranker is loaded lazily on first use (no cost if `--use-reranker` is not set).

### Reranker class API

```python
from pipeline.reranker import Reranker

reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked_chunks = reranker.rerank(
    query="sad song about rain",
    chunks=retrieved_chunks,  # list of dicts with "text" field
    top_k=5,
)
# Each chunk now has an added "cross_encoder_score" field
```

## Design Decisions

- **Module-first design** — `QueryPipeline` class is importable by Phase 5 (Streamlit) and Phase 6 (evaluation) without duplication.
- **Lazy reranker loading** — cross-encoder model only loads when `use_reranker=True`, saving memory and startup time.
- **HyDE uses same LLM backend** — if using Ollama, HyDE also runs locally.
- **Reranker uses original query, not HyDE hypothesis** — the hypothesis is for retrieval (finding candidates); relevance judgment should be against what the user actually asked.
- **Configurable fetch multiplier for reranking** — when reranking, we fetch `top_k × fetch_k_multiplier` candidates from FAISS (default 3x), then the reranker selects the best top-k from that pool. Phase 6 evaluation showed 8x yields significantly better results than 3x.
- **Hybrid retrieval on by default** — BM25 complements FAISS well for lyrics search (keyword matches for direct recall, semantic for descriptive queries). Falls back gracefully to FAISS-only if `bm25.pkl` is missing.
- **RRF over score normalization** — Reciprocal Rank Fusion doesn't require normalizing scores across methods (BM25 scores and cosine similarities are on completely different scales).
- **Streaming via generator** — `query_stream()` yields retrieval data first, then LLM tokens. This lets the Streamlit app show results immediately while the answer streams in.
- **Graceful Ollama error handling** — clear error messages with install instructions if Ollama isn't running.
