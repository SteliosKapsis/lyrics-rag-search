# Phase 5 — Streamlit Frontend

## Overview

A lightweight Streamlit web app that wraps the `QueryPipeline` from Phase 4, providing an interactive UI for the lyrics RAG search system. Designed for thesis demonstrations and user-facing exploration of retrieval techniques.

`app.py` supports two mutually exclusive execution modes, selected at startup by whether the `RABBITMQ_URL` environment variable is set:

- **Direct mode (local/dev, `RABBITMQ_URL` unset)** — `app.py` itself loads and owns the `QueryPipeline` in-process via `@st.cache_resource`, and calls `pipeline.query_stream()` directly.
- **RabbitMQ RPC mode (Docker deployment, `RABBITMQ_URL` set)** — `app.py` never instantiates a pipeline. Instead it publishes the query to the `rag.queries` queue and blocks for an RPC reply (`_query_via_rabbitmq()`). `worker/worker.py` is the process that actually consumes the queue, instantiates and caches `QueryPipeline`, runs the query, and publishes the structured result back.

The Results Display, streaming, and rendering behavior described below is specific to direct mode, since that is where `app.py` renders the UI from live pipeline output. In RabbitMQ mode, `app.py` simply renders whatever complete result it receives back from the worker.

## Script

`app/app.py`

## Usage

```bash
# From project root:
.venv\Scripts\streamlit run app/app.py
```

The app opens in the browser at `http://localhost:8501`.

## Dependencies

- `streamlit>=1.30.0` — web framework
- All Phase 4 dependencies (anthropic, sentence-transformers, faiss-cpu, etc.)

## Prerequisites

The embedding pipeline (Phase 3) must have been run first to generate:
- `data/processed/faiss.index`
- `data/processed/metadata.json`

If these files are missing, the app shows an error with instructions to run the embedding pipeline.

## Features

### Query Input

Text input field at the top of the page. Empty queries show an info message prompting the user.

### Sidebar Controls

| Control | Type | Default | Description |
|---------|------|---------|-------------|
| Embedding model | Selectbox | — | Shown only when multiple indices are available in `data/processed/`; chooses which embedding model / index to query against |
| Top-k results | Slider (1–20) | 20 | Number of chunks to retrieve |
| Similarity threshold | Slider (0.0–1.0) | 0.0 | Filters out results below this score |
| Use HyDE | Checkbox | Off | Enables Hypothetical Document Embedding retrieval |
| Use hybrid search | Checkbox | On | Combines FAISS dense + BM25 keyword search with RRF |
| Use re-ranker | Checkbox | On | Re-scores results with cross-encoder |
| Re-ranker model | Selectbox | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for re-ranking; disabled when "Use re-ranker" is off |
| LLM backend | Radio | Anthropic | Choose between Claude Haiku (cloud) or Llama 3 (local) |

### Results Display

1. **LLM Response (structured, "### Answer")** — rendered in a bordered container as soon as both retrieval and generation have finished. In direct mode, `app.py` consumes the `pipeline.query_stream()` generator by silently accumulating LLM tokens into a plain string as they arrive — there is no live token-by-token UI update (no call to `st.write_stream()`, despite the pipeline itself supporting a streaming generator). Only once the stream is fully consumed is the collected string parsed as JSON into the `LLMResponse` Pydantic model. The parsed data provides:
   - **Confidence badge** — colored indicator: green ("high"), orange ("medium"), red ("low")
   - **Summary** — concise 1-2 sentence answer from the structured response
   - **Per-match citation** — the cited `chunk_index` shown as a caption below the lyric quote (e.g. "chunk #5"), so it can be spot-checked against `metadata.json`. `app.py` calls `verify_citations()` (see `Documentation/phase_4_query_pipeline.md`) after parsing the response and appends a note to the caption if the citation couldn't be verified
   - **Per-match explanations** — why each song was identified as a match
2. **HyDE Hypothesis** — if HyDE was used, the generated hypothetical lyric is shown in a collapsed expander for thesis demonstration purposes
3. **Retrieved Songs ("### Retrieved Songs")** — rendered after the "### Answer" section, once both retrieval and generation are complete (retrieval itself finishes first internally, but nothing is drawn to the page until generation is also done). Each result in a bordered card showing:
   - Song title, artist, album, release date
   - YouTube search link button (opens `youtube.com/results?search_query={artist}+{title}+official+music+video`)
   - Lyric excerpts as blockquotes with FAISS score, BM25 score (if hybrid), and cross-encoder score (if reranker)

### Error Handling

- **Missing pipeline data** — clear message with instructions to run the embedding pipeline
- **Ollama not running** — specific error with install/setup instructions
- **API errors** — generic error display for other failures
- **No results above threshold** — warning suggesting the user lower the similarity threshold

## Architecture

```
Streamlit App (app.py)
    │
    ├── RABBITMQ_URL unset → Direct mode
    │       ├── @st.cache_resource → load_pipeline(embedding_model, backend_key, llm_model, reranker_model)
    │       │       └── QueryPipeline (FAISS index, BM25 index, metadata, embedding model)
    │       └── pipeline.query_stream(query, top_k, use_hyde, use_hybrid, use_reranker)
    │               → tokens accumulated fully, then parsed as JSON (no live streaming to UI)
    │
    ├── RABBITMQ_URL set → RabbitMQ RPC mode
    │       ├── _query_via_rabbitmq() publishes to 'rag.queries' queue, blocks for RPC reply
    │       └── worker/worker.py owns QueryPipeline, runs pipeline.query(), publishes result back
    │
    ├── Sidebar → user parameters
    │
    └── Main area
            ├── Text input → query
            ├── "### Answer" card (confidence badge, summary, per-match explanations)
            ├── HyDE expander (conditional)
            └── "### Retrieved Songs" cards (filtered by threshold)
```

## Design Decisions

- **`@st.cache_resource`** — in direct mode, the pipeline (FAISS index, embedding model, Anthropic client) is loaded once via `load_pipeline()` and reused across all queries and page reruns for a given parameter combination. Without caching, each interaction would reload ~100MB+ of model weights. In RabbitMQ mode this caching instead happens inside `worker/worker.py`, keyed by `(embedding_model, reranker_model)`.
- **No pipeline logic in app.py** — in direct mode, the app imports `QueryPipeline` and delegates entirely; in RabbitMQ mode it doesn't even instantiate one, deferring to `worker/worker.py`. This keeps the frontend thin and ensures the CLI, app, worker, and evaluation notebook all use the same retrieval/synthesis logic.
- **YouTube search link (not direct video link)** — there's no reliable way to map song metadata to a specific YouTube video ID without a YouTube API key. A search URL is robust and always works.
- **Similarity threshold as post-filter** — filtering happens after retrieval, not during. This keeps the pipeline call simple and lets users adjust the threshold without re-querying FAISS.
- **LLM backend selector in sidebar** — allows live switching between Anthropic and Ollama. In direct mode, the pipeline is cached per `(embedding_model, backend_key, llm_model, reranker_model)` tuple, so switching backends, embedding model, or re-ranker model all trigger a pipeline reload.
- **Streaming retrieval, buffered synthesis** — in direct mode, `app.py` uses `pipeline.query_stream()` instead of `query()`. The generator yields retrieval results first, then yields LLM tokens — but `app.py` currently accumulates those tokens into a single string before parsing and rendering anything, rather than streaming them live via `st.write_stream()`. There is no live token-by-token UI update today, despite the pipeline itself supporting a streaming generator.
- **Hybrid search on by default** — BM25 keyword search complements dense retrieval for lyrics (exact matches for direct recall, semantic matches for descriptive queries). Can be toggled off in sidebar.
