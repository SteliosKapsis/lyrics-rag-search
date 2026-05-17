# Phase 5 — Streamlit Frontend

## Overview

A lightweight Streamlit web app that wraps the `QueryPipeline` from Phase 4, providing an interactive UI for the lyrics RAG search system. Designed for thesis demonstrations and user-facing exploration of retrieval techniques.

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
| Top-k results | Slider (1–20) | 5 | Number of chunks to retrieve |
| Similarity threshold | Slider (0.0–1.0) | 0.0 | Filters out results below this score |
| Use HyDE | Checkbox | Off | Enables Hypothetical Document Embedding retrieval |
| Use hybrid search | Checkbox | On | Combines FAISS dense + BM25 keyword search with RRF |
| Use re-ranker | Checkbox | Off | Re-scores results with cross-encoder |
| LLM backend | Radio | Anthropic | Choose between Claude Haiku (cloud) or Llama 3 (local) |

### Results Display

1. **LLM Response (streamed + structured)** — synthesized answer streams token-by-token in a bordered container at the top, using `st.write_stream()`. After the stream completes, the collected JSON is parsed into the `LLMResponse` Pydantic model. The parsed data provides:
   - **Confidence badge** — colored indicator: green ("high"), yellow ("medium"), red ("low")
   - **Summary** — concise 1-2 sentence answer from the structured response
   - **Per-match explanations** — why each song was identified as a match
2. **HyDE Hypothesis** — if HyDE was used, the generated hypothetical lyric is shown in a collapsed expander for thesis demonstration purposes
3. **Retrieved Songs** — displayed immediately after retrieval completes (before LLM response finishes streaming). Each result in a bordered card showing:
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
Streamlit App
    │
    ├── @st.cache_resource → QueryPipeline (loaded once, cached across reruns)
    │       ├── FAISS index
    │       ├── BM25 index
    │       ├── Metadata
    │       └── Embedding model
    │
    ├── Sidebar → user parameters
    │
    └── Main area
            ├── Text input → query
            ├── pipeline.query_stream(query, top_k, use_hyde, use_hybrid, use_reranker)
            ├── LLM response card (streamed via st.write_stream)
            ├── HyDE expander (conditional)
            └── Song result cards (filtered by threshold)
```

## Design Decisions

- **`@st.cache_resource`** — the pipeline (FAISS index, embedding model, Anthropic client) is loaded once and reused across all queries and page reruns. Without caching, each interaction would reload ~100MB+ of model weights.
- **No pipeline logic in app.py** — the app imports `QueryPipeline` and delegates entirely. This keeps the frontend thin and ensures the CLI, app, and evaluation notebook all use the same retrieval/synthesis logic.
- **YouTube search link (not direct video link)** — there's no reliable way to map song metadata to a specific YouTube video ID without a YouTube API key. A search URL is robust and always works.
- **Similarity threshold as post-filter** — filtering happens after retrieval, not during. This keeps the pipeline call simple and lets users adjust the threshold without re-querying FAISS.
- **LLM backend selector in sidebar** — allows live switching between Anthropic and Ollama. The pipeline is cached per `(embedding_model, backend_key, llm_model)` tuple, so switching backends loads a new pipeline instance.
- **Streaming LLM responses** — uses `query_stream()` instead of `query()`. The generator yields retrieval results first (displayed immediately as song cards), then yields LLM tokens rendered in real time via `st.write_stream()`. This makes the app feel much more responsive.
- **Hybrid search on by default** — BM25 keyword search complements dense retrieval for lyrics (exact matches for direct recall, semantic matches for descriptive queries). Can be toggled off in sidebar.
