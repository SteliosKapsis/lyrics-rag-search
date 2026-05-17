# Phase 5 — Streamlit Frontend

You are an expert in building LLM-powered web applications with Streamlit. 
I'm building a lyrics search assistant for my thesis.

I have a working RAG query pipeline (Python module) that:
- Takes a natural language query string
- Retrieves relevant lyric chunks using hybrid search (FAISS dense +
  BM25 sparse, fused with Reciprocal Rank Fusion)
- Sends them to Claude via the Anthropic API
- Returns a structured JSON response (validated Pydantic model) with:
  - matches: list of {title, artist, album, release_date,
    relevant_lyric, explanation}
  - confidence: "high" / "medium" / "low"
  - summary: 1-2 sentence answer
  - Also includes: retrieval_results (grouped by song, with FAISS
    cosine score, BM25 score if hybrid was used, and cross-encoder
    score if re-ranking was used), and optionally hyde_hypothesis
- Supports streaming responses via query_stream() which yields tokens
  as they arrive from the LLM, then the caller parses the collected
  JSON after the stream completes

Help me build a Streamlit app (app.py) that wraps this pipeline.
The app should:

1. A text input for the user's natural language query
   (e.g., "that sad song about leaving home in the rain")

2. A sidebar with configurable parameters:
   - Top-k results (slider, 1–20, default 5)
   - Similarity threshold (slider, 0.0–1.0, default 0.0) — only show
     results above this score
   - Use HyDE toggle (checkbox, default off) — when enabled, show the
     generated hypothetical lyric in an expandable section below the
     results so the user can see what the model imagined
   - Use hybrid search toggle (checkbox, default on) — when enabled,
     combines FAISS dense retrieval with BM25 keyword search. When off,
     uses FAISS only.
   - Use re-ranker toggle (checkbox, default off)
   - LLM backend selector (radio button: "Anthropic (Claude Haiku)"
     or "Ollama (Llama 3 local)", default Anthropic)

3. A results display that shows for each matched song:
   - Song title, artist, album, release date
   - The relevant lyric excerpt(s), visually distinct
     (e.g., blockquote or highlighted box)
   - The FAISS similarity score, and the cross-encoder score if
     re-ranking was used (display both clearly labelled)
   - A YouTube search link constructed from the artist and title
     using the pattern:
     https://www.youtube.com/results?search_query={artist}+{title}+official+music+video
     Display this as a clickable button or link that opens in a new
     tab, labelled "▶ Find on YouTube"

4. The LLM's synthesized answer displayed prominently above the
   individual chunk results, using streaming output:
   - Use the pipeline's query_stream() method instead of query()
   - Display results (song cards) immediately as soon as retrieval
     completes, while the LLM response streams in above them
   - Use st.write_stream() or equivalent to render LLM tokens as they
     arrive, so the user sees the answer being written in real time
   - After the stream completes, parse the collected JSON into the
     LLMResponse Pydantic model. Use the parsed structured data to
     display the confidence level (e.g., a colored badge: green for
     "high", yellow for "medium", red for "low") and the summary
   - This gives a much more responsive feel than waiting for the full
     LLM response before showing anything

5. If HyDE was used, show the generated hypothetical lyric in a
   collapsed expander below the main results, labelled
   "HyDE: Hypothetical lyric used for retrieval" — this is useful
   for thesis demonstrations to show what the model imagined

6. Graceful handling of:
   - Empty queries
   - No results found above the similarity threshold
   - API errors from Anthropic or Ollama
   - Ollama not running locally (clear error message with instructions
     to start it)

Keep the UI clean and minimal. Do not add features I haven't asked for.
Import the QueryPipeline class from pipeline/query.py — do not duplicate
its logic in the app. Use st.cache_resource to load the pipeline once
at startup so the FAISS index and embedding model are not reloaded on
every query.
