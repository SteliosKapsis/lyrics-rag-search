You are an expert in LLM application development and RAG systems. I'm
building a lyrics search assistant for my thesis.

I have a FAISS index of lyric chunks, a BM25 keyword index (rank_bm25),
and a metadata JSON file (fields: text, title, artist, release_date,
album, chunk_index, total_chunks). I also have a Reranker class
(pipeline/reranker.py) that accepts a query and list of retrieved chunks
and returns them re-scored and re-sorted.

Help me build the query pipeline in Python (pipeline/query.py). It should:

1. Accept a natural language query (e.g., "that sad song about leaving
   home in the rain")

2. Support two retrieval modes, toggled by a use_hyde flag:
   - Standard: embed the raw query and retrieve from FAISS directly
   - HyDE (Hypothetical Document Embeddings): before retrieval, send the
     query to the LLM with a prompt asking it to generate a short
     hypothetical lyric that would answer the query, then embed that
     hypothetical lyric instead of the raw query. Explain the intuition
     behind HyDE and why it may improve retrieval for a lyrics use case.

3. Support hybrid retrieval (dense + sparse), toggled by a use_hybrid
   flag (default True):
   - When enabled: retrieve top candidates from BOTH FAISS (dense) and
     BM25 (sparse keyword search), then merge the two ranked lists using
     Reciprocal Rank Fusion (RRF)
   - When disabled: retrieve from FAISS only (original behavior)
   - RRF formula: score(d) = Σ 1 / (k + rank_i(d)) where k=60 is the
     standard constant and rank_i is the rank from each retrieval method
   - Fetch top (top_k * 3) from each method before fusing, then take the
     top_k fused results
   - Explain why hybrid retrieval helps: dense retrieval captures
     semantic meaning (good for descriptive queries like "sad song about
     rain"), while BM25 captures exact keyword matches (good for direct
     lyric recall like "never gonna give you up"). Combining them covers
     both query types.
   - Load the BM25 index from data/processed/bm25.pkl at pipeline init

4. Retrieve the top-k most similar chunks from FAISS (k configurable,
   default 5)

5. Optionally re-rank retrieved chunks using the Reranker class
   (use_reranker flag, default False)

6. Group retrieved chunks by song (title + artist) to deduplicate repeated
   choruses before sending to the LLM

7. Support two synthesis backends, toggled by a llm_backend parameter:
   - "anthropic": use Claude Haiku via the Anthropic Python SDK
     (default, as in previous phases)
   - "ollama": use a locally running Llama 3 model via the Ollama REST
     API (http://localhost:11434). Explain what Ollama is, how to install
     it, and how to pull and run Llama 3 locally.
   Both backends should use the same system prompt and produce the same
   output structure.

8. Support streaming responses from both LLM backends:
   - Add a query_stream() method that yields response tokens as they
     arrive, instead of waiting for the full response
   - For Anthropic: use client.messages.stream() context manager and
     yield each text delta from the stream
   - For Ollama: use the REST API with stream=true and yield each
     token from the chunked response
   - query_stream() should perform retrieval identically to query(),
     then yield a dict with retrieval_results and hyde_hypothesis
     first (so the caller can display those immediately), followed
     by individual string tokens from the LLM response
   - The non-streaming query() method should remain unchanged for
     backwards compatibility (used by evaluation notebook)

9. The LLM synthesis must return structured JSON output, not free text.
   Define a Pydantic schema for the response and enforce it:

   - For Anthropic: use tool_use — define a tool whose input_schema
     matches the Pydantic model, and Claude will return structured JSON
     as the tool call arguments. Parse with Pydantic for validation.
   - For Ollama: use the `format` parameter with the Pydantic model's
     JSON schema (`Model.model_json_schema()`), which forces the
     response to conform to the schema.

   The response schema should be:

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
       summary: str                # 1-2 sentence answer to the query
   ```

   The system prompt should still instruct the model to:
   - Identify the most likely matching song(s) from retrieved context
   - Show relevant lyric excerpts
   - Acknowledge uncertainty (confidence="low") if match is weak
   - Never fabricate lyrics or metadata not present in the retrieved context

   Both backends must produce the same schema. The non-streaming query()
   method should parse and return the validated Pydantic object. The
   streaming query_stream() method should stream raw text tokens (the
   JSON being built), then the caller parses after stream completes.

10. Return a structured dict with: query, retrieval_results (grouped by
    song, with FAISS score, BM25 score if hybrid was used, and
    cross-encoder score if re-ranking was used), llm_response (the
    parsed LLMResponse Pydantic object for query(), or raw text for
    query_stream()), hyde_hypothesis (the generated lyric if HyDE was
    used, else None)

11. Expose a QueryPipeline class importable by the Streamlit app (Phase 5)
    and the evaluation notebook (Phase 6), with a thin CLI wrapper

Configurable parameters (all exposed as CLI flags and constructor args):
   - --top-k (default 5)
   - --use-hyde (flag, default False)
   - --use-hybrid (flag, default True)
   - --use-reranker (flag, default False)
   - --llm-backend ("anthropic" or "ollama", default "anthropic")
   - --llm-model (default "claude-haiku-4-5-20251001" for anthropic,
     "llama3" for ollama)
   - --embedding-model (default "all-MiniLM-L6-v2")

12. Support alternative re-ranker models via a `--reranker-model` CLI
    flag (default "cross-encoder/ms-marco-MiniLM-L-6-v2"). Pass this
    through to the Reranker class constructor:
    `Reranker(model_name=reranker_model)`.
    Models to support:
    - `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, current default)
    - `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M params, same family,
      double depth)
    - `BAAI/bge-reranker-base` (110M params, different architecture)
    Also expose this as a `reranker_model` constructor argument on
    QueryPipeline so the evaluation notebook (Phase 6) can sweep across
    re-ranker models programmatically.

Think step-by-step through the prompt design for the LLM synthesis step
and explain any tradeoffs you make. If you're unsure about any Ollama API
behavior, say so.