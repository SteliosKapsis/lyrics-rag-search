# Langfuse Observability

Langfuse traces every query through the RAG pipeline: retrieval, re-ranking, LLM synthesis, and DeepEval evaluation scores — all visible in one dashboard.

---

## Setup

1. Create a free account at [langfuse.com](https://langfuse.com) (or self-host).
2. Go to **Project Settings → API Keys** and copy your public and secret keys.
3. Add to your `.env` file:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

If these keys are absent, the pipeline runs normally with no tracing — no code changes needed.

---

## What Gets Traced

Each call to `pipeline.query()` or `pipeline.query_stream()` creates a **trace** in Langfuse with the following nested spans:

```
rag-query  (top-level trace)
├── hyde-hypothesis            (optional, if use_hyde=True)
│   └── generation: hyde-hypothesis / hyde-hypothesis-ollama
├── retrieval                  (whole retrieval step — see per-stage breakdown below)
│   ├── dense-retrieval        (FAISS only — direct child when hybrid is off/BM25 missing)
│   ├── hybrid-fusion          (when hybrid retrieval runs)
│   │   ├── dense-retrieval    (nests here — hybrid-fusion calls retrieve() internally)
│   │   └── sparse-retrieval   (BM25)
│   └── rerank                 (cross-encoder, when use_reranker=True)
└── generation: anthropic-structured / ollama-structured   (skipped entirely if the
    retrieval gate abstains — see "Retrieval-gated abstention" below)
    └── model, token usage, input prompt, structured output
```

Streaming queries (`query_stream`) follow the same structure with `rag-query-stream` at the top.

**Per-stage retrieval breakdown:** `dense-retrieval`, `sparse-retrieval`, `hybrid-fusion`, and `rerank` each get their own span with their own candidate list (title/artist/chunk_index/score — not full lyric text, to keep payloads lean) and their own latency, from Langfuse's automatic span start/end timestamps. This means a slow or wrong-looking query can be diagnosed to a specific stage (e.g. "reranking took 1.9s and dominated total latency" or "sparse-retrieval returned nothing, dense-retrieval carried the result") directly from the trace, instead of only knowing "retrieval" as a whole was slow or empty.

### Retrieval-gated abstention

The `retrieval` span (and the root trace) also carry `retrieval_outcome` (`"empty"` / `"low_relevance"` / `"sufficient"`), `best_relevance_score`, and `abstain_threshold` in their metadata. When the outcome isn't `"sufficient"`, `query()`/`query_stream()` skip the `generation` span entirely — the LLM is never called, and the trace output shows `abstained: true`. This means the trace itself tells you whether a query got a real answer or was gated before ever reaching the LLM, without needing to inspect the response body.

### Trace metadata

Every trace carries:
- `session_id` — groups all queries from one Streamlit session or one RabbitMQ correlation
- `embedding_model`, `llm_model`, `llm_backend`
- `use_hyde`, `use_hybrid`, `use_reranker`, `top_k`
- `abstained` (root trace output) — whether this query was gated/abstained rather than answered

### Token usage

Token counts (`input_tokens`, `output_tokens`) are logged for all Anthropic API calls. Ollama calls log model name only (Ollama does not expose token counts in its REST API).

---

## Session Correlation

| Path | How session_id is set |
|---|---|
| Streamlit direct streaming | `st.session_state.session_id` — one UUID per browser tab session |
| Streamlit → RabbitMQ worker | `correlation_id` from the RabbitMQ message, forwarded through the worker |

---

## Evaluation Scores

When the evaluation notebook (`notebooks/evaluation.ipynb`) is run fresh (cache cleared), each judge query is traced and its DeepEval scores are uploaded to the corresponding Langfuse trace:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

Scores appear in Langfuse under each trace's **Scores** tab.

To re-run evaluation with fresh traces (to get new score uploads), delete the cached checkpoints:
```
data/processed/judge_inputs.pkl   # or wherever save_ckpt stores files
```

---

## Reading the Dashboard

1. **Traces** tab: see every query, its latency, cost, and session grouping
2. Click a trace → expand the span tree to see per-stage timing: dense-retrieval vs sparse-retrieval vs hybrid-fusion vs rerank vs LLM generation — not just "retrieval vs LLM" as one lump
3. **Generations** tab: filter by model name to compare Anthropic vs Ollama cost
4. **Scores** tab: filter by `faithfulness` / `answer_relevancy` etc. to compare configs
5. Use **Sessions** to see all queries from one Streamlit session in sequence
