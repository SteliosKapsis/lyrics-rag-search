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
├── retrieval          (FAISS + BM25 + RRF + cross-encoder)
├── hyde-hypothesis    (optional, if use_hyde=True)
│   └── generation: hyde-hypothesis / hyde-hypothesis-ollama
└── generation: anthropic-structured / ollama-structured
    └── model, token usage, input prompt, structured output
```

Streaming queries (`query_stream`) follow the same structure with `rag-query-stream` at the top.

### Trace metadata

Every trace carries:
- `session_id` — groups all queries from one Streamlit session or one RabbitMQ correlation
- `embedding_model`, `llm_model`, `llm_backend`
- `use_hyde`, `use_hybrid`, `use_reranker`, `top_k`

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
2. Click a trace → expand the span tree to see retrieval timing vs LLM timing
3. **Generations** tab: filter by model name to compare Anthropic vs Ollama cost
4. **Scores** tab: filter by `faithfulness` / `answer_relevancy` etc. to compare configs
5. Use **Sessions** to see all queries from one Streamlit session in sequence
