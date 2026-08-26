# Documentation Index

This folder has one file per pipeline stage plus a few cross-cutting references. If you're new to the project (or coming back after a break), read in this order.

## 1. Start here — the big picture

| File | What it covers |
|---|---|
| [`project_summary.md`](project_summary.md) | Thesis-level overview: what was built and why, dataset stats, architecture summary, current status. Read this first. |
| [`architecture.md`](architecture.md) | End-to-end technical architecture: the 7-stage pipeline diagram, data flow between stages, the critical FAISS/metadata/BM25 positional-alignment invariant, and default configuration values. |

## 2. Deep reference — one file per pipeline stage

Read these as needed once you know the overall shape from step 1.

| File | Pipeline stage | Source |
|---|---|---|
| [`phase_1_ingestion.md`](phase_1_ingestion.md) | Data collection from the Genius API | `collection/ingest_lyrics.py` |
| [`phase_1.5_cleaning.md`](phase_1.5_cleaning.md) | Cleaning, normalization, language/validity filtering | `collection/clean_lyrics.py` |
| [`phase_2_chunking.md`](phase_2_chunking.md) | Hybrid section/stanza chunking strategy | `pipeline/chunking.py` |
| [`phase_3_embedding_indexing.md`](phase_3_embedding_indexing.md) | Embedding models, contextual embedding, FAISS/BM25 indexing, the model-identity sidecar, checkpointed regeneration | `pipeline/embedding.py` |
| [`phase_4_query_pipeline.md`](phase_4_query_pipeline.md) | Dense/sparse/hybrid/HyDE retrieval, cross-encoder re-ranking, retrieval-gated abstention, citation linking, LLM synthesis | `pipeline/query.py`, `pipeline/reranker.py` |
| [`phase_5_streamlit.md`](phase_5_streamlit.md) | The interactive frontend | `app/app.py` |
| [`worker_rabbitmq.md`](worker_rabbitmq.md) | The RabbitMQ worker / RPC pattern used in Docker deployment | `worker/worker.py` |
| [`phase_7_deployment.md`](phase_7_deployment.md) | Docker Compose setup, volumes, prerequisites, rebuild/restart flow | `Dockerfile`, `docker-compose.yml` |
| [`phase_6_evaluation.md`](phase_6_evaluation.md) | Evaluation methodology: the golden set, metrics, all 14 experiments | `notebooks/evaluation.ipynb` |

## 3. Reference / lookup

| File | What it covers |
|---|---|
| [`api_reference.md`](api_reference.md) | Function-by-function reference for the query pipeline and re-ranker (signatures, parameters, return shapes, examples) |
| [`../langfuse.md`](../langfuse.md) | Observability/tracing reference — lives at the project root, not in this folder |
| [`future_research.md`](future_research.md) | Experiments and extensions not yet implemented or not yet run |

## 4. Known caveats — read before citing numbers

- **`project_summary.md`'s "Evaluation & Findings" and "Future work" sections are flagged stale** (marked inline with `⚠️ FLAGGED`) — their numbers and completion claims don't match the real checkpointed results. Use `phase_6_evaluation.md` as the source of truth for evaluation numbers instead.
- **`phase_6_evaluation.md`'s Experiment 14 (abstention rate) has no result yet** — it describes the methodology, but the actual number requires regenerating the FAISS/BM25 indices and re-running the notebook first (see `phase_3_embedding_indexing.md` and `phase_7_deployment.md` for why the indices need regenerating).
