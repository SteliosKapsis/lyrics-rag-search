# Lyrics-Based Semantic Search: A RAG Pipeline for Song Retrieval

## Overview

This thesis project implements a complete Retrieval-Augmented Generation (RAG) pipeline for lyrics-based song retrieval. Given a natural-language query, whether exact lyrics, a vague description, or a half-remembered fragment, the system retrieves the most relevant songs from a corpus of nearly 15,000 tracks and generates a structured, explainable answer using a large language model (LLM). The project spans the full machine-learning engineering lifecycle: data collection, cleaning, chunking, embedding, indexing, retrieval, LLM synthesis, evaluation, frontend development, and containerized deployment.

## Dataset

The corpus originates from the Billboard Hot 100 chart (1958-2021), comprising 29,681 unique song-artist pairs across 330,087 weekly chart entries. Lyrics and metadata were collected via the Genius API with rate limiting, exponential backoff, and resume support. After cleaning (removal of scraping artifacts, normalization of section headers, language filtering of non-English entries) and a post-cleaning data quality pass (deduplication of repeat chart entries, removal of non-song Genius mismatches — see `Documentation/phase_1.5_cleaning.md`), the final dataset contains **14,759 songs** from **3,739 artists**, split into **130,336 semantically coherent chunks** using a hybrid chunking strategy that respects song structure (verses, choruses, bridges) with configurable size bounds (80-400 characters).

## Architecture

The retrieval pipeline supports multiple interchangeable components, enabling systematic experimentation:

- **Embedding & Indexing**: Chunks are embedded using sentence-transformer models or cloud APIs and indexed with FAISS (exact inner-product search). A key design choice is *contextual embedding*: each chunk is prepended with a metadata header (title, artist, album, section type) before encoding, enriching the vector representation with song identity. The embedding pipeline's `--model` flag supports `all-MiniLM-L6-v2` (384d), `BAAI/bge-small-en-v1.5` (384d), `all-mpnet-base-v2` (768d), and in principle any sentence-transformers model name, plus OpenAI `text-embedding-3-small` (1536d, cloud API) via `--openai`; the Streamlit UI exposes the first three of these (MiniLM, bge, and OpenAI) at query time. Non-contextual variants can be built with `--skip-contextual` for ablation studies. A parallel BM25 sparse index enables keyword-based retrieval. Each index is paired with a model-identity sidecar (`{index}_info.json`) that `QueryPipeline` asserts against at load time — it refuses to start rather than silently serve results from a query-model/index-model mismatch, previously the single highest-severity unguarded failure mode in the pipeline. Embedding regeneration is checkpointed (resumable after an interruption, whether local-model or `--openai`) rather than restarting from scratch.

- **Retrieval Strategies**: The pipeline implements three optional retrieval techniques evaluated in a full combinatorial grid (2^3 = 8 configurations): (1) hybrid retrieval combining FAISS and BM25 via Reciprocal Rank Fusion (RRF), (2) HyDE (Hypothetical Document Embeddings), which uses an LLM to generate synthetic lyrics from the query before retrieval, and (3) cross-encoder re-ranking to refine the initial top-k results. Three re-ranker models are supported: `ms-marco-MiniLM-L-6-v2`, `ms-marco-MiniLM-L-12-v2`, and `BAAI/bge-reranker-base`. Each technique is tested both in isolation against the baseline and in every combination, yielding 8 configurations per embedding model. A full combinatorial sweep (104 runs) across all tunable parameters and re-ranker models identifies the global optimum. Additionally, three parameter sensitivity studies sweep top-k, RRF fusion constant, and re-ranker candidate pool size.

- **LLM Synthesis**: Retrieved chunks are passed to an LLM (Anthropic Claude Haiku 4.5 or local Llama 3 via Ollama) that produces a structured JSON response containing song matches with titles, artists, textual explanations, and a `chunk_index` binding each cited lyric excerpt to the specific retrieved chunk it was drawn from (checked, best-effort, by `verify_citations()` against the chunk's real text) — along with a single overall confidence rating (high/medium/low) for the response. Structured output is enforced through Pydantic schemas — via tool-use for the Anthropic backend, and via JSON-schema-constrained generation for the Ollama backend.

- **Retrieval-gated abstention**: before synthesis, retrieval confidence is classified as empty / low-relevance / sufficient; on anything but "sufficient" the LLM is never called at all — the pipeline returns a structured `abstained=true` response with a reason instead of risking a fabricated match. The system prompt frames abstaining as the correct, expected choice rather than a fallback. The score threshold that drives this classification is an explicit, documented placeholder, not yet tuned against real data.

- **Observability**: every query is traced end-to-end via Langfuse, including separate per-stage spans for dense retrieval, sparse retrieval, RRF fusion, and re-ranking (not one opaque "retrieval" span) — so a slow or empty result can be attributed to a specific stage rather than the pipeline as a whole.

- **Frontend**: A Streamlit web application provides an interactive interface with real-time streaming, configurable retrieval parameters, and result cards with similarity score breakdowns and YouTube search links.

## Evaluation & Findings

> **⚠️ FLAGGED — the numbers and conclusions in this entire section (technique grid, parameter sensitivity tables, Key Findings, Domain Transferability) do not match the actual checkpointed evaluation results and need to be re-verified/rewritten before being cited.** Spot-checked against `notebooks/checkpoints/all_results.pkl` and `topk_results.pkl` — the same source `Documentation/phase_6_evaluation.md`'s already-corrected numbers use. Confirmed mismatch on the main technique grid: the real MiniLM values are Baseline (FAISS only) 0.571, + Re-ranker 0.674, + Hybrid 0.590, **+ Hybrid + Re-ranker 0.698 (the actual best config, not Baseline + Re-ranker as stated below)**, + HyDE 0.060, + HyDE + Re-ranker 0.127, + HyDE + Hybrid 0.036, + HyDE + Hybrid + Re-ranker 0.044 — not close to the table below, and the Top-k Sensitivity table below doesn't match its corresponding checkpoint either (compare against `Documentation/phase_6_evaluation.md`'s Experiment 2 table, which was already corrected). This whole section appears to predate the finalized evaluation run and was never reconciled with it. Left as-is (not rewritten) per explicit decision — `Documentation/phase_6_evaluation.md` is the source of truth for these results in the meantime.

The evaluation uses a curated test set — see `Documentation/phase_6_evaluation.md`'s "Test Set" section for the current, versioned (`TEST_SET_VERSION`) description; this file's query-count description below predates that versioning. Retrieval quality is measured with Precision@k, Mean Reciprocal Rank (MRR), and Hit Rate, with per-query-type breakdowns, statistical significance testing (Wilcoxon signed-rank + paired bootstrap), and per-stage latency profiling.

### Technique comparison grid (8 configurations x 2 embedding models, k=5)

| Configuration | MiniLM MRR | bge MRR |
|---|---|---|
| **Baseline + Re-ranker** | **0.653** | **0.620** |
| Baseline (FAISS only) | 0.563 | 0.513 |
| + Hybrid + Re-ranker | 0.553 | 0.553 |
| + HyDE + Re-ranker | 0.460 | 0.300 |
| + Hybrid | 0.451 | 0.463 |
| + HyDE + Hybrid + Re-ranker | 0.260 | 0.160 |
| + HyDE | 0.240 | 0.160 |
| + HyDE + Hybrid | 0.113 | 0.223 |

### Parameter sensitivity studies

**Top-k sensitivity (MiniLM, Baseline + Re-ranker):**

| top_k | Baseline MRR | Baseline HR | + Re-ranker MRR | + Re-ranker HR |
|---|---|---|---|---|
| 3 | 0.540 | 0.560 | 0.580 | 0.600 |
| 5 | 0.563 | 0.640 | 0.653 | 0.680 |
| 10 | 0.573 | 0.680 | 0.687 | 0.760 |
| **20** | **0.585** | **0.760** | **0.767** | **0.880** |

The re-ranker's benefit scales with k: at k=20, it achieves **0.767 MRR and 88% Hit Rate**, finding correct songs buried in positions 11-20 and promoting them to the top. The re-ranker advantage grows from +0.040 MRR at k=3 to +0.182 at k=20.

**RRF fusion constant tuning (MiniLM, k=5):**

| rrf_k | Hybrid MRR | Hybrid + RR MRR |
|---|---|---|
| **20** | 0.451 | **0.633** |
| 40 | 0.451 | 0.593 |
| 60 (default) | 0.451 | 0.553 |
| 80 | 0.451 | 0.553 |
| 100 | 0.451 | 0.553 |

Hybrid retrieval without re-ranking is completely insensitive to the RRF constant (identical 0.451 across all values). Lower rrf_k improves Hybrid + Re-ranker by weighting FAISS results more heavily, but even the best (0.633 at rrf_k=20) cannot surpass pure Baseline + Re-ranker (0.653).

**Re-ranker candidate pool size (MiniLM, k=5):**

| Multiplier | fetch_k | Baseline + RR MRR | Baseline + RR HR | Hybrid + RR MRR | Hybrid + RR HR |
|---|---|---|---|---|---|
| 2x | 10 | 0.620 | 0.640 | 0.523 | 0.600 |
| 3x (default) | 15 | 0.653 | 0.680 | 0.553 | 0.600 |
| 5x | 25 | 0.653 | 0.720 | 0.633 | 0.720 |
| **8x** | **40** | 0.693 | 0.760 | **0.753** | **0.840** |

With a sufficiently large candidate pool (8x), Hybrid + Re-ranker (0.753 MRR, 0.840 HR) surpasses Baseline + Re-ranker (0.693/0.760) for the first time. At high multipliers, BM25's additional candidates give the re-ranker more material to work with, overcoming the noise that harms lower-pool configurations.

### Key findings

1. **FAISS + Cross-encoder re-ranking is the optimal pipeline.** At the default k=5, Baseline + Re-ranker (0.653 MRR) is the best configuration. At k=20 it reaches **0.767 MRR and 88% Hit Rate**, the highest scores achieved. The re-ranker was only discovered to be effective in isolation after expanding the evaluation grid; earlier experiments that only paired it with hybrid retrieval masked its true effectiveness.

2. **The re-ranker's benefit scales with candidate pool size.** Whether increased via higher top-k (k=20 yields +0.182 MRR) or higher fetch multiplier (8x yields +0.040 MRR over 3x), giving the cross-encoder more candidates consistently improves performance. This is the single most impactful tunable parameter in the pipeline.

3. **Contextual embeddings make dense retrieval a strong baseline.** Prepending metadata headers (title, artist, section type) to each chunk before embedding achieves MRR 0.563 with no additional techniques. This confirms that contextual embedding can substitute for more complex retrieval strategies.

4. **HyDE consistently degrades performance (-57% to -86% MRR).** Unlike general document retrieval where queries and documents occupy different semantic spaces, lyrics queries often already contain actual song text. HyDE's generated hypotheses replace authentic lyrics with hallucinated content, pushing the query further from the target chunks.

5. **Hybrid search harms at default settings but helps at scale.** At default parameters (rrf_k=60, fetch 3x), BM25 dilutes the dense retrieval signal. However, with a large re-ranker candidate pool (8x multiplier), Hybrid + Re-ranker surpasses Baseline + Re-ranker, showing that BM25's keyword matching adds value when the cross-encoder has enough capacity to filter noise. RRF tuning further confirms that weighting FAISS higher (lower rrf_k) improves Hybrid configurations.

6. **Query specificity is the dominant factor.** Improving test queries from generic descriptions to lyrics-grounded formulations improved baseline MRR by +51% (0.373 to 0.563), a larger gain than any technique combination at fixed k=5. For lyrics search, the quality of the user's query matters more than retrieval sophistication.

### Domain transferability of techniques

The techniques that underperformed in lyrics search are well-established in other domains where their assumptions hold:

- **HyDE** is effective when queries and documents occupy different semantic spaces. In academic paper search, a query like "methods for reducing LLM hallucination" looks nothing like an abstract, so generating a hypothetical abstract bridges the gap. Similarly, legal document retrieval (short questions vs. lengthy case law), medical literature search (patient symptoms vs. clinical papers), and FAQ retrieval (user questions vs. pre-written answers) all exhibit the query-document mismatch that HyDE is designed to address. In lyrics search, queries often already contain the target document's text, making the hypothesis redundant or harmful.

- **Hybrid search (BM25 + dense)** excels in domains with specialized vocabulary that embedding models underrepresent: biomedical literature (drug names, gene symbols, ICD codes), patent search (technical jargon, patent numbers), e-commerce (product SKUs, brand names), and code search (exact function/class names). In these settings, BM25's exact keyword matching catches critical terms that dense models compress or lose. In lyrics search, the vocabulary is conversational English with no specialized terminology, and contextual embeddings already capture title/artist keywords. However, our fetch-k multiplier experiment shows that even in lyrics search, BM25 becomes useful when paired with a strong re-ranker and a large candidate pool.

### Future work

> **⚠️ FLAGGED — the five items below are described as not-yet-done, but `Documentation/phase_6_evaluation.md` documents all five as already completed** (Experiments 13, "OpenAI" default-config comparison, Experiment 5, and Experiments 7/12 respectively — two of the bullets below even describe their own results in the past tense while sitting under a "Future work" heading). This section appears to predate those experiments' completion and was never updated. Left as-is (not rewritten) alongside the numbers flag above — see `Documentation/phase_6_evaluation.md` for the actual current-vs-future status of each.

- **Contextual embedding ablation**: The `--skip-contextual` flag and non-contextual index support are implemented. Building the non-contextual index and running the ablation comparison will quantify the isolated contribution of contextual embeddings. Anthropic reported a 35% reduction in retrieval failure from contextual embeddings alone, but the actual gain in a lyrics-specific domain may differ.
- **OpenAI embedding comparison**: The `--openai` flag and OpenAI query-time embedding are implemented. Building the OpenAI index will enable a local vs. cloud embedding quality comparison across the full technique grid.
- **Chunk size experiments**: The evaluation framework auto-detects variant chunking indices. Building indices with different chunk sizes (200, 400, 600 chars) and overlap values (0, 1, 2 lines) will reveal the optimal chunking parameters for lyrics retrieval.
- **LLM-as-judge (dual judges)**: Experiment 7 uses Claude Haiku and GPT-4o-mini as independent judges via DeepEval's built-in `AnthropicModel` and `GPTModel`. Inter-judge agreement table reported alongside per-judge heatmaps. Ollama no longer required.
- **RAG vs. Pure LLM baseline (Experiment 12, Section 8)**: Five-condition comparison across all 75 queries — Claude (no ctx), GPT (no ctx), RAG retrieval only, RAG+Claude, RAG+GPT. Directly answers whether retrieval adds value over a knowledgeable LLM alone.

### Genuinely remaining future work (per `Instructions/AUDIT_RESULTS.md`)

- **Tune `abstain_score_threshold`**: currently a documented placeholder (`0.3`), not yet swept against the new `UNANSWERABLE_QUERIES` subset's actual abstention-rate results (`evaluate_abstention()`, Section 7.9).
- **Claim-level citation verification by entailment**: the current `verify_citations()` checks only that `relevant_lyric` plausibly matches its cited chunk — it does not decompose or verify `explanation`/`summary` against the source (file 08's roadmap item).
- **CI-driven regression gate**: no automated pipeline currently blocks a change that regresses retrieval quality, faithfulness, or false-refusal rate — these are audit-flagged blocking gaps, not yet closed.

## Technology Stack

Python 3.11, FAISS, sentence-transformers, rank-bm25, Anthropic API (Claude Haiku 4.5), Ollama (Llama 3), Streamlit, Pydantic, Langfuse (observability/tracing), DeepEval, Docker. The full codebase is organized into modular pipeline stages (`pipeline/`), a web frontend (`app/`), evaluation notebooks (`notebooks/`), and phase-specific documentation (`documentation/`).

## Current Status

All pipeline stages are complete and evaluated. `QueryPipeline`'s actual default is `all-MiniLM-L6-v2` (`top_k=20`, `use_hybrid=True`, `use_reranker=True`, `fetch_k_multiplier=5`); `text-embedding-3-small` is offered as a UI-selectable alternative, auto-selected only when its index already exists — see `Documentation/architecture.md`'s "Default Configuration" table (not the flagged numbers above). The system is containerized with Docker Compose including a RabbitMQ worker for asynchronous query dispatch.

**Local index regeneration required before first run or redeploy**: as of the model/index safety check, `QueryPipeline` refuses to load any FAISS index lacking a `{index}_info.json` sidecar. All 5 local index variants must be regenerated via `pipeline/embedding.py` (each interruptible/resumable) before the app or worker will start. The golden set used for evaluation is versioned (`TEST_SET_VERSION`, currently `2.1.0`) and now includes 20 deliberately unanswerable queries alongside the 75 answerable ones, so hallucination/abstention behavior is measurable, not just retrieval accuracy.

**Corpus regeneration in progress**: the dataset figures above (14,759 songs, 130,336 chunks) reflect a post-cleaning dedup + non-song-contamination filter pass and a chunking algorithm fix (see `Documentation/phase_1.5_cleaning.md` and `Documentation/phase_2_chunking.md`'s Design Decisions), both applied after the numbers in the flagged "Evaluation & Findings" section above were originally produced. The 5 index variants and the evaluation notebook are being regenerated against this corrected corpus; the previously-committed `notebooks/checkpoints/*.pkl` results (which the flagged section's numbers were checked against) were from the *previous* 175,792-chunk/14,997-song corpus and have been cleared to force a fresh run — a second reconciliation pass, replacing fresh, correctly-corpused numbers into the flagged section, is needed once that run completes.
