# Lyrics-Based Semantic Search: A RAG Pipeline for Song Retrieval

## Overview

This thesis project implements a complete Retrieval-Augmented Generation (RAG) pipeline for lyrics-based song retrieval. Given a natural-language query, whether exact lyrics, a vague description, or a half-remembered fragment, the system retrieves the most relevant songs from a corpus of nearly 15,000 tracks and generates a structured, explainable answer using a large language model (LLM). The project spans the full machine-learning engineering lifecycle: data collection, cleaning, chunking, embedding, indexing, retrieval, LLM synthesis, evaluation, frontend development, and containerized deployment.

## Dataset

The corpus originates from the Billboard Hot 100 chart (1958-2021), comprising 29,681 unique song-artist pairs across 330,087 weekly chart entries. Lyrics and metadata were collected via the Genius API with rate limiting, exponential backoff, and resume support. After cleaning (removal of scraping artifacts, normalization of section headers, language filtering of non-English entries), the final dataset contains **14,997 songs** from **3,878 artists**, split into **175,792 semantically coherent chunks** using a hybrid chunking strategy that respects song structure (verses, choruses, bridges) with configurable size bounds (80-400 characters).

## Architecture

The retrieval pipeline supports multiple interchangeable components, enabling systematic experimentation:

- **Embedding & Indexing**: Chunks are embedded using sentence-transformer models or cloud APIs and indexed with FAISS (exact inner-product search). A key design choice is *contextual embedding*: each chunk is prepended with a metadata header (title, artist, album, section type) before encoding, enriching the vector representation with song identity. Three embedding options are supported: `all-MiniLM-L6-v2` (384d), `BAAI/bge-small-en-v1.5` (384d), and OpenAI `text-embedding-3-small` (1536d, cloud API). Non-contextual variants can be built with `--skip-contextual` for ablation studies. A parallel BM25 sparse index enables keyword-based retrieval.

- **Retrieval Strategies**: The pipeline implements three optional retrieval techniques evaluated in a full combinatorial grid (2^3 = 8 configurations): (1) hybrid retrieval combining FAISS and BM25 via Reciprocal Rank Fusion (RRF), (2) HyDE (Hypothetical Document Embeddings), which uses an LLM to generate synthetic lyrics from the query before retrieval, and (3) cross-encoder re-ranking to refine the initial top-k results. Three re-ranker models are supported: `ms-marco-MiniLM-L-6-v2`, `ms-marco-MiniLM-L-12-v2`, and `BAAI/bge-reranker-base`. Each technique is tested both in isolation against the baseline and in every combination, yielding 8 configurations per embedding model. A full combinatorial sweep (104 runs) across all tunable parameters and re-ranker models identifies the global optimum. Additionally, three parameter sensitivity studies sweep top-k, RRF fusion constant, and re-ranker candidate pool size.

- **LLM Synthesis**: Retrieved chunks are passed to an LLM (Anthropic Claude 3.5 Haiku or local Llama 3 via Ollama) that produces a structured JSON response containing song matches with titles, artists, confidence scores, and textual explanations. Structured output is enforced through Pydantic schemas and the LLM's tool-use capabilities.

- **Frontend**: A Streamlit web application provides an interactive interface with real-time streaming, configurable retrieval parameters, and result cards with similarity score breakdowns and YouTube search links.

## Evaluation & Findings

The evaluation uses a curated test set of 75 queries across four difficulty categories: direct lyric quotes (19), semantic/descriptive queries (19), partial/vague recall (19), and ambiguous queries (18). All songs are verified to exist in the indexed dataset. Retrieval quality is measured with Precision@k, Mean Reciprocal Rank (MRR), and Hit Rate, with per-query-type breakdowns, statistical significance testing (Wilcoxon signed-rank + paired bootstrap), and per-stage latency profiling.

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

- **Contextual embedding ablation**: The `--skip-contextual` flag and non-contextual index support are implemented. Building the non-contextual index and running the ablation comparison will quantify the isolated contribution of contextual embeddings. Anthropic reported a 35% reduction in retrieval failure from contextual embeddings alone, but the actual gain in a lyrics-specific domain may differ.
- **OpenAI embedding comparison**: The `--openai` flag and OpenAI query-time embedding are implemented. Building the OpenAI index will enable a local vs. cloud embedding quality comparison across the full technique grid.
- **Chunk size experiments**: The evaluation framework auto-detects variant chunking indices. Building indices with different chunk sizes (200, 400, 600 chars) and overlap values (0, 1, 2 lines) will reveal the optimal chunking parameters for lyrics retrieval.
- **LLM-as-judge (dual judges)**: Experiment 7 uses Claude Haiku and GPT-4o-mini as independent judges via DeepEval's built-in `AnthropicModel` and `GPTModel`. Inter-judge agreement table reported alongside per-judge heatmaps. Ollama no longer required.
- **RAG vs. Pure LLM baseline (Experiment 12, Section 8)**: Five-condition comparison across all 75 queries — Claude (no ctx), GPT (no ctx), RAG retrieval only, RAG+Claude, RAG+GPT. Directly answers whether retrieval adds value over a knowledgeable LLM alone.

## Technology Stack

Python 3.11, FAISS, sentence-transformers, rank-bm25, Anthropic API (Claude 3.5 Haiku), Ollama (Llama 3), Streamlit, Pydantic, DeepEval, Docker. The full codebase is organized into modular pipeline stages (`pipeline/`), a web frontend (`app/`), evaluation notebooks (`notebooks/`), and phase-specific documentation (`documentation/`).

## Current Status

All pipeline stages are complete and evaluated. The best configuration from Phase 6 has been promoted as the app default (`text-embedding-3-small`, `top_k=20`, `use_hybrid=True`, `use_reranker=True`, `fetch_k_multiplier=5`). The system is containerized with Docker Compose including a RabbitMQ worker for asynchronous query dispatch.
