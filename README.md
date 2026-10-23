# Lyrics-Based Semantic Search: A RAG Pipeline for Song Retrieval

A complete Retrieval-Augmented Generation (RAG) system for finding songs from natural-language queries — whether an exact lyric fragment, a vague description, or a half-remembered line. Built as a thesis project, the system spans the full ML engineering lifecycle: data collection, cleaning, chunking, embedding, indexing, retrieval, LLM synthesis, evaluation, frontend, and containerized deployment.

Beyond basic retrieval, the pipeline includes production-style safety mechanisms often missing from RAG demos: a model/index identity check that refuses to start on a mismatch, claim-level citation linking back to specific chunks, and a retrieval-gated abstention mechanism so the system says "I don't know" instead of guessing when nothing in the corpus actually matches.

## Dataset

- **Source**: Billboard Hot 100 (1958–2021), 29,681 unique song-artist pairs
- **Lyrics**: Collected via Genius API
- **Final corpus**: 14,759 songs · 3,739 artists · 130,336 chunks (after dedup and non-song-content filtering — see `Documentation/phase_1.5_cleaning.md`)

## Quick Start

### Prerequisites

- Python 3.11
- Anthropic API key (Claude Haiku for LLM synthesis)
- OpenAI API key (optional — for `text-embedding-3-small`)
- Docker + Docker Compose (for containerized deployment)

### Local development

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and OPENAI_API_KEY

# 4. Run the app (assumes pipeline data already built)
.venv\Scripts\streamlit run app/app.py
```

### Docker (full stack with RabbitMQ worker)

```bash
# Build and start all services (RabbitMQ + Worker + Streamlit)
docker compose up --build

# App:              http://localhost:8501
# RabbitMQ UI:      http://localhost:15672  (guest / guest)
```

The `worker` and `app` containers bind-mount `./data/processed` read-only;
the indexes live on the host and are not baked into the image. To
(re)build them inside Docker, see the next section.

### Docker (rebuild the indexes)

Chunking and embedding are exposed as one-shot services behind the
`pipeline` profile (they don't auto-start with `docker compose up`):

```bash
# Chunk cleaned lyrics → data/processed/chunks.jsonl
docker compose run --rm chunker

# Embed + index. Each variant writes to its own suffixed files
# (see pipeline/embedding.py for the suffix scheme).
docker compose run --rm indexer                                  # all-MiniLM-L6-v2 (default)
docker compose run --rm indexer --model BAAI/bge-small-en-v1.5   # BGE
docker compose run --rm indexer --openai                         # text-embedding-3-small (needs OPENAI_API_KEY)
docker compose run --rm indexer --skip-contextual                # ablation: no metadata header
```

Both services bind-mount `./data` read-write, so outputs land directly
on the host and are picked up by `worker`/`app` on next restart.

### Build the pipeline from scratch

Run each stage in order. Each stage produces artifacts consumed by the next.

> **Note:** `charts.csv` (Billboard Hot 100 input) is not committed to the
> repo. Get a copy from the [Billboard Hot 100 dataset on Kaggle](https://www.kaggle.com/datasets/dhruvildave/billboard-the-hot-100-songs)
> (or any equivalent source) and place it at the project root before running
> Stage 1.

```bash
# Stage 1: Fetch lyrics from Genius API
.venv\Scripts\python collection\ingest_lyrics.py --input charts.csv --output data/raw/lyrics.json

# Stage 2: Clean artifacts and filter non-English songs
.venv\Scripts\python collection\clean_lyrics.py

# Stage 3: Chunk lyrics into semantically coherent segments
.venv\Scripts\python pipeline\chunking.py

# Stage 4: Embed chunks and build FAISS + BM25 indices
.venv\Scripts\python pipeline\embedding.py                          # all-MiniLM-L6-v2 (default)
.venv\Scripts\python pipeline\embedding.py --model BAAI/bge-small-en-v1.5
.venv\Scripts\python pipeline\embedding.py --openai                 # text-embedding-3-small
.venv\Scripts\python pipeline\embedding.py --skip-contextual        # ablation: no metadata header

# Stage 5: Query the pipeline (CLI)
.venv\Scripts\python pipeline\query.py "I stay out too late got nothin in my brain"
```

## Project Structure

```
RAG system/
│
├── collection/                  # Stage 1-2: Data ingestion and cleaning
│   ├── ingest_lyrics.py         # Genius API fetcher with retry + resume
│   └── clean_lyrics.py          # Artifact removal, language filtering, validation
│
├── pipeline/                    # Stage 3-5: Core RAG components
│   ├── chunking.py              # Section-aware lyrics splitting
│   ├── embedding.py             # Sentence-transformers / OpenAI embedding + FAISS + BM25
│   ├── query.py                 # Orchestration: grouping, abstention gate, citation linking
│   ├── retriever.py             # Dense/sparse/hybrid retrieval mechanics (composed into QueryPipeline)
│   ├── llm_backend.py           # Anthropic/Ollama API calling (composed into QueryPipeline)
│   └── reranker.py              # Cross-encoder re-scoring
│
├── scripts/                     # One-off data-quality and golden-set tooling
│   ├── dedupe_cleaned_lyrics.py       # Removes duplicate (title, artist) chart re-entries
│   ├── filter_non_song_entries.py     # Drops non-song Genius mismatches
│   └── backfill_chunk_ids.py          # Semi-automated chunk-level ground truth for the eval set
│
├── app/                         # Stage 6: Streamlit web frontend
│   └── app.py                   # Interactive UI with streaming, RabbitMQ path
│
├── worker/                      # Async query processing
│   └── worker.py                # RabbitMQ consumer wrapping QueryPipeline
│
├── notebooks/                   # Evaluation and analysis
│   └── evaluation.ipynb         # 14 experiments, 95-query test set (75 answerable + 20 unanswerable)
│
├── data/
│   ├── raw/                     # lyrics.json, cleaned_lyrics.json
│   ├── processed/               # faiss*.index, metadata*.json, bm25*.pkl, chunks.jsonl
│   └── failed/                  # failed_fetches.csv, validation_report.csv
│
├── Documentation/               # Phase-by-phase design docs, architecture, API reference
│
├── Dockerfile                   # Multi-stage Python 3.11 image
├── docker-compose.yml           # RabbitMQ + Worker + App + optional Ollama
├── requirements.txt             # Full dev dependencies
├── requirements-docker.txt      # Runtime-only dependencies
├── .env.example                 # API key template
└── .streamlit/config.toml       # Disables file watcher (suppresses transformers noise)
```

## Architecture Overview

```
User Query
    │
    ▼
Streamlit App (app/app.py)
    │
    ├── [RABBITMQ_URL set] ──► RabbitMQ Queue (rag.queries)
    │                                │
    │                          Worker (worker/worker.py)
    │                                │
    └── [local dev] ─────────────────┤
                                     ▼
                          QueryPipeline (pipeline/query.py)
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
              FAISS dense         BM25 sparse    (optional)
              retrieval           retrieval       HyDE
                    │                │
                    └───────┬────────┘
                            ▼
                     RRF Fusion (hybrid)
                            │
                            ▼
                   Cross-encoder Re-ranker
                            │
                            ▼
                     Group by Song
                            │
                            ▼
              Retrieval-confidence gate ──► below threshold? ──► abstain (no LLM call)
                            │ sufficient
                            ▼
                  LLM Synthesis (Claude / Llama 3)
                            │
                            ▼
              Citation verification (chunk_index ↔ cited text)
                            │
                            ▼
                   Structured JSON Response
```

## Key Results (from Evaluation)

Technique grid at `top_k=5` (`all-MiniLM-L6-v2`), on the final, corrected corpus — see `Documentation/phase_6_evaluation.md`:

| Configuration | MRR (k=5) | Hit Rate (k=5) | MRR (k=20) | Hit Rate (k=20) |
|---|---|---|---|---|
| Baseline (FAISS only) | 0.603 | 0.693 | 0.620 | 0.827 |
| + Re-ranker | 0.700 | 0.733 | 0.811 | 0.867 |
| **+ Hybrid + Re-ranker** | **0.798** | **0.840** | — | — |
| HyDE + anything | 0.06–0.22 | — | — | — |

**Best overall** (312-run combined sweep across embedding models, `top_k`, fetch multiplier, and reranker model): `text-embedding-3-small` + Hybrid + Re-ranker at `top_k=20`, `fetch_k_multiplier=8x` → **MRR 0.897, Hit Rate 97.3%**

Other findings worth knowing before reusing a technique from this repo elsewhere: HyDE consistently *hurts* retrieval here (queries already contain lyric-like text, so the generated "hypothetical document" adds noise instead of bridging a semantic gap), and contextual embeddings (metadata prepended before encoding) help `all-MiniLM-L6-v2` but consistently hurt `bge-small-en-v1.5` — the benefit is model-dependent, not universal. A RAG-vs-pure-LLM comparison also confirmed retrieval isn't redundant here even though the corpus (Billboard lyrics) is public: Claude/GPT-4o-mini without retrieval only name the correct song 13–15% of the time, versus 80–89% with retrieval.

## Technology Stack

| Layer | Technology |
|---|---|
| Embedding (local) | sentence-transformers (`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`) |
| Embedding (cloud) | OpenAI `text-embedding-3-small` |
| Vector index | FAISS `IndexFlatIP` (exact cosine search) |
| Keyword index | BM25Okapi (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranking | cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM synthesis | Anthropic Claude Haiku / Ollama Llama 3 |
| Structured output | Pydantic + Claude tool_use |
| Frontend | Streamlit |
| Message broker | RabbitMQ (AMQP via pika) |
| Containerization | Docker + Docker Compose |
| Evaluation | DeepEval (LLM-as-judge), custom metrics |

## Documentation

Detailed documentation for each component is in the `Documentation/` folder:

| Document | Contents |
|---|---|
| [architecture.md](Documentation/architecture.md) | Full system design, data flow, design decisions |
| [api_reference.md](Documentation/api_reference.md) | All classes, methods, CLI arguments |
| [worker_rabbitmq.md](Documentation/worker_rabbitmq.md) | RabbitMQ worker integration |
| [phase_1_ingestion.md](Documentation/phase_1_ingestion.md) | Genius API ingestion |
| [phase_1.5_cleaning.md](Documentation/phase_1.5_cleaning.md) | Lyrics cleaning and validation |
| [phase_2_chunking.md](Documentation/phase_2_chunking.md) | Section-aware chunking strategy |
| [phase_3_embedding_indexing.md](Documentation/phase_3_embedding_indexing.md) | Embedding models and indices |
| [phase_4_query_pipeline.md](Documentation/phase_4_query_pipeline.md) | Retrieval and LLM synthesis |
| [phase_5_streamlit.md](Documentation/phase_5_streamlit.md) | Streamlit frontend |
| [phase_6_evaluation.md](Documentation/phase_6_evaluation.md) | Evaluation methodology and results |
| [phase_7_deployment.md](Documentation/phase_7_deployment.md) | Docker containerization |
| [project_summary.md](Documentation/project_summary.md) | Complete project summary with all findings |
