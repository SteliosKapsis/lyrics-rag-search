# Phase 7 — Deployment

## Overview

Docker containerization of the Streamlit frontend for reproducible deployment and thesis demonstration. The container runs the app with all pipeline dependencies, while keeping secrets and raw data outside the image.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage build: installs runtime deps, copies app + pipeline + data |
| `.dockerignore` | Excludes raw data, collection scripts, notebooks, .env, cache |
| `docker-compose.yml` | Streamlit service + optional Ollama, volume mounts, env_file |
| `requirements-docker.txt` | Slim runtime-only dependencies (no notebook/collection deps) |
| `.env.example` | Template for API keys |

## Prerequisites

1. **Docker** and **Docker Compose** installed
2. **Pipeline data generated** — you must have already run the full pipeline (Phases 1-3) to produce:
   - `data/processed/faiss.index`
   - `data/processed/metadata.json`
   - `data/processed/bm25.pkl`
3. **API key** — copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`

## Usage

### Build and run (Anthropic only)

```bash
docker compose up --build
```

The app opens at http://localhost:8501.

### Run with Ollama (local LLM)

```bash
# Start both the app and Ollama
docker compose --profile ollama up --build

# First time only — pull the Llama 3 model inside the container:
docker compose exec ollama ollama pull llama3
```

Then select "Ollama (Llama 3 local)" in the app sidebar.

### Stop

```bash
docker compose down
```

### Rebuild after re-embedding

If you re-run `pipeline/embedding.py` to regenerate `data/processed/`, the volume mount picks up the new files automatically — just restart:

```bash
docker compose restart app
```

No image rebuild needed because `data/processed/` is mounted as a volume.

## Architecture

```
Host machine                          Docker
─────────────                         ──────
.env (API keys) ──env_file──────────→ app container
                                        ├── Streamlit (port 8501)
data/processed/ ──volume mount (ro)──→  ├── FAISS index
                                        ├── BM25 index
                                        ├── metadata.json
                                        └── sentence-transformers model
                                             (cached in hf-cache volume)

                                      ollama container (optional)
                                        ├── Ollama server (port 11434)
                                        └── llama3 model
                                             (cached in ollama-models volume)
```

## Docker Image Details

### Base image: `python:3.11-slim`

- **Python 3.11** — latest stable version with full compatibility for faiss-cpu, sentence-transformers, and Streamlit
- **slim variant** (~150MB vs ~900MB full) — excludes compilers and build tools not needed at runtime
- **Debian Bookworm** — provides the glibc that faiss-cpu requires

### Multi-stage build

The Dockerfile uses a two-stage build:
1. **Builder stage** — installs pip packages (generates compiled wheels)
2. **Final stage** — copies only the installed packages and application code

This avoids shipping pip cache, wheel files, and build artifacts in the final image.

### Runtime-only dependencies

`requirements-docker.txt` includes only what the Streamlit app needs:
- `sentence-transformers`, `faiss-cpu`, `rank_bm25` — retrieval
- `anthropic`, `requests` — LLM backends
- `streamlit` — frontend
- `python-dotenv` — .env loading

Excluded from the Docker image:
- `lyricsgenius`, `rapidfuzz`, `langdetect` — collection/cleaning (Phase 1)
- `matplotlib`, `seaborn`, `pandas` — evaluation notebook (Phase 6)

### Image size optimization

- `--no-cache-dir` on pip install — saves ~50-100MB of cached wheels
- `.dockerignore` — prevents raw data (~130MB), notebooks, and cache from entering the build context
- Multi-stage build — only final packages copied, no build artifacts

## Volumes

| Volume | Purpose | Persistence |
|--------|---------|-------------|
| `./data/processed` (bind) | FAISS index, BM25 index, metadata — mounted read-only | Host filesystem |
| `hf-cache` (named) | HuggingFace model cache (~80-400MB depending on model) | Persists across restarts |
| `ollama-models` (named) | Ollama model files (~4GB for Llama 3) | Persists across restarts |

### HuggingFace model caching

sentence-transformers downloads the embedding model (`all-MiniLM-L6-v2`, ~80MB) and cross-encoder model (~80MB) on first use. The `hf-cache` volume persists these so they aren't re-downloaded on every container restart.

Alternative: bake models into the image during build (faster cold start, but larger image and harder to swap models). The volume approach is more flexible for experimentation.

## Gotchas

### faiss-cpu on slim images

`faiss-cpu` is a pre-compiled wheel that depends on `libgomp` (OpenMP) and `glibc`. The `python:3.11-slim` image (Debian Bookworm) includes both. If you switch to Alpine-based images, faiss-cpu will fail because Alpine uses musl instead of glibc.

### host.docker.internal

The `OLLAMA_HOST` environment variable is set to `http://host.docker.internal:11434` to reach Ollama running on the host machine.

| Platform | host.docker.internal | Notes |
|----------|---------------------|-------|
| **Windows** (Docker Desktop) | Works out of the box | |
| **Mac** (Docker Desktop) | Works out of the box | |
| **Linux** (Docker Engine) | Requires `extra_hosts` | Add to the app service in docker-compose.yml: `extra_hosts: ["host.docker.internal:host-gateway"]` |

If running Ollama inside Docker (via the `ollama` profile), you can change `OLLAMA_HOST` to `http://ollama:11434` to use Docker's internal DNS.

### sentence-transformers first-run download

On first container start, sentence-transformers will download the embedding model from HuggingFace. This takes 10-30 seconds and requires internet access. Subsequent starts use the cached version in the `hf-cache` volume.

If running in an air-gapped environment, pre-download the model and bake it into the image:
```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### .env file security

The `.env` file is loaded via `env_file` in docker-compose.yml and is **never** copied into the image (excluded by `.dockerignore`). The `ANTHROPIC_API_KEY` exists only as a runtime environment variable inside the container.

## Design Decisions

- **Volume mount over COPY for data/processed/** — the FAISS index changes when you re-embed (e.g., different model or chunk params). Mounting as a volume means you just restart the container instead of rebuilding the image.
- **Ollama as a separate profile** — most users will use Anthropic only. The Ollama service is opt-in via `--profile ollama` to avoid pulling a ~2GB image by default.
- **Read-only data mount** — `data/processed/` is mounted `:ro` because the container should never modify the index files.
- **Named volume for HF cache** — persists model downloads across container restarts without polluting the host filesystem.
