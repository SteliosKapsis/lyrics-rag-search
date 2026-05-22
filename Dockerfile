# ============================================================
# Lyrics RAG Search Assistant — Streamlit Docker Image
# ============================================================
# Base: python:3.11-slim
#   - 3.11 is the latest stable Python with full support for
#     faiss-cpu, sentence-transformers, and Streamlit
#   - slim variant (~150MB vs ~900MB full) excludes build tools
#     we don't need at runtime (gcc, make, etc.)
#   - Debian Bookworm base provides glibc needed by faiss-cpu
# ============================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install only runtime requirements (not notebooks/collection deps).
# A separate requirements-docker.txt avoids pulling lyricsgenius,
# langdetect, matplotlib, seaborn, etc. into the image.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt

# ============================================================
# Final stage — copy only what the app needs
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY worker/ ./worker/
COPY .streamlit/ ./.streamlit/
COPY .env.example ./.env.example

# Indexes (FAISS / BM25 / metadata) are mounted from the host at runtime
# via docker-compose.yml — see the `./data/processed:/app/data/processed:ro`
# bind mount. Not baked into the image to keep it slim and avoid stale data.

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config: disable usage stats, enable CORS for Docker,
# and set the server address to 0.0.0.0 so it's reachable from outside.
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# HuggingFace model cache — sentence-transformers downloads models
# on first use. Set cache dir inside the container.
ENV HF_HOME=/app/.cache/huggingface

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501"]
