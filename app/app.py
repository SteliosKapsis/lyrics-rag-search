"""
Streamlit frontend for the lyrics RAG search assistant.

Usage (from project root):
    .venv/Scripts/streamlit run app/app.py
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st

# Add project root to path so we can import pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.query import LLMResponse, QueryPipeline  # noqa: E402


# Map embedding model names to their index file suffixes
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "",                        # default files: faiss.index, metadata.json, bm25.pkl
    "BAAI/bge-small-en-v1.5": "_bge",              # suffixed files: faiss_bge.index, metadata_bge.json, bm25_bge.pkl
    "all-MiniLM-L6-v2 (no context)": "_noctx",     # non-contextual: faiss_noctx.index (built with --skip-contextual)
    "BAAI/bge-small-en-v1.5 (no context)": "_bge_noctx",
    "text-embedding-3-small": "_openai",           # OpenAI cloud: faiss_openai.index (built with --openai)
}

DATA_DIR = PROJECT_ROOT / "data" / "processed"

RABBITMQ_URL = os.getenv("RABBITMQ_URL")
_RABBIT_QUEUE = "rag.queries"


@st.cache_resource
def load_pipeline(
    embedding_model: str,
    llm_backend: str,
    llm_model: str | None,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    """Load the query pipeline once and cache it across reruns."""
    suffix = EMBEDDING_MODELS.get(embedding_model, "")
    return QueryPipeline(
        index_path=str(DATA_DIR / f"faiss{suffix}.index"),
        metadata_path=str(DATA_DIR / f"metadata{suffix}.json"),
        bm25_path=str(DATA_DIR / f"bm25{suffix}.pkl"),
        embedding_model=embedding_model,
        llm_backend=llm_backend,
        llm_model=llm_model,
        reranker_model=reranker_model,
    )


def build_youtube_url(artist: str, title: str) -> str:
    """Build a YouTube search URL for a song."""
    query = f"{artist} {title} official music video"
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"


def _query_via_rabbitmq(query: str, **kwargs) -> dict:
    """Publish query to the worker queue and block until result arrives (RPC pattern)."""
    import pika

    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    reply_queue = channel.queue_declare(queue="", exclusive=True).method.queue
    correlation_id = str(uuid.uuid4())
    response: dict = {}

    def on_response(ch, method, props, body):
        if props.correlation_id == correlation_id:
            response["data"] = json.loads(body)
            response["status"] = (props.headers or {}).get("status", "ok")

    channel.basic_consume(queue=reply_queue, on_message_callback=on_response, auto_ack=True)
    channel.basic_publish(
        exchange="",
        routing_key=_RABBIT_QUEUE,
        properties=pika.BasicProperties(
            reply_to=reply_queue,
            correlation_id=correlation_id,
            delivery_mode=2,  # persistent request
        ),
        body=json.dumps({"query": query, **kwargs}),
    )

    deadline = time.time() + 120
    while "data" not in response and time.time() < deadline:
        connection.process_data_events(time_limit=1)

    connection.close()

    if "data" not in response:
        raise TimeoutError("Worker did not respond within 120 seconds")
    if response.get("status") == "error":
        raise RuntimeError(response["data"].get("error", "Worker error"))

    return response["data"]


def main():
    st.set_page_config(
        page_title="Lyrics Search Assistant",
        page_icon="🎵",
        layout="wide",
    )

    st.title("Lyrics Search Assistant")
    st.caption("RAG-powered semantic search over song lyrics")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        # Detect which embedding models have indices available
        available_models = {
            name: suffix
            for name, suffix in EMBEDDING_MODELS.items()
            if (DATA_DIR / f"faiss{suffix}.index").exists()
        }

        if len(available_models) > 1:
            _model_keys = list(available_models.keys())
            embedding_model = st.selectbox(
                "Embedding model",
                options=_model_keys,
                index=(
                    _model_keys.index("text-embedding-3-small")
                    if "text-embedding-3-small" in _model_keys
                    else 0
                ),
                help="Choose which embedding model / index to query against",
            )
        elif available_models:
            embedding_model = next(iter(available_models))
        else:
            st.error("No FAISS indices found in data/processed/.")
            return

        top_k = st.slider("Top-k results", min_value=1, max_value=20, value=20)

        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Only show results above this FAISS similarity score",
        )

        use_hyde = st.checkbox(
            "Use HyDE",
            value=False,
            help="Generate a hypothetical lyric before retrieval (not recommended for lyrics search)",
        )

        use_hybrid = st.checkbox(
            "Use hybrid search",
            value=True,
            help="Combine FAISS dense retrieval with BM25 keyword search via RRF",
        )

        use_reranker = st.checkbox(
            "Use re-ranker",
            value=True,
            help="Re-score results with a cross-encoder for more accurate ranking (recommended)",
        )

        RERANKER_MODELS = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "BAAI/bge-reranker-base",
        ]
        reranker_model = st.selectbox(
            "Re-ranker model",
            options=RERANKER_MODELS,
            index=0,
            help="Cross-encoder model for re-ranking. Larger models are more accurate but slower.",
            disabled=not use_reranker,
        )

        llm_backend = st.radio(
            "LLM backend",
            options=["Anthropic (Claude Haiku)", "Ollama (Llama 3 local)"],
            index=0,
        )

        # Map display names to backend keys
        backend_key = "anthropic" if "Anthropic" in llm_backend else "ollama"
        llm_model = None  # use defaults per backend

    # --- Load pipeline ---
    try:
        pipeline = load_pipeline(embedding_model, backend_key, llm_model, reranker_model)
    except FileNotFoundError as e:
        st.error(
            f"Pipeline data not found: {e}\n\n"
            "Make sure you've run the embedding pipeline first:\n"
            "```\n.venv\\Scripts\\python pipeline\\embedding.py\n```"
        )
        return
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return

    # --- Query input ---
    query = st.text_input(
        "What song are you looking for?",
        placeholder="e.g., that sad song about leaving home in the rain",
    )

    if not query:
        st.info("Enter a query above to search for songs.")
        return

    # --- Run query (RabbitMQ worker path or direct streaming path) ---
    retrieval_data = None
    llm_response = None

    if RABBITMQ_URL:
        # ── Queue path: dispatch to worker, block until result ──────────
        with st.spinner("Processing via worker queue..."):
            try:
                result = _query_via_rabbitmq(
                    query,
                    embedding_model=embedding_model,
                    reranker_model=reranker_model,
                    top_k=top_k,
                    use_hyde=use_hyde,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                )
            except Exception as e:
                st.error(f"Worker queue error: {e}")
                return

        retrieval_data = {
            "retrieval_results": result.get("retrieval_results", []),
            "hyde_hypothesis": result.get("hyde_hypothesis"),
        }
        try:
            llm_response = LLMResponse.model_validate(result["llm_response"])
        except Exception:
            llm_response = None

    else:
        # ── Direct streaming path (local dev, no RabbitMQ) ──────────────
        try:
            stream = pipeline.query_stream(
                query,
                top_k=top_k,
                use_hyde=use_hyde,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
            )
            with st.spinner("Retrieving..."):
                retrieval_data = next(stream)
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg and backend_key == "ollama":
                st.error(
                    "Cannot connect to Ollama at localhost:11434.\n\n"
                    "Make sure Ollama is installed and running:\n"
                    "1. Install: https://ollama.com/download\n"
                    "2. Pull model: `ollama pull llama3`\n"
                    "3. Ollama should start automatically after install"
                )
            else:
                st.error(f"Query failed: {e}")
            return

        collected_json = ""
        try:
            for token in stream:
                collected_json += token
        except Exception as e:
            collected_json = f"[Error: Stream failed: {e}]"

        if not collected_json.startswith("[Error"):
            try:
                llm_response = LLMResponse.model_validate_json(collected_json)
            except Exception:
                try:
                    llm_response = LLMResponse.model_validate(json.loads(collected_json))
                except Exception:
                    pass

    # --- LLM Response display ---
    st.markdown("### Answer")

    with st.container(border=True):
        if llm_response:
            badge_colors = {"high": "green", "medium": "orange", "low": "red"}
            color = badge_colors.get(llm_response.confidence, "gray")
            st.markdown(f"**Confidence:** :{color}[{llm_response.confidence}]")
            st.markdown(llm_response.summary)
            if llm_response.matches:
                for match in llm_response.matches:
                    st.markdown(f"**{match.title}** by {match.artist}")
                    if match.relevant_lyric:
                        st.markdown(f"> {match.relevant_lyric}")
                    st.caption(match.explanation)
        else:
            st.caption("No structured response available.")

    # --- HyDE hypothesis ---
    if retrieval_data.get("hyde_hypothesis"):
        with st.expander("HyDE: Hypothetical lyric used for retrieval"):
            st.code(retrieval_data["hyde_hypothesis"], language=None)

    # --- Retrieved results ---
    st.markdown("### Retrieved Songs")

    # Filter by similarity threshold
    filtered_songs = [
        song for song in retrieval_data["retrieval_results"]
        if song["best_score"] >= similarity_threshold
    ]

    if not filtered_songs:
        st.warning(
            f"No results above the similarity threshold ({similarity_threshold:.2f}). "
            "Try lowering the threshold in the sidebar."
        )
        return

    for song in filtered_songs:
        youtube_url = build_youtube_url(song["artist"], song["title"])

        with st.container(border=True):
            # Song header
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{song['title']}** by {song['artist']}")
                st.caption(
                    f"Album: {song['album'] or 'Unknown'} | "
                    f"Released: {song['release_date'] or 'Unknown'}"
                )
            with col2:
                st.link_button("Find on YouTube", youtube_url)

            # Lyric excerpts
            for chunk in song["chunks"]:
                # Build score label
                score_parts = [f"FAISS: {chunk['score']:.3f}"]
                if "bm25_score" in chunk:
                    score_parts.append(f"BM25: {chunk['bm25_score']:.1f}")
                if "cross_encoder_score" in chunk:
                    score_parts.append(f"Reranker: {chunk['cross_encoder_score']:.3f}")

                st.caption(" | ".join(score_parts))
                st.markdown(
                    f"> {chunk['text'][:500].replace(chr(10), chr(10) + '> ')}",
                )


if __name__ == "__main__":
    main()
