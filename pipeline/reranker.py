"""
Cross-encoder reranker for the RAG pipeline.
Re-scores retrieved chunks using a cross-encoder model that considers
the query and chunk text jointly, producing more accurate relevance scores
than the bi-encoder (FAISS) retrieval.

Usage as a module:
    from pipeline.reranker import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(query="sad song about rain", chunks=retrieved_chunks, top_k=5)

The cross-encoder model scores each (query, chunk) pair directly, which is
more accurate than bi-encoder cosine similarity but too slow for first-stage
retrieval over the full index. That's why we use it as a second stage:
    FAISS (fast, approximate) → Cross-encoder (slow, precise)
"""

import logging

from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class Reranker:
    """
    Cross-encoder reranker that re-scores retrieved chunks.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default — a lightweight
    cross-encoder trained on MS MARCO passage ranking data. It's small
    (~80MB) and fast enough for re-ranking top-k results in real time.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        log.info("Loading cross-encoder model: %s", model_name)
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-score and re-sort retrieved chunks using the cross-encoder.

        Args:
            query: The user's natural language query.
            chunks: List of chunk dicts from FAISS retrieval. Each must have
                    a "text" field. Other fields are passed through unchanged.
            top_k: If set, return only the top-k re-ranked results.
                   If None, return all chunks re-sorted.

        Returns:
            The same chunk dicts with an added "cross_encoder_score" field,
            sorted by cross-encoder score descending.
        """
        if not chunks:
            return chunks

        # Build (query, text) pairs for the cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["cross_encoder_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c["cross_encoder_score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked
