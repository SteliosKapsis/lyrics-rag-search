"""
Retrieval mechanics for the RAG query pipeline.

Owns query embedding, FAISS dense search, BM25 sparse search, RRF hybrid
fusion, and lazy cross-encoder reranker loading — separated from
QueryPipeline's orchestration/grouping/synthesis logic. Composed into
QueryPipeline the same way pipeline/llm_backend.py's LLMBackend already is.

This is a structural extraction only: every method here is a verbatim move
from pipeline/query.py, not a rewrite. Same FAISS/BM25/RRF logic, same
scoring, same tracing spans — QueryPipeline's public methods (.query(),
.query_stream(), etc.) and their observable behavior are unchanged.

Does not own loading: QueryPipeline still reads the FAISS index, metadata,
BM25 index, and embedding model/client in its own __init__ (so the
model-identity sidecar check keeps running before anything is handed off),
then passes the already-loaded objects in here.
"""

from collections import defaultdict

import numpy as np
from langfuse import get_client, observe


class Retriever:
    """Embeds queries and searches the FAISS/BM25 indexes for a QueryPipeline.
    Holds no LLM-calling or synthesis state of its own."""

    def __init__(self, index, metadata, bm25, embed_model, openai_embed_client, reranker_model: str):
        self.index = index
        self.metadata = metadata
        self.bm25 = bm25
        self.embed_model = embed_model
        self._openai_embed_client = openai_embed_client
        self.reranker_model = reranker_model
        self._reranker = None

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker on first use."""
        if self._reranker is None:
            from pipeline.reranker import Reranker
            self._reranker = Reranker(model_name=self.reranker_model)
        return self._reranker

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string, normalized for cosine similarity."""
        if self._openai_embed_client:
            response = self._openai_embed_client.embeddings.create(
                model="text-embedding-3-small", input=[text],
            )
            emb = np.array([response.data[0].embedding], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return emb / norms
        return self.embed_model.encode(
            [text], normalize_embeddings=True
        ).astype(np.float32)

    @observe(name="dense-retrieval")
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed query and retrieve top-k similar chunks from FAISS.

        Returns a list of dicts with a "score" field (FAISS cosine similarity).
        """
        q_emb = self._embed_text(query)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        # Candidate list only (title/artist/chunk_index/score) — not full chunk
        # text, to keep trace payloads lean; already bounded by top_k.
        get_client().update_current_span(
            input={"query": query, "top_k": top_k},
            output=[{"title": r["title"], "artist": r["artist"], "chunk_index": r["chunk_index"],
                     "score": r["score"]} for r in results],
        )
        return results

    @observe(name="sparse-retrieval")
    def retrieve_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top-k chunks using BM25 keyword search.

        Returns a list of dicts with a "bm25_score" field.
        """
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            if scores[idx] <= 0:
                break
            entry = dict(self.metadata[idx])
            entry["bm25_score"] = float(scores[idx])
            results.append(entry)

        get_client().update_current_span(
            input={"query": query, "top_k": top_k},
            output=[{"title": r["title"], "artist": r["artist"], "chunk_index": r["chunk_index"],
                     "bm25_score": r["bm25_score"]} for r in results],
        )
        return results

    @observe(name="hybrid-fusion")
    def retrieve_hybrid(
        self, query: str, top_k: int = 5, rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid retrieval: FAISS dense + BM25 sparse, merged with
        Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = Σ 1 / (k + rank_i(d))
        where k=60 is the standard constant.

        Why hybrid helps:
        - Dense (FAISS) captures semantic meaning — good for descriptive
          queries like "sad song about rain"
        - Sparse (BM25) captures exact keyword matches — good for direct
          lyric recall like "never gonna give you up"
        - Combining them covers both query types without needing to know
          which type the user is issuing.
        """
        fetch_k = top_k * 3

        # Get ranked lists from both methods
        dense_results = self.retrieve(query, top_k=fetch_k)
        sparse_results = self.retrieve_bm25(query, top_k=fetch_k)

        # Build RRF scores keyed by metadata index
        # We need a unique identifier — use (title, artist, chunk_index)
        def chunk_key(r):
            return (r["title"], r["artist"], r["chunk_index"])

        rrf_scores = defaultdict(float)
        chunk_data = {}

        for rank, r in enumerate(dense_results, 1):
            key = chunk_key(r)
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            if key not in chunk_data:
                chunk_data[key] = dict(r)
            chunk_data[key]["score"] = r["score"]  # FAISS score

        for rank, r in enumerate(sparse_results, 1):
            key = chunk_key(r)
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            if key not in chunk_data:
                chunk_data[key] = dict(r)
                chunk_data[key]["score"] = 0.0  # no FAISS score
            chunk_data[key]["bm25_score"] = r["bm25_score"]

        # Sort by RRF score and return top_k
        ranked_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        results = []
        for key in ranked_keys:
            entry = chunk_data[key]
            entry["rrf_score"] = rrf_scores[key]
            results.append(entry)

        get_client().update_current_span(
            input={"query": query, "top_k": top_k, "rrf_k": rrf_k},
            output=[{"title": r["title"], "artist": r["artist"], "chunk_index": r["chunk_index"],
                     "rrf_score": r["rrf_score"]} for r in results],
        )
        return results
