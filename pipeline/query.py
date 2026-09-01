"""
RAG query pipeline for lyrics search.
Supports standard, hybrid (FAISS + BM25), and HyDE retrieval, optional
cross-encoder re-ranking, streaming responses, and two LLM backends
(Anthropic Claude / Ollama Llama 3).

Usage (from project root):
    .venv/Scripts/python pipeline/query.py "that sad song about leaving home"

    # With HyDE retrieval:
    .venv/Scripts/python pipeline/query.py "upbeat dance song" --use-hyde

    # With hybrid retrieval disabled (FAISS only):
    .venv/Scripts/python pipeline/query.py "never gonna give you up" --no-hybrid

    # With cross-encoder re-ranking:
    .venv/Scripts/python pipeline/query.py "love song" --use-reranker

    # Using local Ollama/Llama 3:
    .venv/Scripts/python pipeline/query.py "rock anthem" --llm-backend ollama

Can also be imported as a module:
    from pipeline.query import QueryPipeline
    pipeline = QueryPipeline()
    result = pipeline.query("sad song about letting someone go", use_hyde=True)

Inputs:  data/processed/faiss.index + data/processed/metadata.json + data/processed/bm25.pkl
"""

import argparse
import difflib
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Generator, Literal

import anthropic
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from pydantic import BaseModel, Field, model_validator
from sentence_transformers import SentenceTransformer

from pipeline.llm_backend import LLMBackend

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Suppress noisy logs from libraries
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

class SongMatch(BaseModel):
    """A single song match from the LLM synthesis."""
    title: str = Field(..., description="The song's title, exactly as given in the retrieved excerpt.")
    artist: str = Field(..., description="The artist, exactly as given in the retrieved excerpt.")
    album: str | None = Field(None, description="The album, if given in the retrieved excerpt.")
    release_date: str | None = Field(None, description="The release date, if given in the retrieved excerpt.")
    relevant_lyric: str = Field(
        ...,
        description="The most relevant excerpt, quoted verbatim (or near-verbatim) from the "
        "lyric excerpt text — not paraphrased — so it can be checked against the source chunk.",
    )
    chunk_index: int = Field(
        ...,
        description="The chunk_index shown in brackets next to the excerpt that relevant_lyric "
        "was drawn from. Must be one of the chunk_index values actually shown for this song.",
    )
    explanation: str = Field(..., description="Why this song matches the user's query.")


class LLMResponse(BaseModel):
    """Structured response from the LLM synthesis step."""
    matches: list[SongMatch] = Field(..., description="Song matches, ranked by relevance (most relevant first).")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Overall confidence in these matches, given the retrieved excerpts.",
    )
    summary: str = Field(..., description="A 1-2 sentence answer to the user's query.")
    abstained: bool = Field(
        False,
        description="True if none of the retrieved excerpts plausibly match the user's query. "
        "This is the CORRECT and EXPECTED choice when nothing fits — not a fallback to avoid. "
        "When true, matches should be empty and abstain_reason must explain why.",
    )
    abstain_reason: str | None = Field(
        None, description="Required when abstained is true: a short explanation of why no excerpt matched.",
    )

    @model_validator(mode="after")
    def _check_abstention_consistency(self) -> "LLMResponse":
        if self.abstained and not self.abstain_reason:
            raise ValueError("abstain_reason is required when abstained=True")
        return self


# Tool definition for Anthropic structured output via tool_use
RESPONSE_TOOL = {
    "name": "format_response",
    "description": "Format the lyrics search results into a structured response.",
    "input_schema": LLMResponse.model_json_schema(),
}


SYSTEM_PROMPT = """\
You are a lyrics search assistant. The user will describe a song they're looking for, \
and you will be given retrieved lyric excerpts from a database along with similarity scores.

Your task:
1. Identify the most likely matching song(s) from the retrieved excerpts — but ONLY if at \
least one excerpt plausibly matches the user's description. Set abstained=true and matches=[] \
if none of the excerpts plausibly match. Abstaining is the CORRECT and EXPECTED response in \
that case, not a failure or a fallback — you will not be judged worse for abstaining honestly \
than for guessing. Guessing wrong is the actual failure mode to avoid.
2. If you do find a match, show the most relevant lyric excerpt(s) that match the user's \
description, quoted verbatim (or as close to verbatim as possible) from the excerpt text — \
not paraphrased. Cite the chunk_index shown in brackets next to the excerpt you drew it from.
3. For each match, provide: song title, artist, album, and release date.
4. If multiple songs could match, rank them by relevance and explain why.
5. If the similarity scores are low (below 0.3) or the excerpts don't clearly match the \
user's description, prefer abstaining (abstained=true, with abstain_reason explaining why) \
over including a low-confidence guess in matches.

Keep your response concise and well-structured. Do not fabricate lyrics or metadata \
that aren't in the provided excerpts.\
"""

HYDE_PROMPT = """\
You are helping with a music search system. The user is looking for a song based on a \
description. Write a short thematic passage (3-5 sentences) describing what a song \
matching this description would sound like: its emotional tone, key themes, imagery, \
and the feelings it evokes. Do not write song lyrics or verse lines — write in plain \
prose, like a music critic describing the song's content and atmosphere.\
"""


def build_context(grouped_results: list[dict]) -> str:
    """
    Build the context string to send to the LLM.
    Includes both FAISS and cross-encoder scores when available.
    """
    parts = []
    for i, song in enumerate(grouped_results, 1):
        parts.append(f"--- Match {i} (best similarity: {song['best_score']:.3f}) ---")
        parts.append(f"Title: {song['title']}")
        parts.append(f"Artist: {song['artist']}")
        parts.append(f"Album: {song['album'] or 'Unknown'}")
        parts.append(f"Release Date: {song['release_date'] or 'Unknown'}")
        parts.append("Lyric Excerpts:")
        for chunk in song["chunks"]:
            score_str = f"FAISS: {chunk['score']:.3f}"
            if "cross_encoder_score" in chunk:
                score_str += f", Reranker: {chunk['cross_encoder_score']:.3f}"
            parts.append(f"  [{score_str}, chunk_index: {chunk['chunk_index']}]")
            parts.append(f"  {chunk['text']}")
            parts.append("")
        parts.append("")

    return "\n".join(parts)


def verify_citations(llm_response: "LLMResponse", grouped_results: list[dict]) -> list[dict]:
    """
    Best-effort check that each SongMatch's cited chunk_index actually exists
    among that song's retrieved chunks, and that relevant_lyric is plausibly
    drawn from that chunk's real text (fuzzy substring match via stdlib
    difflib — deliberately not rapidfuzz, which is excluded from the Docker
    image this function runs inside via worker.py's query()/query_stream()
    call path; difflib needs no extra dependency).

    This is citation LINKING, not citation VERIFICATION by entailment — it
    only checks relevant_lyric against its cited chunk, not the rest of the
    response (summary, explanation). A paraphrased quote can legitimately
    score low here even when the citation is otherwise correct; treat this
    as a cheap fabrication signal, not proof of grounding.

    Returns a list of {title, artist, chunk_index, match_score, verified} —
    one entry per SongMatch in llm_response.matches, in the same order.
    """
    # Index grouped_results by (title, artist) -> {chunk_index: text}
    chunks_by_song: dict[tuple[str, str], dict[int, str]] = {}
    for song in grouped_results:
        key = (song["title"], song["artist"])
        chunks_by_song[key] = {c["chunk_index"]: c["text"] for c in song["chunks"]}

    checks = []
    for match in llm_response.matches:
        song_chunks = chunks_by_song.get((match.title, match.artist), {})
        chunk_text = song_chunks.get(match.chunk_index)

        if chunk_text is None:
            # Cited chunk_index doesn't exist among this song's retrieved chunks
            # (out of range, or the LLM invented one) — never raise, just flag it.
            checks.append({
                "title": match.title, "artist": match.artist,
                "chunk_index": match.chunk_index, "match_score": 0.0, "verified": False,
            })
            continue

        score = difflib.SequenceMatcher(None, match.relevant_lyric.lower(), chunk_text.lower()).ratio()
        checks.append({
            "title": match.title, "artist": match.artist,
            "chunk_index": match.chunk_index, "match_score": round(score, 3), "verified": score >= 0.5,
        })

    for check in checks:
        if not check["verified"]:
            log.warning(
                "Citation could not be verified: '%s' by %s, chunk_index=%s (match_score=%.3f)",
                check["title"], check["artist"], check["chunk_index"], check["match_score"],
            )

    return checks


def classify_retrieval_outcome(raw_results: list[dict], threshold: float) -> tuple[str, float]:
    """
    Classify a retrieval as "empty" (nothing retrieved), "low_relevance"
    (retrieved something, but the best score is below `threshold`), or
    "sufficient" (a plausible match exists) — the distinction query() and
    query_stream() use to decide whether to skip the LLM call and abstain,
    per file 05's own framing: an empty retrieval is a *retrieval* failure,
    a populated-but-unhelpful one is a *coverage* failure, and collapsing
    the two into one generic refusal destroys the signal needed to fix
    the system, so they stay distinguishable in logs/traces.

    Compares whichever score is most informative for the candidates at hand:
    cross-encoder score if reranking ran, else RRF score, else raw FAISS
    score — matching the same priority group_by_song() already uses to rank
    songs. NOTE: these are different scales (cross-encoder logits are
    unbounded, RRF scores are typically ~0.01-0.03, FAISS cosine is 0-1) —
    `threshold` is only meaningful in the same scale as whichever of these
    happens to be present, so a single shared threshold is a real
    simplification, not a calibrated cutoff. See QueryPipeline's
    abstain_score_threshold docstring.

    Returns (outcome, best_score).
    """
    if not raw_results:
        return "empty", 0.0
    best = max(r.get("cross_encoder_score", r.get("rrf_score", r["score"])) for r in raw_results)
    return ("sufficient" if best >= threshold else "low_relevance"), best


class QueryPipeline:
    """
    End-to-end RAG query pipeline with support for:
    - Standard, hybrid (FAISS + BM25), and HyDE retrieval
    - Optional cross-encoder re-ranking
    - Streaming LLM responses
    - Anthropic (Claude) and Ollama (Llama 3) LLM backends

    Loads the FAISS index, BM25 index, metadata, and embedding model once,
    then handles multiple queries efficiently.
    """

    def __init__(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
        bm25_path: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str | None = None,
        llm_backend: str = "anthropic",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        abstain_score_threshold: float | None = None,
    ):
        project_root = Path(__file__).resolve().parent.parent
        self.index_path = index_path or str(project_root / "data" / "processed" / "faiss.index")
        self.metadata_path = metadata_path or str(project_root / "data" / "processed" / "metadata.json")
        self.bm25_path = bm25_path or str(project_root / "data" / "processed" / "bm25.pkl")
        self.embedding_model_name = embedding_model
        self.llm_backend = llm_backend

        # Set default LLM model per backend
        if llm_model:
            self.llm_model = llm_model
        elif llm_backend == "ollama":
            self.llm_model = "llama3"
        else:
            self.llm_model = "claude-haiku-4-5-20251001"

        # Load FAISS index
        log.info("Loading FAISS index from %s", self.index_path)
        self.index = faiss.read_index(self.index_path)

        # Assert the index was actually built with this embedding model. Query
        # and index embeddings from different models occupy incomparable vector
        # spaces — mismatched retrieval degrades silently (no exception, just
        # worse results), so this must fail loudly rather than proceed on a
        # guess. See pipeline/embedding.py::save_index_info() for the sidecar.
        self._assert_index_model_matches(self.index_path, self.embedding_model_name)

        # Load metadata
        with open(self.metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load BM25 index (optional — hybrid retrieval degrades gracefully)
        self.bm25 = None
        if Path(self.bm25_path).exists():
            log.info("Loading BM25 index from %s", self.bm25_path)
            with open(self.bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            log.warning("BM25 index not found at %s — hybrid retrieval disabled", self.bm25_path)

        # Load embedding model (local or OpenAI)
        self._openai_embed_client = None
        if self.embedding_model_name == "text-embedding-3-small":
            log.info("Using OpenAI embedding model: %s", self.embedding_model_name)
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                log.warning("OPENAI_API_KEY not set — query embedding will fail")
            else:
                self._openai_embed_client = openai.OpenAI(api_key=api_key)
            self.embed_model = None
        else:
            log.info("Loading embedding model: %s", self.embedding_model_name)
            self.embed_model = SentenceTransformer(self.embedding_model_name)

        # Init Anthropic client (only if using anthropic backend)
        self.anthropic_client = None
        if self.llm_backend == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                log.warning("ANTHROPIC_API_KEY not set — LLM synthesis will fail")
            else:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)

        # Owns all direct Anthropic/Ollama API calls (structured, plain-text,
        # streaming) — separated from retrieval/grouping logic so this class
        # doesn't grow yet another concern. See pipeline/llm_backend.py.
        self._backend = LLMBackend(
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            anthropic_client=self.anthropic_client,
        )

        # Reranker loaded lazily on first use
        self._reranker = None
        self.reranker_model = reranker_model

        # Score threshold below which query()/query_stream() skip the LLM call
        # and abstain instead of guessing. PLACEHOLDER VALUE — 0.3 is not tuned
        # against real data. Real tuning requires running the golden set's
        # unanswerable-question subset (notebooks/evaluation.ipynb) through
        # this pipeline and sweeping the threshold against the resulting
        # abstention rate; that has not been done. Override via the
        # RAG_ABSTAIN_SCORE_THRESHOLD env var or the constructor arg.
        if abstain_score_threshold is not None:
            self.abstain_score_threshold = abstain_score_threshold
        else:
            self.abstain_score_threshold = float(os.environ.get("RAG_ABSTAIN_SCORE_THRESHOLD", 0.3))

    def _assert_index_model_matches(self, index_path: str, embedding_model: str) -> None:
        """
        Read the model-identity sidecar written by pipeline/embedding.py's
        save_index_info() and fail loudly if it's missing or doesn't match the
        embedding model this pipeline is about to use to embed queries.

        Query embeddings and index embeddings from different models occupy
        incomparable vector spaces — this is the single highest-severity
        failure mode in the pipeline, because a mismatch does not throw on
        its own; it just quietly returns worse results.
        """
        info_path = Path(index_path).with_name(Path(index_path).stem + "_info.json")
        if not info_path.exists():
            raise ValueError(
                f"No model-identity sidecar found for index '{index_path}' "
                f"(expected '{info_path}'). This index predates the model-identity "
                f"check and cannot be safely trusted to match embedding_model="
                f"'{embedding_model}'. Re-run pipeline/embedding.py with the matching "
                f"--model / --openai / --skip-contextual flags to regenerate it."
            )

        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        recorded_model = info.get("embedding_model")
        if recorded_model != embedding_model:
            raise ValueError(
                f"Embedding model mismatch for index '{index_path}': it was built "
                f"with '{recorded_model}' but this QueryPipeline is configured for "
                f"'{embedding_model}'. Query embeddings and index embeddings must "
                f"come from the same model, or retrieval silently degrades. Pass "
                f"embedding_model='{recorded_model}' or rebuild the index for "
                f"'{embedding_model}'."
            )

        recorded_dim = info.get("embedding_dim")
        if recorded_dim is not None and recorded_dim != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch for index '{index_path}': the sidecar "
                f"records {recorded_dim} dimensions but the loaded FAISS index has "
                f"{self.index.d}. The sidecar may be stale or hand-edited — "
                f"re-run pipeline/embedding.py to regenerate the index and sidecar together."
            )

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

    def _generate_hyde_hypothesis(self, query: str) -> str:
        """
        Generate a hypothetical lyric using the LLM for HyDE retrieval.

        HyDE (Hypothetical Document Embeddings) intuition:
        Instead of embedding the user's natural language query (which lives in
        "question space"), we ask the LLM to generate what the answer might
        look like (a hypothetical lyric). This hypothetical lyric lives in
        "document space" — the same space as the indexed chunks — so it
        produces better similarity matches.

        For lyrics specifically, this helps because:
        - A query like "sad song about rain" is very different linguistically
          from actual lyrics about rain and sadness
        - A hypothetical lyric like "raindrops fall on empty streets / tears I
          cannot hide" is much closer to how real lyrics are written
        - The embedding of the hypothesis will be nearer to relevant chunks
          in the vector space
        """
        try:
            if self.llm_backend == "anthropic":
                return self._backend.call_anthropic(HYDE_PROMPT, f"User is looking for: {query}")
            else:
                return self._backend.call_ollama(HYDE_PROMPT, f"User is looking for: {query}")
        except Exception as e:
            # Content filtering or API errors — fall back to raw query
            log.warning("HyDE hypothesis generation failed: %s. Falling back to raw query.", e)
            return query

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

    def group_by_song(self, results: list[dict]) -> list[dict]:
        """
        Group retrieved chunks by song (title + artist).
        Preserves both FAISS and cross-encoder scores when available.
        Songs are ranked by their best score.
        """
        groups = defaultdict(lambda: {
            "title": "", "artist": "", "album": None,
            "release_date": None, "best_score": 0.0, "chunks": [],
        })

        for r in results:
            key = (r["title"], r["artist"])
            group = groups[key]
            group["title"] = r["title"]
            group["artist"] = r["artist"]
            group["album"] = r.get("album")
            group["release_date"] = r.get("release_date")

            # Use cross-encoder > RRF > FAISS in priority order
            ranking_score = r.get("cross_encoder_score", r.get("rrf_score", r["score"]))
            group["best_score"] = max(group["best_score"], ranking_score)

            chunk_data = {
                "text": r["text"],
                "score": r["score"],
                "chunk_index": r["chunk_index"],
            }
            if "bm25_score" in r:
                chunk_data["bm25_score"] = r["bm25_score"]
            if "rrf_score" in r:
                chunk_data["rrf_score"] = r["rrf_score"]
            if "cross_encoder_score" in r:
                chunk_data["cross_encoder_score"] = r["cross_encoder_score"]

            group["chunks"].append(chunk_data)

        ranked = sorted(groups.values(), key=lambda g: g["best_score"], reverse=True)
        return ranked

    def _build_user_message(self, query: str, grouped_results: list[dict]) -> str:
        """Build the user message for LLM synthesis."""
        context = build_context(grouped_results)
        return (
            f"User query: \"{query}\"\n\n"
            f"Retrieved lyrics from database:\n\n{context}\n\n"
            f"Based on these retrieved excerpts, identify the song(s) the user is looking for."
        )

    def synthesize(self, query: str, grouped_results: list[dict]) -> LLMResponse:
        """
        Send the query and retrieved context to the LLM for structured synthesis.

        Returns a validated LLMResponse Pydantic object.
        """
        user_message = self._build_user_message(query, grouped_results)

        if self.llm_backend == "anthropic":
            return self._backend.call_anthropic_structured(SYSTEM_PROMPT, user_message)
        else:
            return self._backend.call_ollama_structured(SYSTEM_PROMPT, user_message)

    @observe(name="retrieval")
    def _retrieve_step(
        self,
        query: str,
        retrieval_query: str,
        top_k: int,
        use_hybrid: bool,
        use_reranker: bool,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 3,
    ) -> list[dict]:
        """
        Shared retrieval logic for query() and query_stream().
        Returns raw results (before grouping).
        """
        # Fetch more than top_k if reranking, so the reranker has more to work with
        fetch_k = top_k * fetch_k_multiplier if use_reranker else top_k
        get_client().update_current_span(
            input={"query": query, "retrieval_query": retrieval_query},
            metadata={"use_hybrid": use_hybrid, "use_reranker": use_reranker, "top_k": top_k, "fetch_k": fetch_k},
        )

        if use_hybrid and self.bm25 is not None:
            log.info("Hybrid retrieval (FAISS + BM25 with RRF)...")
            raw_results = self.retrieve_hybrid(retrieval_query, top_k=fetch_k, rrf_k=rrf_k)
        else:
            raw_results = self.retrieve(retrieval_query, top_k=fetch_k)

        log.info("Retrieved %d chunks", len(raw_results))

        # Optionally re-rank with cross-encoder
        if use_reranker and raw_results:
            log.info("Re-ranking with cross-encoder...")
            reranker = self._get_reranker()
            # Rerank using the ORIGINAL query (not the HyDE hypothesis),
            # because the cross-encoder should judge relevance to what the
            # user actually asked for.
            raw_results = reranker.rerank(query, raw_results, top_k=top_k)
            log.info("Re-ranked to top %d", len(raw_results))

        # Attach the retrieval outcome as span metadata (empty / low_relevance /
        # sufficient — see classify_retrieval_outcome) purely for trace
        # visibility. This does NOT change what's returned — the actual
        # abstain-or-continue decision is made by the caller (query()/
        # query_stream()), which re-classifies after this returns.
        outcome, best_score = classify_retrieval_outcome(raw_results, self.abstain_score_threshold)
        get_client().update_current_span(
            metadata={
                "retrieval_outcome": outcome,
                "best_relevance_score": best_score,
                "abstain_threshold": self.abstain_score_threshold,
            },
        )

        return raw_results

    @observe(name="rag-query")
    def query(
        self,
        query: str,
        top_k: int = 20,
        use_hyde: bool = False,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 5,
        session_id: str | None = None,
    ) -> dict:
        """
        Full RAG pipeline: [HyDE] → retrieve [hybrid] → [rerank] → group → synthesize.

        Args:
            query: Natural language query.
            top_k: Number of chunks to retrieve.
            use_hyde: If True, generate a hypothetical lyric and embed that instead.
            use_hybrid: If True, combine FAISS + BM25 with RRF (default True).
            use_reranker: If True, re-rank retrieved chunks with cross-encoder.
            rrf_k: RRF fusion constant for hybrid retrieval (default 60).
            fetch_k_multiplier: How many more candidates to fetch before re-ranking (default 3).
            session_id: Optional session identifier forwarded to Langfuse for trace correlation.

        Returns:
            {
                "query": str,
                "retrieval_results": list[dict],  # grouped by song
                "llm_response": str,
                "hyde_hypothesis": str | None,
            }
        """
        lf = get_client()
        log.info(
            "Query: '%s' (top_k=%d, hyde=%s, hybrid=%s, reranker=%s, backend=%s)",
            query, top_k, use_hyde, use_hybrid, use_reranker, self.llm_backend,
        )

        _pa_kwargs: dict = {"metadata": {
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model,
            "llm_backend": self.llm_backend,
            "use_hyde": use_hyde,
            "use_hybrid": use_hybrid,
            "use_reranker": use_reranker,
            "top_k": top_k,
        }}
        if session_id:
            _pa_kwargs["session_id"] = session_id

        with propagate_attributes(**_pa_kwargs):
            lf.set_current_trace_io(input=query)

            # Step 1: Optionally generate HyDE hypothesis
            hyde_hypothesis = None
            retrieval_query = query

            if use_hyde:
                log.info("Generating HyDE hypothesis...")
                hyde_hypothesis = self._generate_hyde_hypothesis(query)
                retrieval_query = hyde_hypothesis
                log.info("HyDE hypothesis: %s", hyde_hypothesis[:100] + "...")

            # Step 2 + 3: Retrieve and optionally re-rank
            raw_results = self._retrieve_step(
                query, retrieval_query, top_k, use_hybrid, use_reranker,
                rrf_k=rrf_k, fetch_k_multiplier=fetch_k_multiplier,
            )

            # Step 4: Group by song
            grouped = self.group_by_song(raw_results)
            log.info("Grouped into %d unique song(s)", len(grouped))

            # Step 5: Synthesize with LLM (structured output) — unless retrieval
            # didn't turn up anything plausible, in which case skip the LLM call
            # entirely and abstain. Cheapest and safest option: zero chance of
            # hallucination if the LLM is never asked to guess in the first place.
            outcome, best_score = classify_retrieval_outcome(raw_results, self.abstain_score_threshold)
            log.info(
                "Retrieval outcome: %s (best_score=%.3f, threshold=%.3f, n_chunks=%d)",
                outcome, best_score, self.abstain_score_threshold, len(raw_results),
            )
            if outcome != "sufficient":
                reason = (
                    "No lyrics were retrieved for this query." if outcome == "empty"
                    else f"Retrieved excerpts scored below the relevance threshold "
                         f"({best_score:.3f} < {self.abstain_score_threshold})."
                )
                llm_response = LLMResponse(
                    matches=[], confidence="low",
                    summary="I couldn't confidently identify a matching song.",
                    abstained=True, abstain_reason=reason,
                )
            else:
                llm_response = self.synthesize(query, grouped)
            log.info("LLM confidence: %s, matches: %d", llm_response.confidence, len(llm_response.matches))

            # Best-effort check that each match's chunk_index citation is real and its
            # relevant_lyric plausibly matches that chunk's actual text (see verify_citations
            # docstring — a fabrication signal, not proof of grounding).
            citation_checks = verify_citations(llm_response, grouped)

            lf.set_current_trace_io(
                output={
                    "confidence": llm_response.confidence,
                    "num_matches": len(llm_response.matches),
                    "abstained": llm_response.abstained,
                },
            )
            return {
                "query": query,
                "retrieval_results": grouped,
                "llm_response": llm_response,
                "hyde_hypothesis": hyde_hypothesis,
                "citation_checks": citation_checks,
            }

    @observe(name="rag-query-stream")
    def query_stream(
        self,
        query: str,
        top_k: int = 20,
        use_hyde: bool = False,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 5,
        session_id: str | None = None,
    ) -> Generator[dict | str, None, None]:
        """
        Streaming RAG pipeline. Yields:
        1. First yield: a dict with retrieval_results + hyde_hypothesis
           (so the caller can display results immediately)
        2. Subsequent yields: individual string tokens from the LLM response

        Usage:
            stream = pipeline.query_stream("sad song about rain")
            first = next(stream)  # dict with retrieval_results
            for token in stream:  # str tokens from LLM
                print(token, end="")

        Note: unlike query(), this does not compute citation_checks — the tokens
        here are streamed straight from the LLM and never assembled into an
        LLMResponse inside this generator. Once the caller has collected and
        parsed the full response, call verify_citations(llm_response,
        first["retrieval_results"]) directly to get the same check.
        """
        lf = get_client()
        log.info(
            "Stream query: '%s' (top_k=%d, hyde=%s, hybrid=%s, reranker=%s, backend=%s)",
            query, top_k, use_hyde, use_hybrid, use_reranker, self.llm_backend,
        )

        _pa_kwargs: dict = {"metadata": {
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model,
            "llm_backend": self.llm_backend,
            "use_hyde": use_hyde,
            "use_hybrid": use_hybrid,
            "use_reranker": use_reranker,
            "top_k": top_k,
        }}
        if session_id:
            _pa_kwargs["session_id"] = session_id

        with propagate_attributes(**_pa_kwargs):
            lf.set_current_trace_io(input=query)

            # Step 1: Optionally generate HyDE hypothesis
            hyde_hypothesis = None
            retrieval_query = query

            if use_hyde:
                log.info("Generating HyDE hypothesis...")
                hyde_hypothesis = self._generate_hyde_hypothesis(query)
                retrieval_query = hyde_hypothesis

            # Step 2 + 3: Retrieve and optionally re-rank
            raw_results = self._retrieve_step(
                query, retrieval_query, top_k, use_hybrid, use_reranker,
                rrf_k=rrf_k, fetch_k_multiplier=fetch_k_multiplier,
            )

            # Step 4: Group by song
            grouped = self.group_by_song(raw_results)

            # Yield retrieval results immediately so the UI can display them
            yield {
                "retrieval_results": grouped,
                "hyde_hypothesis": hyde_hypothesis,
            }

            # Step 5: Stream LLM synthesis — unless retrieval didn't turn up
            # anything plausible, in which case skip the LLM call and yield an
            # abstain response directly (same logic as query(), see its comments).
            outcome, best_score = classify_retrieval_outcome(raw_results, self.abstain_score_threshold)
            log.info(
                "Retrieval outcome: %s (best_score=%.3f, threshold=%.3f, n_chunks=%d)",
                outcome, best_score, self.abstain_score_threshold, len(raw_results),
            )
            if outcome != "sufficient":
                reason = (
                    "No lyrics were retrieved for this query." if outcome == "empty"
                    else f"Retrieved excerpts scored below the relevance threshold "
                         f"({best_score:.3f} < {self.abstain_score_threshold})."
                )
                abstain_response = LLMResponse(
                    matches=[], confidence="low",
                    summary="I couldn't confidently identify a matching song.",
                    abstained=True, abstain_reason=reason,
                )
                # Yield the whole thing as one chunk — app.py's existing collect-then-parse
                # loop (concatenate all yielded tokens, then LLMResponse.model_validate_json())
                # handles this identically to a real multi-token stream, no caller change needed.
                yield abstain_response.model_dump_json()
                return

            context = build_context(grouped)
            user_message = (
                f"User query: \"{query}\"\n\n"
                f"Retrieved lyrics from database:\n\n{context}\n\n"
                f"Based on these retrieved excerpts, identify the song(s) the user is looking for."
            )

            if self.llm_backend == "anthropic":
                yield from self._backend.stream_anthropic(SYSTEM_PROMPT, user_message)
            else:
                yield from self._backend.stream_ollama(SYSTEM_PROMPT, user_message)


def print_results(result: dict) -> None:
    """Pretty-print query results for CLI usage."""
    print("\n" + "=" * 60)
    print(f"  Query: \"{result['query']}\"")
    print("=" * 60)

    if result.get("hyde_hypothesis"):
        print("\n--- HyDE Hypothesis ---\n")
        print(result["hyde_hypothesis"])

    llm = result["llm_response"]
    print(f"\n--- LLM Response [confidence: {llm.confidence}] ---\n")
    print(f"Summary: {llm.summary}\n")
    for match in llm.matches:
        print(f"  {match.title} by {match.artist}")
        if match.relevant_lyric:
            print(f"    Lyric: \"{match.relevant_lyric[:150]}\"")
        print(f"    Why: {match.explanation}")
        print()

    print("--- Retrieved Chunks ---\n")
    for song in result["retrieval_results"]:
        print(f"  {song['title']} by {song['artist']} "
              f"[best score: {song['best_score']:.3f}]")
        print(f"  Album: {song['album'] or 'Unknown'} | "
              f"Released: {song['release_date'] or 'Unknown'}")
        for chunk in song["chunks"]:
            score_parts = [f"FAISS: {chunk['score']:.3f}"]
            if "cross_encoder_score" in chunk:
                score_parts.append(f"Reranker: {chunk['cross_encoder_score']:.3f}")
            preview = chunk["text"][:200].replace("\n", "\n    ")
            print(f"    [{', '.join(score_parts)}] {preview}...")
        print()


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Query the lyrics RAG pipeline")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model (must match index-time model)")
    parser.add_argument("--use-hyde", action="store_true", help="Use HyDE (Hypothetical Document Embeddings) for retrieval")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid retrieval (use FAISS only)")
    parser.add_argument("--use-reranker", action="store_true", help="Re-rank results with cross-encoder")
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for re-ranking (default: cross-encoder/ms-marco-MiniLM-L-6-v2). "
             "Alternatives: cross-encoder/ms-marco-MiniLM-L-12-v2, BAAI/bge-reranker-base"
    )
    parser.add_argument("--llm-backend", choices=["anthropic", "ollama"], default="anthropic", help="LLM backend for synthesis")
    parser.add_argument("--llm-model", default=None, help="LLM model name (default: claude-haiku-4-5-20251001 for anthropic, llama3 for ollama)")
    args = parser.parse_args()

    pipeline = QueryPipeline(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        llm_backend=args.llm_backend,
        reranker_model=args.reranker_model,
    )
    result = pipeline.query(
        args.query,
        top_k=args.top_k,
        use_hyde=args.use_hyde,
        use_hybrid=not args.no_hybrid,
        use_reranker=args.use_reranker,
    )
    print_results(result)

    # Flush Langfuse background queue before the script exits.
    # Without this, async traces are lost when the process terminates.
    get_client().flush()


if __name__ == "__main__":
    main()
