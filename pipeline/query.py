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
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

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
    title: str
    artist: str
    album: str | None = None
    release_date: str | None = None
    relevant_lyric: str       # most relevant excerpt
    explanation: str           # why this matches the query


class LLMResponse(BaseModel):
    """Structured response from the LLM synthesis step."""
    matches: list[SongMatch]  # ranked by relevance
    confidence: Literal["high", "medium", "low"]  # constrained to 3 values
    summary: str              # 1-2 sentence answer to the query


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
1. Identify the most likely matching song(s) from the retrieved excerpts.
2. Show the most relevant lyric excerpt(s) that match the user's description.
3. For each match, provide: song title, artist, album, and release date.
4. If multiple songs could match, rank them by relevance and explain why.
5. If the similarity scores are low (below 0.3) or the excerpts don't clearly match \
the user's description, explicitly acknowledge the uncertainty — say something like \
"I'm not fully confident in this match" rather than guessing.

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
            parts.append(f"  [{score_str}]")
            parts.append(f"  {chunk['text']}")
            parts.append("")
        parts.append("")

    return "\n".join(parts)


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

        # Reranker loaded lazily on first use
        self._reranker = None
        self.reranker_model = reranker_model

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
                return self._call_anthropic(HYDE_PROMPT, f"User is looking for: {query}")
            else:
                return self._call_ollama(HYDE_PROMPT, f"User is looking for: {query}")
        except Exception as e:
            # Content filtering or API errors — fall back to raw query
            log.warning("HyDE hypothesis generation failed: %s. Falling back to raw query.", e)
            return query

    def _call_anthropic(self, system: str, user_message: str) -> str:
        """Call Claude via the Anthropic API (plain text, used for HyDE)."""
        if not self.anthropic_client:
            return "[Error: ANTHROPIC_API_KEY not configured]"

        response = self.anthropic_client.messages.create(
            model=self.llm_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _call_anthropic_structured(self, system: str, user_message: str) -> LLMResponse:
        """
        Call Claude with tool_use to get structured JSON output.

        Uses tool_use: defines a tool whose input_schema matches the Pydantic
        model. Claude returns structured JSON as the tool call arguments, which
        we parse and validate with Pydantic.
        """
        if not self.anthropic_client:
            return LLMResponse(
                matches=[], confidence="low",
                summary="Error: ANTHROPIC_API_KEY not configured",
            )

        response = self.anthropic_client.messages.create(
            model=self.llm_model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            tools=[RESPONSE_TOOL],
            tool_choice={"type": "tool", "name": "format_response"},
        )

        # Extract tool call arguments
        for block in response.content:
            if block.type == "tool_use":
                return LLMResponse.model_validate(block.input)

        # Fallback if no tool_use block found
        return LLMResponse(
            matches=[], confidence="low",
            summary="Failed to parse structured response from LLM.",
        )

    def _get_ollama_url(self) -> str:
        """Get the Ollama API base URL from env or default."""
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return f"{host}/api/chat"

    def _call_ollama(self, system: str, user_message: str) -> str:
        """
        Call a local LLM via the Ollama REST API (plain text, used for HyDE).

        Ollama is a tool for running open-source LLMs locally. It downloads,
        manages, and serves models via a simple REST API on localhost:11434.

        Install: https://ollama.com/download
        Pull Llama 3: ollama pull llama3
        It runs automatically after install — no manual start needed.
        """
        url = self._get_ollama_url()
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.ConnectionError:
            return (
                "[Error: Cannot connect to Ollama. "
                "Make sure Ollama is installed and running. "
                "Install: https://ollama.com/download | "
                "Pull model: ollama pull llama3]"
            )
        except requests.Timeout:
            return "[Error: Ollama request timed out after 120s]"
        except Exception as e:
            return f"[Error: Ollama request failed: {e}]"

    def _call_ollama_structured(self, system: str, user_message: str) -> LLMResponse:
        """
        Call Ollama with the format parameter to get structured JSON output.

        Uses the format parameter with the Pydantic model's JSON schema,
        which forces Ollama to return conforming JSON.
        """
        url = self._get_ollama_url()
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": LLMResponse.model_json_schema(),
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return LLMResponse.model_validate_json(content)
        except requests.ConnectionError:
            return LLMResponse(
                matches=[], confidence="low",
                summary="Error: Cannot connect to Ollama. Install: https://ollama.com/download",
            )
        except Exception as e:
            return LLMResponse(
                matches=[], confidence="low",
                summary=f"Error: Ollama request failed: {e}",
            )

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

        return results

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

        return results

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
            return self._call_anthropic_structured(SYSTEM_PROMPT, user_message)
        else:
            return self._call_ollama_structured(SYSTEM_PROMPT, user_message)

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

        return raw_results

    def query(
        self,
        query: str,
        top_k: int = 20,
        use_hyde: bool = False,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 5,
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

        Returns:
            {
                "query": str,
                "retrieval_results": list[dict],  # grouped by song
                "llm_response": str,
                "hyde_hypothesis": str | None,
            }
        """
        log.info(
            "Query: '%s' (top_k=%d, hyde=%s, hybrid=%s, reranker=%s, backend=%s)",
            query, top_k, use_hyde, use_hybrid, use_reranker, self.llm_backend,
        )

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

        # Step 5: Synthesize with LLM (structured output)
        llm_response = self.synthesize(query, grouped)
        log.info("LLM confidence: %s, matches: %d", llm_response.confidence, len(llm_response.matches))

        return {
            "query": query,
            "retrieval_results": grouped,
            "llm_response": llm_response,
            "hyde_hypothesis": hyde_hypothesis,
        }

    def query_stream(
        self,
        query: str,
        top_k: int = 20,
        use_hyde: bool = False,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 5,
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
        """
        log.info(
            "Stream query: '%s' (top_k=%d, hyde=%s, hybrid=%s, reranker=%s, backend=%s)",
            query, top_k, use_hyde, use_hybrid, use_reranker, self.llm_backend,
        )

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

        # Step 5: Stream LLM synthesis
        context = build_context(grouped)
        user_message = (
            f"User query: \"{query}\"\n\n"
            f"Retrieved lyrics from database:\n\n{context}\n\n"
            f"Based on these retrieved excerpts, identify the song(s) the user is looking for."
        )

        if self.llm_backend == "anthropic":
            yield from self._stream_anthropic(SYSTEM_PROMPT, user_message)
        else:
            yield from self._stream_ollama(SYSTEM_PROMPT, user_message)

    def _stream_anthropic(self, system: str, user_message: str) -> Generator[str, None, None]:
        """
        Stream tokens from Claude via the Anthropic API using tool_use.

        Streams the JSON being built as tool input. The caller collects all
        tokens and parses the final JSON into LLMResponse after stream completes.
        """
        if not self.anthropic_client:
            yield "[Error: ANTHROPIC_API_KEY not configured]"
            return

        try:
            with self.anthropic_client.messages.stream(
                model=self.llm_model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                tools=[RESPONSE_TOOL],
                tool_choice={"type": "tool", "name": "format_response"},
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                        yield event.delta.partial_json
        except anthropic.APIStatusError as e:
            error_msg = e.body.get("error", {}).get("message", str(e)) if isinstance(e.body, dict) else str(e)
            if "content filtering" in error_msg.lower():
                yield "[Error: Response blocked by content filter. Try rephrasing your query.]"
            else:
                yield f"[Error: Anthropic API error: {error_msg}]"
        except anthropic.APIConnectionError:
            yield "[Error: Cannot connect to Anthropic API. Check your network.]"
        except anthropic.RateLimitError:
            yield "[Error: Rate limit exceeded. Please wait and retry.]"
        except anthropic.APITimeoutError:
            yield "[Error: Request timed out. Please retry.]"

    def _stream_ollama(self, system: str, user_message: str) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama REST API with structured format.

        Streams the JSON being built token by token. The caller collects all
        tokens and parses the final JSON into LLMResponse after stream completes.
        """
        url = self._get_ollama_url()
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": True,
            "format": LLMResponse.model_json_schema(),
        }

        try:
            resp = requests.post(url, json=payload, timeout=120, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except requests.ConnectionError:
            yield (
                "[Error: Cannot connect to Ollama. "
                "Install: https://ollama.com/download | Pull model: ollama pull llama3]"
            )
        except Exception as e:
            yield f"[Error: Ollama streaming failed: {e}]"


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


if __name__ == "__main__":
    main()
