"""
RAG query pipeline — LangChain version.

Replaces the hand-rolled Anthropic / FAISS / BM25 code with:

  Embeddings   langchain_huggingface.HuggingFaceEmbeddings
               langchain_openai.OpenAIEmbeddings
  Vector store langchain_community.vectorstores.FAISS
  LLMs         langchain_anthropic.ChatAnthropic
               langchain_community.chat_models.ChatOllama
  Structured   llm.with_structured_output(LLMResponse)   (non-streaming)
  Prompts      langchain_core.prompts.ChatPromptTemplate
  Reranker     CrossEncoderReranker.compress_documents()
  Tracing      langfuse.callback.CallbackHandler          (auto-traces all LangChain calls)

BM25 hybrid retrieval still uses the pickled BM25Okapi index + metadata.json
(LangChain's BM25Retriever is in-memory only; keeping the pickle avoids
rebuilding 175 k documents on every worker startup).

The public interface of QueryPipeline is unchanged so app.py / worker.py
require only a path-format update (faiss_lc/ folders instead of .index files).

Usage:
    from pipeline.query import QueryPipeline
    pipeline = QueryPipeline()
    result = pipeline.query("sad song about letting someone go")
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

import numpy as np
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Pydantic response schemas  (unchanged from custom branch)
# ---------------------------------------------------------------------------

class SongMatch(BaseModel):
    title: str
    artist: str
    album: str | None = None
    release_date: str | None = None
    relevant_lyric: str
    explanation: str


class LLMResponse(BaseModel):
    matches: list[SongMatch]
    confidence: Literal["high", "medium", "low"]
    summary: str


# ---------------------------------------------------------------------------
# Prompts  (unchanged)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a lyrics search assistant. The user will describe a song they're looking for, \
and you will be given retrieved lyric excerpts from a database along with similarity scores.

Your task:
1. Identify the most likely matching song(s) from the retrieved excerpts.
2. Show the most relevant lyric excerpt(s) that match the user's description.
3. For each match, provide: song title, artist, album, and release date.
4. If multiple songs could match, rank them by relevance and explain why.
5. If the similarity scores are low (below 0.3) or the excerpts don't clearly match \
the user's description, explicitly acknowledge the uncertainty.

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

# JSON schema string injected into the streaming prompt so the LLM outputs
# parseable JSON without requiring tool_use (which can't stream partial tokens).
_RESPONSE_SCHEMA_STR = json.dumps(LLMResponse.model_json_schema(), indent=2)
# {{ and }} are literal braces in LangChain prompt templates — escape the
# JSON schema so its {…} don't get parsed as template variable placeholders.
_RESPONSE_SCHEMA_ESCAPED = _RESPONSE_SCHEMA_STR.replace("{", "{{").replace("}", "}}")

STREAMING_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n\nYou MUST respond with a single valid JSON object — no markdown, no prose "
    "outside the JSON — that strictly matches this schema:\n"
    + _RESPONSE_SCHEMA_ESCAPED
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_context(grouped_results: list[dict]) -> str:
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


# ---------------------------------------------------------------------------
# QueryPipeline
# ---------------------------------------------------------------------------

class QueryPipeline:
    """
    End-to-end RAG query pipeline backed by LangChain components.

    Dense retrieval  — LangChain FAISS vectorstore (loaded from faiss_lc/ folder)
    Sparse retrieval — BM25Okapi (loaded from bm25.pkl, same format as custom branch)
    Hybrid           — manual Reciprocal Rank Fusion (same algorithm as custom branch)
    Reranker         — LangChain CrossEncoderReranker
    LLM synthesis    — ChatAnthropic / ChatOllama with structured output (non-streaming)
                       plain .stream() for streaming path
    Tracing          — LangfuseCallbackHandler (injected per query via config=)
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
        processed = project_root / "data" / "processed"

        # index_path is now a folder (LangChain FAISS format)
        self.index_path = index_path or str(processed / "faiss_lc")
        self.metadata_path = metadata_path or str(processed / "metadata.json")
        self.bm25_path = bm25_path or str(processed / "bm25.pkl")
        self.embedding_model_name = embedding_model
        self.llm_backend = llm_backend
        self.reranker_model = reranker_model

        if llm_model:
            self.llm_model = llm_model
        elif llm_backend == "ollama":
            self.llm_model = "llama3"
        else:
            self.llm_model = "claude-haiku-4-5-20251001"

        # ------------------------------------------------------------------
        # 1. LangChain embeddings function
        # ------------------------------------------------------------------
        if embedding_model == "text-embedding-3-small":
            from langchain_openai import OpenAIEmbeddings
            log.info("Using OpenAI embeddings: %s", embedding_model)
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            log.info("Loading HuggingFace embeddings: %s", embedding_model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )

        # ------------------------------------------------------------------
        # 2. LangChain FAISS vectorstore
        # ------------------------------------------------------------------
        from langchain_community.vectorstores import FAISS

        log.info("Loading FAISS vectorstore from %s/", self.index_path)
        self._vectorstore = FAISS.load_local(
            self.index_path,
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

        # ------------------------------------------------------------------
        # 3. BM25 index + metadata (positionally aligned)
        # ------------------------------------------------------------------
        self._bm25 = None
        self._metadata: list[dict] = []

        if Path(self.bm25_path).exists():
            log.info("Loading BM25 index from %s", self.bm25_path)
            with open(self.bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
        else:
            log.warning("BM25 index not found at %s — hybrid retrieval disabled", self.bm25_path)

        if Path(self.metadata_path).exists():
            with open(self.metadata_path, encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            log.warning("metadata.json not found at %s — BM25 lookup disabled", self.metadata_path)

        # ------------------------------------------------------------------
        # 4. LangChain LLM
        # ------------------------------------------------------------------
        if llm_backend == "anthropic":
            from langchain_anthropic import ChatAnthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                log.warning("ANTHROPIC_API_KEY not set — LLM synthesis will fail")
            self._llm = ChatAnthropic(
                model=self.llm_model,
                max_tokens=2048,
                api_key=api_key or "missing",
            )
        else:
            from langchain_community.chat_models import ChatOllama
            self._llm = ChatOllama(
                model=self.llm_model,
                base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            )

        # Structured-output LLM (used for non-streaming synthesis)
        self._structured_llm = self._llm.with_structured_output(LLMResponse)

        # Synthesis prompt
        from langchain_core.prompts import ChatPromptTemplate
        self._synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",
             'User query: "{query}"\n\n'
             "Retrieved lyrics from database:\n\n{context}\n\n"
             "Based on these retrieved excerpts, identify the song(s) the user is looking for."),
        ])
        # Streaming synthesis uses a plain JSON-in-prompt approach
        self._stream_prompt = ChatPromptTemplate.from_messages([
            ("system", STREAMING_SYSTEM_PROMPT),
            ("human",
             'User query: "{query}"\n\n'
             "Retrieved lyrics from database:\n\n{context}\n\n"
             "Identify the song(s) and return ONLY valid JSON."),
        ])
        # HyDE prompt
        self._hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", HYDE_PROMPT),
            ("human", "User is looking for: {query}"),
        ])

        # Reranker loaded lazily on first use
        self._reranker = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_langfuse_handler(self, session_id: str | None):
        """Create a per-query LangfuseCallbackHandler for automatic tracing."""
        try:
            from langfuse.langchain import CallbackHandler  # langfuse v3
        except ImportError:
            from langfuse.callback import CallbackHandler  # langfuse v2
        kwargs = {
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        }
        if session_id:
            kwargs["session_id"] = session_id
        return CallbackHandler(**kwargs)

    def _get_reranker(self):
        if self._reranker is None:
            from pipeline.reranker import build_reranker
            self._reranker = build_reranker(self.reranker_model, top_n=1)
        return self._reranker

    def _doc_to_chunk(self, doc, score: float = 0.0) -> dict:
        """Convert a LangChain Document returned by FAISS to a chunk dict."""
        meta = doc.metadata
        chunk = {
            "text": meta.get("original_text", doc.page_content),
            "title": meta.get("title", ""),
            "artist": meta.get("artist", ""),
            "album": meta.get("album"),
            "release_date": meta.get("release_date"),
            "chunk_index": meta.get("chunk_index", 0),
            "score": score,
        }
        if "relevance_score" in meta:
            chunk["cross_encoder_score"] = meta["relevance_score"]
        return chunk

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Dense retrieval via LangChain FAISS vectorstore."""
        results = self._vectorstore.similarity_search_with_score(query, k=top_k)
        return [self._doc_to_chunk(doc, float(score)) for doc, score in results]

    def retrieve_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """Sparse BM25 keyword retrieval using the pickled BM25Okapi index."""
        if self._bm25 is None or not self._metadata:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            if scores[idx] <= 0:
                break
            entry = dict(self._metadata[idx])
            entry["bm25_score"] = float(scores[idx])
            entry.setdefault("score", 0.0)
            results.append(entry)
        return results

    def retrieve_hybrid(
        self, query: str, top_k: int = 5, rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid retrieval: FAISS dense + BM25 sparse merged with RRF.

        EnsembleRetriever (LangChain's built-in RRF combiner) requires both
        retrievers to return Documents with identical page_content for
        deduplication to work correctly.  Because the FAISS Documents store
        contextual text while BM25 aligns on raw lyrics, we keep the manual
        RRF implementation from the custom branch — the algorithm is unchanged.
        """
        fetch_k = top_k * 3

        dense_results = self.retrieve(query, top_k=fetch_k)
        sparse_results = self.retrieve_bm25(query, top_k=fetch_k)

        def chunk_key(r):
            return (r["title"], r["artist"], r["chunk_index"])

        rrf_scores: dict = defaultdict(float)
        chunk_data: dict = {}

        for rank, r in enumerate(dense_results, 1):
            key = chunk_key(r)
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            if key not in chunk_data:
                chunk_data[key] = dict(r)
            chunk_data[key]["score"] = r["score"]

        for rank, r in enumerate(sparse_results, 1):
            key = chunk_key(r)
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            if key not in chunk_data:
                chunk_data[key] = dict(r)
                chunk_data[key]["score"] = 0.0
            chunk_data[key]["bm25_score"] = r["bm25_score"]

        ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
        results = []
        for key in ranked_keys:
            entry = chunk_data[key]
            entry["rrf_score"] = rrf_scores[key]
            results.append(entry)
        return results

    def group_by_song(self, results: list[dict]) -> list[dict]:
        """Group retrieved chunks by (title, artist) and rank by best score."""
        groups: dict = defaultdict(lambda: {
            "title": "", "artist": "", "album": None,
            "release_date": None, "best_score": 0.0, "chunks": [],
        })

        for r in results:
            key = (r["title"], r["artist"])
            g = groups[key]
            g["title"] = r["title"]
            g["artist"] = r["artist"]
            g["album"] = r.get("album")
            g["release_date"] = r.get("release_date")

            ranking_score = r.get("cross_encoder_score", r.get("rrf_score", r["score"]))
            g["best_score"] = max(g["best_score"], ranking_score)

            chunk_data = {"text": r["text"], "score": r["score"], "chunk_index": r["chunk_index"]}
            for field in ("bm25_score", "rrf_score", "cross_encoder_score"):
                if field in r:
                    chunk_data[field] = r[field]
            g["chunks"].append(chunk_data)

        return sorted(groups.values(), key=lambda g: g["best_score"], reverse=True)

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank_chunks(
        self, query: str, chunks: list[dict], top_k: int, config: dict,
    ) -> list[dict]:
        """
        Rerank chunk dicts using LangChain's CrossEncoderReranker.

        We wrap dicts as Documents (so the LangChain compressor can score them),
        then unwrap back to dicts with the relevance_score attached.
        """
        from langchain_core.documents import Document

        reranker = self._get_reranker()
        # Override top_n for this specific call
        reranker.top_n = top_k

        docs = [Document(page_content=c["text"], metadata=c) for c in chunks]
        reranked = reranker.compress_documents(docs, query, callbacks=config.get("callbacks"))

        result = []
        for doc in reranked:
            chunk = dict(doc.metadata)
            chunk["cross_encoder_score"] = doc.metadata.get("relevance_score", 0.0)
            result.append(chunk)
        return result

    # ------------------------------------------------------------------
    # Retrieval step (shared by query / query_stream)
    # ------------------------------------------------------------------

    def _retrieve_step(
        self,
        query: str,
        retrieval_query: str,
        top_k: int,
        use_hybrid: bool,
        use_reranker: bool,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 3,
        config: dict | None = None,
    ) -> list[dict]:
        config = config or {}
        fetch_k = top_k * fetch_k_multiplier if use_reranker else top_k

        if use_hybrid and self._bm25 is not None:
            log.info("Hybrid retrieval (FAISS + BM25 with RRF)...")
            raw_results = self.retrieve_hybrid(retrieval_query, top_k=fetch_k, rrf_k=rrf_k)
        else:
            raw_results = self.retrieve(retrieval_query, top_k=fetch_k)

        log.info("Retrieved %d chunks", len(raw_results))

        if use_reranker and raw_results:
            log.info("Re-ranking with LangChain CrossEncoderReranker...")
            raw_results = self._rerank_chunks(query, raw_results, top_k, config)
            log.info("Re-ranked to top %d", len(raw_results))

        return raw_results

    # ------------------------------------------------------------------
    # HyDE
    # ------------------------------------------------------------------

    def _generate_hyde_hypothesis(self, query: str, config: dict) -> str:
        """
        Generate a hypothetical lyric passage using the LLM for HyDE retrieval.

        Uses a plain ChatPromptTemplate + LLM chain via LCEL so that the call
        is automatically traced by the Langfuse callback handler.
        """
        try:
            chain = self._hyde_prompt | self._llm
            result = chain.invoke({"query": query}, config=config)
            return result.content
        except Exception as e:
            log.warning("HyDE hypothesis generation failed: %s. Falling back to raw query.", e)
            return query

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        All LangChain calls are automatically traced by the LangfuseCallbackHandler
        injected via config= — no manual @observe decorators needed.

        Returns the same dict structure as the custom branch:
            {query, retrieval_results, llm_response, hyde_hypothesis}
        """
        langfuse_handler = self._get_langfuse_handler(session_id)
        config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "embedding_model": self.embedding_model_name,
                "llm_model": self.llm_model,
                "llm_backend": self.llm_backend,
                "use_hyde": use_hyde,
                "use_hybrid": use_hybrid,
                "use_reranker": use_reranker,
                "top_k": top_k,
            },
        }

        log.info(
            "Query: '%s' (top_k=%d, hyde=%s, hybrid=%s, reranker=%s, backend=%s)",
            query, top_k, use_hyde, use_hybrid, use_reranker, self.llm_backend,
        )

        # Step 1: HyDE
        hyde_hypothesis = None
        retrieval_query = query
        if use_hyde:
            log.info("Generating HyDE hypothesis...")
            hyde_hypothesis = self._generate_hyde_hypothesis(query, config)
            retrieval_query = hyde_hypothesis
            log.info("HyDE: %s", hyde_hypothesis[:100])

        # Step 2 + 3: Retrieve + rerank
        raw_results = self._retrieve_step(
            query, retrieval_query, top_k, use_hybrid, use_reranker,
            rrf_k=rrf_k, fetch_k_multiplier=fetch_k_multiplier, config=config,
        )

        # Step 4: Group by song
        grouped = self.group_by_song(raw_results)
        log.info("Grouped into %d unique song(s)", len(grouped))

        # Step 5: LLM synthesis via LCEL chain with structured output
        context = build_context(grouped)
        synthesis_chain = self._synthesis_prompt | self._structured_llm
        llm_response: LLMResponse = synthesis_chain.invoke(
            {"query": query, "context": context}, config=config,
        )
        log.info("LLM confidence: %s, matches: %d", llm_response.confidence, len(llm_response.matches))

        langfuse_handler.flush()
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
        session_id: str | None = None,
    ) -> Generator[dict | str, None, None]:
        """
        Streaming RAG pipeline — same yield contract as the custom branch:

        1. First yield: dict {retrieval_results, hyde_hypothesis}
        2. Subsequent yields: str JSON token fragments from the LLM

        The LLM is called with a JSON-format system prompt (no tool_use) so
        that .stream() yields simple text tokens that app.py can accumulate
        and parse into LLMResponse at the end.
        """
        langfuse_handler = self._get_langfuse_handler(session_id)
        config = {"callbacks": [langfuse_handler]}

        log.info(
            "Stream query: '%s' (top_k=%d, hyde=%s, hybrid=%s, reranker=%s, backend=%s)",
            query, top_k, use_hyde, use_hybrid, use_reranker, self.llm_backend,
        )

        # Step 1: HyDE
        hyde_hypothesis = None
        retrieval_query = query
        if use_hyde:
            hyde_hypothesis = self._generate_hyde_hypothesis(query, config)
            retrieval_query = hyde_hypothesis

        # Step 2 + 3: Retrieve + rerank
        raw_results = self._retrieve_step(
            query, retrieval_query, top_k, use_hybrid, use_reranker,
            rrf_k=rrf_k, fetch_k_multiplier=fetch_k_multiplier, config=config,
        )

        # Step 4: Group by song — yield immediately so UI can display while LLM runs
        grouped = self.group_by_song(raw_results)
        yield {"retrieval_results": grouped, "hyde_hypothesis": hyde_hypothesis}

        # Step 5: Stream LLM synthesis
        context = build_context(grouped)
        stream_chain = self._stream_prompt | self._llm
        try:
            for chunk in stream_chain.stream(
                {"query": query, "context": context}, config=config,
            ):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    yield token
        except Exception as e:
            yield f"[Error: LLM streaming failed: {e}]"
        finally:
            langfuse_handler.flush()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def print_results(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f'  Query: "{result["query"]}"')
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
            print(f'    Lyric: "{match.relevant_lyric[:150]}"')
        print(f"    Why: {match.explanation}")
        print()

    print("--- Retrieved Chunks ---\n")
    for song in result["retrieval_results"]:
        print(
            f"  {song['title']} by {song['artist']} "
            f"[best score: {song['best_score']:.3f}]"
        )
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

    parser = argparse.ArgumentParser(description="Query the lyrics RAG pipeline (LangChain)")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--use-hyde", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    parser.add_argument("--llm-backend", choices=["anthropic", "ollama"], default="anthropic")
    parser.add_argument("--llm-model", default=None)
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
