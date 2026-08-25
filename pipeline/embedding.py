"""
Embedding and FAISS indexing script for RAG pipeline.
Embeds lyric chunks using sentence-transformers and stores them in a FAISS index.
Also builds a BM25 keyword index for hybrid retrieval.

Usage (from project root):
    .venv/Scripts/python pipeline/embedding.py

Or with custom parameters:
    .venv/Scripts/python pipeline/embedding.py --model all-mpnet-base-v2 --batch-size 64

Inputs:  data/processed/chunks.jsonl
Outputs: data/processed/faiss.index + data/processed/metadata.json + data/processed/bm25.pkl
         + data/processed/faiss_info.json (model-identity sidecar, asserted against
           at query time by pipeline/query.py::QueryPipeline — see save_index_info())

If interrupted mid-run, a partial-embedding checkpoint (*_embed_checkpoint.npz)
is left next to the index output; re-running the same command resumes instead
of re-embedding from scratch. It's deleted automatically on a successful run.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_chunks(jsonl_path: str) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info("Loaded %d chunks from %s", len(chunks), jsonl_path)
    return chunks


def embedding_fingerprint(texts: list[str], model_label: str, contextual: bool) -> str:
    """
    Cheap fingerprint identifying a specific (texts, model, contextual) combo,
    used to check whether an on-disk embedding checkpoint actually matches the
    run trying to resume from it (not a stale leftover from a different config).
    Not a full corpus hash — just enough to catch a mismatch, not tamper-proof.
    """
    sample = f"{len(texts)}|{model_label}|{contextual}|{texts[0][:200] if texts else ''}|{texts[-1][:200] if texts else ''}"
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()[:16]


def checkpoint_path_for(index_output_path: str) -> Path:
    """
    Derive the embedding-progress checkpoint path from an (already-suffixed)
    FAISS index output path, so each of the model/openai/skip-contextual
    variants gets its own independent checkpoint file.
    """
    p = Path(index_output_path)
    return p.with_name(p.stem + "_embed_checkpoint.npz")


def _load_embed_checkpoint(checkpoint_path: Path, fingerprint: str) -> tuple[np.ndarray, int] | None:
    """
    Load a partial-embedding checkpoint if it exists and matches `fingerprint`.
    Returns (embeddings_so_far, next_index) or None if there's nothing usable
    to resume from (missing, corrupt, or from a different run's config) — a
    failed/corrupt checkpoint is always treated as "start fresh", never a crash.
    """
    if not checkpoint_path.exists():
        return None
    try:
        data = np.load(checkpoint_path, allow_pickle=False)
        saved_fingerprint = str(data["fingerprint"].item())
        if saved_fingerprint != fingerprint:
            log.warning(
                "Checkpoint %s exists but doesn't match this run's input "
                "(different model/corpus/contextual setting) — ignoring it, starting fresh.",
                checkpoint_path,
            )
            return None
        embeddings = data["embeddings"]
        next_index = int(data["next_index"].item())
        log.info("Resuming from checkpoint %s: %d texts already embedded.", checkpoint_path, next_index)
        return embeddings, next_index
    except Exception as e:
        log.warning(
            "Checkpoint %s exists but could not be loaded (%s) — treating as no checkpoint, starting fresh.",
            checkpoint_path, e,
        )
        return None


def _save_embed_checkpoint(checkpoint_path: Path, embeddings: np.ndarray, next_index: int, fingerprint: str) -> None:
    """
    Atomically write embedding progress to disk (write to a temp file, then
    os.replace) so a crash mid-write never leaves a corrupt checkpoint that
    silently fails to load on the next attempt.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.stem + ".tmp.npz")
    np.savez(tmp_path, embeddings=embeddings, next_index=np.array(next_index), fingerprint=np.array(fingerprint))
    os.replace(tmp_path, checkpoint_path)


def embed_chunks(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
    checkpoint_path: Path | None = None,
    fingerprint: str | None = None,
) -> np.ndarray:
    """
    Embed a list of texts in batches.

    Returns a numpy array of shape (n_texts, embedding_dim), L2-normalized
    so that inner product == cosine similarity.

    If checkpoint_path (and fingerprint) are given, progress is saved to disk
    after every 5,000-text slice, so an interrupted run resumes instead of
    re-embedding everything from scratch on retry — pass the same
    checkpoint_path/fingerprint again to resume.
    """
    log.info("Embedding %d texts with batch_size=%d...", len(texts), batch_size)

    slice_size = 5000
    start_index = 0
    embedded_chunks: list[np.ndarray] = []

    if checkpoint_path is not None and fingerprint is not None:
        resumed = _load_embed_checkpoint(checkpoint_path, fingerprint)
        if resumed is not None:
            prior_embeddings, start_index = resumed
            embedded_chunks.append(prior_embeddings)

    for i in range(start_index, len(texts), slice_size):
        text_slice = texts[i : i + slice_size]
        slice_embeddings = model.encode(
            text_slice,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via inner product
        ).astype(np.float32)
        embedded_chunks.append(slice_embeddings)
        next_index = i + len(text_slice)

        if checkpoint_path is not None and fingerprint is not None:
            _save_embed_checkpoint(checkpoint_path, np.concatenate(embedded_chunks, axis=0), next_index, fingerprint)
            log.info("Checkpointed progress: %d/%d texts embedded.", next_index, len(texts))

    embeddings = np.concatenate(embedded_chunks, axis=0) if embedded_chunks else np.zeros((0, 0), dtype=np.float32)
    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


def embed_chunks_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,
    checkpoint_path: Path | None = None,
    fingerprint: str | None = None,
) -> np.ndarray:
    """
    Embed texts using the OpenAI Embeddings API.

    Returns a numpy array of shape (n_texts, 1536), L2-normalized
    so that inner product == cosine similarity (matching the local model convention).

    If checkpoint_path (and fingerprint) are given, progress is saved after
    every batch — this matters here specifically because this path is a
    sequence of network-bound API calls that can take hours for the full
    corpus, and a rate limit / timeout / outage partway through would
    otherwise lose all prior batches' work. Pass the same
    checkpoint_path/fingerprint again to resume instead of re-calling the
    API for already-embedded texts.
    """
    import openai
    import tiktoken

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment / .env file")

    client = openai.OpenAI(api_key=api_key)

    max_tokens = 8192
    enc = tiktoken.encoding_for_model(model)
    truncated = 0
    safe_texts = []
    for t in texts:
        token_ids = enc.encode(t)
        if len(token_ids) > max_tokens:
            safe_texts.append(enc.decode(token_ids[:max_tokens]))
            truncated += 1
        else:
            safe_texts.append(t)
    if truncated:
        log.warning("Truncated %d texts to %d tokens for OpenAI API", truncated, max_tokens)

    start_index = 0
    all_embeddings: list[np.ndarray] = []
    if checkpoint_path is not None and fingerprint is not None:
        resumed = _load_embed_checkpoint(checkpoint_path, fingerprint)
        if resumed is not None:
            prior_embeddings, start_index = resumed
            all_embeddings.append(prior_embeddings)

    total_batches = (len(safe_texts) + batch_size - 1) // batch_size
    log.info(
        "Embedding %d texts via OpenAI '%s' (batch_size=%d, %d batches)%s...",
        len(safe_texts), model, batch_size, total_batches,
        f" — resuming from text {start_index}" if start_index else "",
    )

    for i in range(start_index, len(safe_texts), batch_size):
        batch = safe_texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embs = np.array([item.embedding for item in response.data], dtype=np.float32)
        all_embeddings.append(batch_embs)
        next_index = i + len(batch)
        log.info("  Batch %d/%d done (%d texts)", i // batch_size + 1, total_batches, len(batch))

        if checkpoint_path is not None and fingerprint is not None:
            # Checkpoint stores raw (pre-normalization) embeddings — normalization
            # happens once below, after the full loop, so resuming never double-normalizes.
            _save_embed_checkpoint(checkpoint_path, np.concatenate(all_embeddings, axis=0), next_index, fingerprint)

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, 1536), dtype=np.float32)

    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms

    log.info("OpenAI embeddings shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using inner product (cosine similarity on normalized vectors).

    FAISS gotcha: FAISS does not store metadata — only vectors. The index position
    (0, 1, 2, ...) maps to the corresponding entry in the metadata JSON file.
    This mapping is positional, so the metadata file and FAISS index must always
    be generated together and never modified independently.

    We use IndexFlatIP (flat inner product) because:
        - Our dataset (~200k chunks) is small enough for exact search
        - No training required (unlike IVF indexes)
        - With L2-normalized vectors, inner product == cosine similarity
        - For larger datasets (>1M), consider IndexIVFFlat for speed
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS index built: %d vectors, %d dimensions", index.ntotal, dim)
    return index


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """
    Build a BM25 keyword index over chunk texts.

    Tokenization is simple: lowercase + split on whitespace. Lyrics don't
    benefit from heavy NLP tokenization (stemming, lemmatization) because
    users often search for exact words and phrases.

    The BM25 index is positionally aligned with the FAISS index and metadata
    JSON — index i in BM25 corresponds to index i in FAISS and metadata.
    """
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    log.info("BM25 index built over %d documents", len(tokenized))
    return bm25


def save_bm25_index(bm25: BM25Okapi, output_path: str) -> None:
    """Save BM25 index to disk using pickle."""
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)
    log.info("BM25 index saved to %s", output_path)


def build_contextual_text(chunk: dict) -> str:
    """
    Build contextualized text for embedding by prepending metadata.

    Based on Anthropic's Contextual Retrieval technique. The intuition: a verse
    like "she left me standing in the cold" gets a generic embedding without
    context, but when the embedding model also sees the song title, artist, and
    section type, the resulting vector is much more specific and retrievable.

    The BM25 index uses the original text (not this), because BM25 should match
    on actual lyrics keywords, not the metadata prefix.
    """
    # Extract section header from the first line if present (e.g., "[Chorus]")
    text = chunk["text"]
    first_line = text.split("\n", 1)[0].strip()
    section = first_line if first_line.startswith("[") and first_line.endswith("]") else None

    header_parts = [f"Song: '{chunk['title']}' by {chunk['artist']}."]
    if chunk.get("album"):
        header_parts.append(f"Album: {chunk['album']}.")
    if chunk.get("release_date"):
        header_parts.append(f"Released: {chunk['release_date']}.")
    if section:
        header_parts.append(f"Section: {section}.")

    header = " ".join(header_parts)
    return f"{header}\n{text}"


def save_metadata(chunks: list[dict], output_path: str) -> None:
    """
    Save chunk metadata to JSON. Each entry's position matches its FAISS index position.

    The 'text' field is the original chunk text (used for display in the UI and
    as LLM context). The 'text_for_embedding' field is the contextualized text
    (used only at embedding time — not stored in metadata since it can be
    reconstructed, but the embedding was computed from it).
    """
    metadata = []
    for chunk in chunks:
        metadata.append({
            "text": chunk["text"],
            "title": chunk["title"],
            "artist": chunk["artist"],
            "release_date": chunk.get("release_date"),
            "album": chunk.get("album"),
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info("Saved metadata for %d chunks to %s", len(metadata), output_path)


def save_index_info(
    index_output_path: str,
    embedding_model: str,
    embedding_dim: int,
    contextual: bool,
    num_vectors: int,
) -> None:
    """
    Save a small sidecar JSON recording which embedding model built this FAISS
    index. QueryPipeline reads this at load time and refuses to start if the
    embedding model it's about to use to embed queries doesn't match what
    actually built the index — see pipeline/query.py::QueryPipeline.__init__.

    This is a new sibling file only — it does NOT touch metadata.json's
    existing flat-list-of-chunks schema.
    """
    info_path = Path(index_output_path).with_name(Path(index_output_path).stem + "_info.json")
    info = {
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "normalize": True,
        "distance_metric": "ip",
        "contextual": contextual,
        "num_vectors": num_vectors,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    log.info("Saved index info sidecar to %s", info_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "processed" / "chunks.jsonl"
    default_index = project_root / "data" / "processed" / "faiss.index"
    default_metadata = project_root / "data" / "processed" / "metadata.json"
    default_bm25 = project_root / "data" / "processed" / "bm25.pkl"

    parser = argparse.ArgumentParser(description="Embed and index lyric chunks")
    parser.add_argument("--input", default=str(default_input), help="Path to chunks JSONL")
    parser.add_argument("--index-output", default=str(default_index), help="Path to save FAISS index")
    parser.add_argument("--metadata-output", default=str(default_metadata), help="Path to save metadata JSON")
    parser.add_argument("--bm25-output", default=str(default_bm25), help="Path to save BM25 index (pickle)")
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2). "
             "Alternatives: BAAI/bge-small-en-v1.5 (same 384 dims, higher MTEB), "
             "all-mpnet-base-v2 (768 dims, higher quality, slower)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for embedding (default: 64 for local models, 2048 for --openai)"
    )
    parser.add_argument(
        "--skip-contextual", action="store_true",
        help="Skip contextual enrichment (embed raw lyrics without metadata header). "
             "Outputs are suffixed with _noctx to coexist with contextual index."
    )
    parser.add_argument(
        "--openai", action="store_true",
        help="Use OpenAI text-embedding-3-small instead of a local model. "
             "Requires OPENAI_API_KEY in .env. Outputs suffixed with _openai."
    )
    args = parser.parse_args()

    if args.batch_size is None:
        # OpenAI's embeddings endpoint comfortably handles large batches (up to a
        # few thousand inputs/request); the local-model default of 64 was silently
        # being reused for --openai runs, turning a ~86-call job into ~2,750 calls.
        args.batch_size = 2048 if args.openai else 64

    MODEL_SUFFIXES = {
        "all-MiniLM-L6-v2": "",
        "BAAI/bge-small-en-v1.5": "_bge",
        "all-mpnet-base-v2": "_mpnet",
    }

    # Detect whether the user supplied explicit output paths before we mutate them
    using_default_index = args.index_output == str(default_index)
    using_default_metadata = args.metadata_output == str(default_metadata)
    using_default_bm25 = args.bm25_output == str(default_bm25)

    def _add_suffix(path: str, suffix: str) -> str:
        p = Path(path)
        return str(p.with_stem(p.stem + suffix))

    # Apply _openai suffix when using OpenAI embeddings
    if args.openai:
        if using_default_index:
            args.index_output = _add_suffix(args.index_output, "_openai")
        if using_default_metadata:
            args.metadata_output = _add_suffix(args.metadata_output, "_openai")
        if using_default_bm25:
            args.bm25_output = _add_suffix(args.bm25_output, "_openai")

    # Apply model suffix for non-default local models (e.g. _bge, _mpnet)
    elif args.model != "all-MiniLM-L6-v2":
        model_suffix = MODEL_SUFFIXES.get(args.model)
        if model_suffix is None:
            slug = args.model.split("/")[-1]
            model_suffix = "_" + "".join(c if c.isalnum() else "_" for c in slug).strip("_")
        if using_default_index:
            args.index_output = _add_suffix(args.index_output, model_suffix)
        if using_default_metadata:
            args.metadata_output = _add_suffix(args.metadata_output, model_suffix)
        if using_default_bm25:
            args.bm25_output = _add_suffix(args.bm25_output, model_suffix)

    # Apply _noctx suffix to output paths when skipping contextual enrichment
    if args.skip_contextual:
        if using_default_index:
            args.index_output = _add_suffix(args.index_output, "_noctx")
        if using_default_metadata:
            args.metadata_output = _add_suffix(args.metadata_output, "_noctx")
        if using_default_bm25:
            args.bm25_output = _add_suffix(args.bm25_output, "_noctx")

    # Load chunks
    chunks = load_chunks(args.input)
    if not chunks:
        log.error("No chunks found. Run pipeline/chunking.py first.")
        return

    # Build texts for embedding: contextual (with metadata header) or raw
    texts_for_bm25 = [c["text"] for c in chunks]
    if args.skip_contextual:
        log.info("Skipping contextual enrichment — embedding raw lyrics text")
        texts_for_embedding = texts_for_bm25
    else:
        texts_for_embedding = [build_contextual_text(c) for c in chunks]
        log.info("Sample contextual text:\n%s", texts_for_embedding[0][:300])

    # Model label + checkpoint/fingerprint computed up front (both embed
    # functions and the index-info sidecar below need it)
    model_label = "text-embedding-3-small" if args.openai else args.model
    checkpoint_path = checkpoint_path_for(args.index_output)
    fingerprint = embedding_fingerprint(texts_for_embedding, model_label, not args.skip_contextual)

    # Embed chunks
    if args.openai:
        log.info("Using OpenAI text-embedding-3-small")
        embeddings = embed_chunks_openai(
            texts_for_embedding, batch_size=args.batch_size,
            checkpoint_path=checkpoint_path, fingerprint=fingerprint,
        )
    else:
        log.info("Loading model: %s", args.model)
        model = SentenceTransformer(args.model)
        embeddings = embed_chunks(
            model, texts_for_embedding, batch_size=args.batch_size,
            checkpoint_path=checkpoint_path, fingerprint=fingerprint,
        )

    # Build and save FAISS index
    index = build_faiss_index(embeddings)
    Path(args.index_output).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_output))
    log.info("FAISS index saved to %s", args.index_output)

    # Save metadata (positionally aligned with FAISS index)
    save_metadata(chunks, args.metadata_output)

    # Build and save BM25 index (uses original text, not contextualized)
    bm25 = build_bm25_index(texts_for_bm25)
    save_bm25_index(bm25, args.bm25_output)

    # Save the model-identity sidecar QueryPipeline asserts against at load time
    save_index_info(
        args.index_output,
        embedding_model=model_label,
        embedding_dim=embeddings.shape[1],
        contextual=not args.skip_contextual,
        num_vectors=embeddings.shape[0],
    )

    # Everything above succeeded — safe to drop the embedding checkpoint now.
    # Kept until this point (not right after embedding finishes) so a failure
    # in the FAISS/metadata/BM25 writes still lets a re-run skip re-embedding.
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Removed embedding checkpoint %s (run completed successfully).", checkpoint_path)

    # Summary
    log.info(
        "Done. %d chunks embedded with '%s' (%d dimensions). "
        "Index: %s | Metadata: %s | BM25: %s",
        len(chunks), model_label, embeddings.shape[1],
        args.index_output, args.metadata_output, args.bm25_output,
    )


if __name__ == "__main__":
    main()
