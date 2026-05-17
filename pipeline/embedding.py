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
"""

import argparse
import json
import logging
import pickle
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


def embed_chunks(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embed a list of texts in batches.

    Returns a numpy array of shape (n_texts, embedding_dim), L2-normalized
    so that inner product == cosine similarity.
    """
    log.info("Embedding %d texts with batch_size=%d...", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via inner product
    )

    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


def embed_chunks_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Embed texts using the OpenAI Embeddings API.

    Returns a numpy array of shape (n_texts, 1536), L2-normalized
    so that inner product == cosine similarity (matching the local model convention).
    """
    import os
    import openai
    import tiktoken

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment / .env file")

    client = openai.OpenAI(api_key=api_key)
    all_embeddings = []

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

    log.info("Embedding %d texts via OpenAI '%s' (batch_size=%d)...", len(safe_texts), model, batch_size)

    for i in range(0, len(safe_texts), batch_size):
        batch = safe_texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embs)
        log.info("  Batch %d/%d done (%d texts)", i // batch_size + 1, (len(safe_texts) + batch_size - 1) // batch_size, len(batch))

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms

    log.info("OpenAI embeddings shape: %s", embeddings.shape)
    return embeddings


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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding (default: 64)")
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

    # Embed chunks
    if args.openai:
        log.info("Using OpenAI text-embedding-3-small")
        embeddings = embed_chunks_openai(texts_for_embedding, batch_size=args.batch_size)
        model_label = "text-embedding-3-small"
    else:
        log.info("Loading model: %s", args.model)
        model = SentenceTransformer(args.model)
        embeddings = embed_chunks(model, texts_for_embedding, batch_size=args.batch_size)
        model_label = args.model

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

    # Summary
    log.info(
        "Done. %d chunks embedded with '%s' (%d dimensions). "
        "Index: %s | Metadata: %s | BM25: %s",
        len(chunks), model_label, embeddings.shape[1],
        args.index_output, args.metadata_output, args.bm25_output,
    )


if __name__ == "__main__":
    main()
