"""
Embedding and FAISS indexing script for RAG pipeline — LangChain version.

Replaces the manual faiss / numpy / SentenceTransformer code with:
  - langchain_huggingface.HuggingFaceEmbeddings  (local models)
  - langchain_openai.OpenAIEmbeddings             (OpenAI cloud)
  - langchain_community.vectorstores.FAISS        (vector store)

The FAISS vectorstore is saved to a folder (LangChain format) instead of a
single .index file.  BM25 and metadata.json are kept for hybrid retrieval
alignment (same format as the custom branch).

Usage (from project root):
    .venv/Scripts/python pipeline/embedding.py
    .venv/Scripts/python pipeline/embedding.py --model BAAI/bge-small-en-v1.5
    .venv/Scripts/python pipeline/embedding.py --openai
    .venv/Scripts/python pipeline/embedding.py --skip-contextual

Inputs:  data/processed/chunks.jsonl
Outputs:
    data/processed/faiss_lc/          (LangChain FAISS folder — index.faiss + index.pkl)
    data/processed/metadata.json      (positionally aligned with BM25)
    data/processed/bm25.pkl           (BM25Okapi keyword index)

  Suffixed variants coexist in the same folder:
    faiss_lc_bge/, faiss_lc_openai/, faiss_lc_noctx/, …
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def build_contextual_text(chunk: dict) -> str:
    """
    Build contextualized text by prepending metadata as a header.

    Based on Anthropic's Contextual Retrieval technique — the embedding model
    sees the song title, artist, and section label alongside the lyrics, so
    the resulting vector is far more specific than a bare lyric fragment.

    The BM25 index uses the original text (not this), because BM25 should
    match on actual lyrics keywords, not the metadata prefix.
    """
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

    return " ".join(header_parts) + "\n" + text


def build_documents(chunks: list[dict], use_contextual: bool = True):
    """
    Convert chunk dicts to LangChain Document objects.

    page_content  — text that gets embedded (contextual or raw lyrics).
    metadata      — original lyrics text + all chunk fields for downstream use.
                    Storing the original text separately lets query.py display
                    raw lyrics while the FAISS index was built on contextual text.
    """
    from langchain_core.documents import Document

    docs = []
    for chunk in chunks:
        page_content = build_contextual_text(chunk) if use_contextual else chunk["text"]
        metadata = {
            "original_text": chunk["text"],
            "title": chunk["title"],
            "artist": chunk["artist"],
            "album": chunk.get("album"),
            "release_date": chunk.get("release_date"),
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """
    Build a BM25 keyword index over chunk texts.

    Simple whitespace tokenisation — lyrics don't benefit from stemming because
    users often search for exact words.  Positional alignment with FAISS and
    metadata.json is maintained by processing in the same chunk order.
    """
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    log.info("BM25 index built over %d documents", len(tokenized))
    return bm25


def save_bm25_index(bm25: BM25Okapi, output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(bm25, f)
    log.info("BM25 index saved to %s", output_path)


def save_metadata(chunks: list[dict], output_path: str) -> None:
    """
    Save chunk metadata to JSON — positionally aligned with BM25.

    Each position i in this file corresponds to position i in the BM25 index,
    so query.py can convert a BM25 integer result back to a full chunk dict.
    """
    metadata = [
        {
            "text": c["text"],
            "title": c["title"],
            "artist": c["artist"],
            "release_date": c.get("release_date"),
            "album": c.get("album"),
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
        }
        for c in chunks
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info("Saved metadata for %d chunks to %s", len(metadata), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    processed = project_root / "data" / "processed"
    default_input = project_root / "data" / "processed" / "chunks.jsonl"
    # LangChain FAISS saves to a folder, not a single file
    default_index = str(processed / "faiss_lc")
    default_metadata = str(processed / "metadata.json")
    default_bm25 = str(processed / "bm25.pkl")

    parser = argparse.ArgumentParser(description="Embed and index lyric chunks (LangChain version)")
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--index-output", default=default_index,
                        help="Folder path for the LangChain FAISS vectorstore")
    parser.add_argument("--metadata-output", default=default_metadata)
    parser.add_argument("--bm25-output", default=default_bm25)
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformers model (default: all-MiniLM-L6-v2). "
             "Alternatives: BAAI/bge-small-en-v1.5, all-mpnet-base-v2",
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for local model encoding (default: 64)")
    parser.add_argument("--skip-contextual", action="store_true",
                        help="Skip contextual enrichment; outputs suffixed with _noctx")
    parser.add_argument("--openai", action="store_true",
                        help="Use OpenAI text-embedding-3-small; outputs suffixed with _openai")
    args = parser.parse_args()

    MODEL_SUFFIXES = {
        "all-MiniLM-L6-v2": "",
        "BAAI/bge-small-en-v1.5": "_bge",
        "all-mpnet-base-v2": "_mpnet",
    }

    using_default_index = args.index_output == default_index
    using_default_metadata = args.metadata_output == default_metadata
    using_default_bm25 = args.bm25_output == default_bm25

    def _add_suffix(path: str, suffix: str) -> str:
        p = Path(path)
        return str(p.parent / (p.name + suffix))

    if args.openai:
        if using_default_index:
            args.index_output = _add_suffix(args.index_output, "_openai")
        if using_default_metadata:
            args.metadata_output = _add_suffix(args.metadata_output, "_openai")
        if using_default_bm25:
            args.bm25_output = _add_suffix(args.bm25_output, "_openai")
    elif args.model != "all-MiniLM-L6-v2":
        model_suffix = MODEL_SUFFIXES.get(args.model)
        if model_suffix is None:
            slug = args.model.split("/")[-1]
            model_suffix = "_" + "".join(c if c.isalnum() else "_" for c in slug).strip("_")
        if using_default_index:
            args.index_output = _add_suffix(args.index_output, model_suffix)
        if using_default_metadata:
            args.metadata_output = _add_suffix(args.metadata_output, "_" + model_suffix.lstrip("_"))
        if using_default_bm25:
            args.bm25_output = _add_suffix(args.bm25_output, model_suffix)

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

    # Build LangChain Document objects
    use_contextual = not args.skip_contextual
    docs = build_documents(chunks, use_contextual=use_contextual)
    if use_contextual:
        log.info("Sample contextual page_content:\n%s", docs[0].page_content[:300])
    else:
        log.info("Skipping contextual enrichment — embedding raw lyrics text")

    # Create LangChain embeddings function
    if args.openai:
        from langchain_openai import OpenAIEmbeddings
        log.info("Using OpenAI text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        model_label = "text-embedding-3-small"
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        log.info("Loading HuggingFace model: %s", args.model)
        embeddings = HuggingFaceEmbeddings(
            model_name=args.model,
            encode_kwargs={"normalize_embeddings": True, "batch_size": args.batch_size},
            show_progress=True,
        )
        model_label = args.model

    # Build and save LangChain FAISS vectorstore
    from langchain_community.vectorstores import FAISS

    log.info("Building FAISS vectorstore from %d documents...", len(docs))
    vectorstore = FAISS.from_documents(docs, embeddings)

    Path(args.index_output).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(args.index_output)
    log.info("FAISS vectorstore saved to %s/", args.index_output)

    # Save metadata.json — positionally aligned with BM25 for hybrid retrieval
    save_metadata(chunks, args.metadata_output)

    # Build and save BM25 index (uses original lyrics text, not contextual)
    bm25_texts = [c["text"] for c in chunks]
    bm25 = build_bm25_index(bm25_texts)
    save_bm25_index(bm25, args.bm25_output)

    log.info(
        "Done. %d chunks embedded with '%s'. "
        "FAISS: %s/ | Metadata: %s | BM25: %s",
        len(chunks), model_label,
        args.index_output, args.metadata_output, args.bm25_output,
    )


if __name__ == "__main__":
    main()
