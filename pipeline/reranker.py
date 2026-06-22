"""
Cross-encoder reranker — LangChain version.

Replaces the custom Reranker class with LangChain's CrossEncoderReranker,
which wraps HuggingFaceCrossEncoder and implements BaseDocumentCompressor.
This means it can plug directly into a ContextualCompressionRetriever chain,
or be called standalone via compress_documents().

Usage as a module:
    from pipeline.reranker import build_reranker
    from langchain_core.documents import Document

    reranker = build_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)
    docs = [Document(page_content="...", metadata={...}), ...]
    reranked = reranker.compress_documents(docs, query="sad song about rain")
    # each doc gains doc.metadata["relevance_score"]

The cross-encoder scores each (query, doc) pair jointly — more accurate than
bi-encoder cosine similarity but too slow for first-stage retrieval over the
full index.  Use it as a second stage after FAISS narrows the candidates.
"""

import logging

log = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def build_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
):
    """
    Return a configured LangChain CrossEncoderReranker.

    The returned object is a BaseDocumentCompressor — call
    compress_documents(docs, query) to rerank a list of Documents.
    Scores are attached to doc.metadata["relevance_score"].

    Args:
        model_name: HuggingFace cross-encoder model.  Lightweight options:
                    cross-encoder/ms-marco-MiniLM-L-6-v2  (~80 MB, default)
                    cross-encoder/ms-marco-MiniLM-L-12-v2 (more accurate, slower)
                    BAAI/bge-reranker-base                 (alternative backbone)
        top_n:      How many top documents to keep after reranking.
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    try:
        # langchain 1.x: moved to langchain_classic
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    except ImportError:
        try:
            from langchain_community.document_compressors import CrossEncoderReranker
        except ImportError:
            from langchain.retrievers.document_compressors import CrossEncoderReranker

    log.info("Loading cross-encoder model: %s", model_name)
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    return CrossEncoderReranker(model=cross_encoder, top_n=top_n)
