You are an expert in information retrieval and RAG systems. I'm building a 
lyrics RAG pipeline for my thesis.

I have a working FAISS retrieval step (Phase 3) that returns top-k lyric 
chunks by cosine similarity using sentence-transformers embeddings. I now 
want to add a re-ranking step between FAISS retrieval and LLM synthesis.

Help me write a Python script (pipeline/reranker.py) that implements 
cross-encoder re-ranking. Specifically:

1. Use the sentence-transformers CrossEncoder class with the model 
   'cross-encoder/ms-marco-MiniLM-L-6-v2'. Explain why this model is 
   appropriate for this task.
2. Accept as input: a query string and a list of retrieved chunk dicts 
   (with fields: text, title, artist, release_date, album, score, 
   chunk_index)
3. Re-score each chunk by passing the (query, chunk_text) pair through 
   the cross-encoder
4. Return the re-ranked list sorted by cross-encoder score, with both 
   the original FAISS cosine score and the new cross-encoder score 
   preserved in each chunk dict
5. Expose a Reranker class that loads the model once and can re-rank 
   multiple queries efficiently
6. Make re-ranking optional via a flag (use_reranker=True/False) so 
   Phase 6 can compare retrieval with and without re-ranking

The Reranker class should be importable by query.py (Phase 4) and the 
evaluation notebook (Phase 6). If you make any assumptions about the 
cross-encoder's score range or normalization, state them clearly.

Explain the key tradeoff between bi-encoder retrieval (FAISS) and 
cross-encoder re-ranking, and why combining them (retrieve then re-rank) 
is better than using either alone.