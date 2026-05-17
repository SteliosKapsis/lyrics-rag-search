# Phase 6 — Evaluation & Experimentation Notebook

You are an expert in information retrieval evaluation and NLP 
experimentation. I'm building a lyrics RAG pipeline for my thesis and 
need to rigorously evaluate and tune it.

I have a QueryPipeline class (pipeline/query.py) that supports the
following configurable options:
   - Embedding model (all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, or
     all-mpnet-base-v2)
   - Chunk size and overlap (set at index time via chunking.py)
   - HyDE retrieval (use_hyde flag)
   - Hybrid retrieval (use_hybrid flag) — FAISS dense + BM25 sparse,
     merged with Reciprocal Rank Fusion
   - Cross-encoder re-ranking (use_reranker flag)
   - LLM synthesis backend (anthropic or ollama/llama3)

Help me create a Jupyter notebook (notebooks/evaluation.ipynb) for 
evaluating my RAG pipeline's retrieval quality and comparing technique 
combinations. The notebook should:

1. Define a test set of 20-30 evaluation queries with ground truth 
   (expected song title + artist). Include a mix of:
   - Direct queries ("song with the lyric never gonna give you up")
   - Semantic/descriptive queries ("upbeat song about dancing and 
     feeling free")
   - Partial/vague lyric recall ("the song that goes something about 
     rivers and gold")
   - Ambiguous queries that could match multiple songs
   Ground truth should be stored as a list of dicts: 
   {query, expected_title, expected_artist}

2. Implement evaluation metrics:
   - Precision@k: is the correct song in the top-k retrieved results?
   - Mean Reciprocal Rank (MRR): where does the correct song rank?
   - Hit Rate: simpler binary — did the correct song appear at all?
   For each metric, explain what it measures and why it's appropriate 
   for this use case.

3. Run a structured experiment grid comparing these technique combinations:
   - Baseline (FAISS only): dense retrieval, no re-ranking, no HyDE
   - + Hybrid: FAISS + BM25 with Reciprocal Rank Fusion
   - + Hybrid + Re-ranking: hybrid retrieval + cross-encoder re-ranking
   - + HyDE: HyDE retrieval (FAISS only), no re-ranking
   - + HyDE + Hybrid: HyDE + hybrid retrieval
   - + HyDE + Hybrid + Re-ranking: all techniques combined
   Run all combinations against available embedding models
   (all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, and optionally
   all-mpnet-base-v2), keeping chunk size fixed at the best value
   found in preliminary experiments.

   bge-small-en-v1.5 is the same 384 dimensions as MiniLM but scores
   ~3 points higher on MTEB — a zero-effort quality comparison for
   the thesis. Both use the same FAISS index structure, but each
   model requires its own index (the embeddings are different).

   The key comparison is: how much does adding BM25 sparse retrieval
   improve over dense-only? This should be especially visible on
   direct lyric recall queries where exact keyword matching matters.

4. Also run a separate parameter sweep for:
   - Chunk size (200, 400, 600 characters)
   - Chunk overlap (0, 1, 2 lines)
   Using the baseline configuration only (no HyDE, no re-ranking) to 
   isolate the effect of chunking on retrieval quality.

4b. Run a combined optimal parameters experiment:
   - Test k=20 with fetch_k_multiplier=8 using Hybrid + Re-ranker.
     Previous parameter sweeps tested top-k and fetch_k_multiplier
     independently; this experiment tests their combination to see if
     the gains stack. Compare against the individual best results
     (k=20 alone: 0.767 MRR, 8x multiplier alone: 0.753 MRR).

4c. Run a re-ranker model comparison:
   - Sweep across three cross-encoder models using the best configuration
     (Baseline + Re-ranker, k=5):
     - `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, current)
     - `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M params)
     - `BAAI/bge-reranker-base` (110M params)
   - Report MRR, Hit Rate, and average inference latency per query for
     each model to quantify the accuracy-latency tradeoff.

4d. If the non-contextual index exists (built with `--skip-contextual`
   in Phase 3), run a contextual embedding ablation:
   - Re-run the full 8-configuration technique grid on the non-contextual
     index (same queries, same metrics).
   - Compare contextual vs. non-contextual MRR for each configuration.
   - Report the isolated contribution of contextual embeddings as the
     delta between the two indices. Anthropic reported 35% reduction in
     retrieval failure; measure the actual gain in this lyrics domain.

5. Visualize results:
   - A heatmap or table comparing all technique combinations vs. metrics
   - A separate table for the chunking parameter sweep
   - Distribution plot of similarity scores for correct vs. incorrect 
     matches (for baseline vs. best configuration)
   - A bar chart comparing MRR across all technique combinations

6. Include an LLM synthesis comparison section:
   - Run the same 5 representative queries through both the Anthropic 
     (Claude Haiku) and Ollama (Llama 3) backends
   - Display responses side by side for qualitative comparison
   - Note: this section is qualitative, not metric-based

8. Add an LLM-as-a-judge evaluation section that goes beyond retrieval
   metrics to assess the quality of the full RAG pipeline (retrieval +
   generation). Use the DeepEval framework with Ollama as the local
   judge LLM (no additional API costs).

   Evaluate on 5 representative queries (one per query type) using
   these metrics:
   - Faithfulness: does the LLM response only contain claims supported
     by the retrieved context? (Detects hallucinated lyrics/metadata)
   - Answer Relevancy: does the response actually answer the user's
     query? (Detects off-topic or generic responses)
   - Context Precision: are the relevant chunks ranked higher than
     irrelevant ones in the retrieved context?
   - Context Recall: does the retrieved context cover all the
     information needed to answer the query?

   For each metric, explain what it measures, why it matters for a
   lyrics RAG system, and how it's computed (LLM-based scoring).
   Display results as a per-query table with scores for each metric,
   plus averages. Include a brief interpretation of what the scores
   mean for the pipeline's strengths and weaknesses.

   This section transforms the evaluation from "retrieval metrics only"
   to a complete RAG evaluation covering both retrieval and generation
   quality — important for a thesis.

9. Output a final summary section that:
   - Identifies the best performing configuration with its metric scores
   - States which techniques provided the most improvement over baseline
   - Formats results as a clean table suitable for inclusion in a 
     thesis appendix

Use matplotlib and seaborn for plots. Structure the notebook with clear 
markdown sections and subsection headers so it reads as a standalone 
document in a thesis appendix. Each experiment cell should be 
independently re-runnable. If you're unsure about any metric 
implementation detail, say so rather than silently making an assumption.