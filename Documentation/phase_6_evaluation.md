# Phase 6 — Evaluation & Experimentation

## Overview

A Jupyter notebook that rigorously evaluates the RAG pipeline's retrieval quality across different technique combinations, parameter configurations, and LLM backends. Produces metrics, visualizations, and thesis-ready summary tables. The evaluation covers both retrieval quality (Precision@k, MRR, Hit Rate) and full RAG pipeline quality (Faithfulness, Answer Relevancy, Context Precision, Context Recall via LLM-as-judge).

## Notebook

`notebooks/evaluation.ipynb`

## Usage

```bash
# From project root:
.venv\Scripts\jupyter notebook notebooks/evaluation.ipynb

# Or with JupyterLab:
.venv\Scripts\jupyter lab notebooks/evaluation.ipynb
```

Each section is independently re-runnable. The notebook saves plot images to `notebooks/` for easy inclusion in the thesis.

## Prerequisites

- **FAISS index + metadata** from Phase 3 (`data/processed/faiss.index`, `data/processed/metadata.json`)
- **Anthropic API key** in `.env` (for LLM synthesis, HyDE, and LLM-as-judge)
- **OpenAI API key** in `.env` as `OPENAI_API_KEY` (for LLM-as-judge second opinion and RAG vs. Pure LLM baseline)
- **Ollama + Llama 3** (optional, for LLM synthesis comparison section only)

## Dependencies

- `matplotlib>=3.7.0` — plots
- `seaborn>=0.12.0` — heatmaps and styling
- `pandas>=2.0.0` — data tables
- `deepeval>=1.0.0` — LLM-as-judge evaluation metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- All Phase 4 dependencies (sentence-transformers, faiss-cpu, anthropic, etc.)

## Test Set

75 evaluation queries with ground truth (`expected_title`, `expected_artist`), covering four query types:

| Type | Count | Description |
|------|-------|-------------|
| Direct lyric | 19 | User quotes or closely paraphrases actual lyrics |
| Semantic/descriptive | 19 | User describes the song's theme, mood, or genre |
| Partial recall | 19 | User vaguely remembers a lyric fragment |
| Ambiguous | 18 | Query could plausibly match multiple songs |

Songs in the test set are drawn from the actual indexed dataset (confirmed present in `chunks.jsonl`). The test set was expanded from 25 to 75 queries for more robust statistical significance testing and reduced variance from individual query difficulty. Queries were iteratively refined during evaluation — generic descriptions were replaced with lyrics-grounded formulations, non-English songs were removed, and duplicate entries were replaced.

## Evaluation Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Precision@k | Fraction of queries where the correct song appears in top-k | Basic retrieval success within the display window |
| MRR (Mean Reciprocal Rank) | Average of 1/rank across all queries | How high the correct song ranks (rewards top-1 placement) |
| Hit Rate | Fraction of queries where the correct song appears at any rank | Whether the pipeline can find the song at all |

## Experiments

### Experiment 1: Full Technique Combination Grid (Section 3.1)

Tests all 2^3 = 8 combinations of {Hybrid, HyDE, Re-ranker} across available embedding models:

| Configuration | Hybrid (BM25+FAISS) | HyDE | Re-ranker |
|--------------|---------------------|------|-----------|
| Baseline (FAISS only) | No | No | No |
| + Re-ranker | No | No | Yes |
| + Hybrid | Yes | No | No |
| + Hybrid + Re-ranker | Yes | No | Yes |
| + HyDE | No | Yes | No |
| + HyDE + Re-ranker | No | Yes | Yes |
| + HyDE + Hybrid | Yes | Yes | No |
| + HyDE + Hybrid + Re-ranker | Yes | Yes | Yes |

The full combinatorial grid ensures no technique interaction is missed. Earlier experiments with only 6 configurations (missing `+ Re-ranker` and `+ HyDE + Re-ranker`) masked the re-ranker's true effectiveness as the single most impactful technique in isolation.

**Embedding models tested:**
- `all-MiniLM-L6-v2` (default, 384 dims, ~80MB)
- `BAAI/bge-small-en-v1.5` (384 dims, ~50MB — same index structure as MiniLM, ~3 points higher on MTEB)
- `all-mpnet-base-v2` (optional, 768 dims, ~420MB — requires separate index)
- `text-embedding-3-small` (OpenAI API, 1536 dims — requires `--openai` flag and API key; separate index)

Each embedding model requires its own FAISS index. The notebook auto-detects which indices are available and runs the grid on those.

**Key result:** Baseline + Re-ranker (0.653 MRR with MiniLM) is the best configuration at default k=5, surpassing all other combinations including those with Hybrid and HyDE.

### Experiment 2: Top-k Sensitivity (Section 3.2)

Sweeps top_k = {3, 5, 10, 20} for Baseline and Baseline + Re-ranker to measure how the number of retrieved results affects retrieval quality.

| top_k | Baseline MRR | Baseline HR | + Re-ranker MRR | + Re-ranker HR |
|-------|-------------|-------------|-----------------|----------------|
| 3     | 0.540       | 0.560       | 0.580           | 0.600          |
| 5     | 0.563       | 0.640       | 0.653           | 0.680          |
| 10    | 0.573       | 0.680       | 0.687           | 0.760          |
| **20**| **0.585**   | **0.760**   | **0.767**       | **0.880**      |

The re-ranker's benefit scales with k: at k=20, it achieves **0.767 MRR and 88% Hit Rate** — the highest scores in the entire evaluation. The re-ranker advantage grows from +0.040 MRR at k=3 to +0.182 at k=20, demonstrating that giving the cross-encoder more candidates consistently improves performance.

### Experiment 3: RRF Parameter Tuning (Section 3.3)

Sweeps rrf_k = {20, 40, 60, 80, 100} for Hybrid and Hybrid + Re-ranker configurations to test the impact of the Reciprocal Rank Fusion constant on retrieval quality.

| rrf_k | Hybrid MRR | Hybrid + RR MRR |
|-------|-----------|-----------------|
| **20**| 0.451     | **0.633**       |
| 40    | 0.451     | 0.593           |
| 60    | 0.451     | 0.553           |
| 80    | 0.451     | 0.553           |
| 100   | 0.451     | 0.553           |

Hybrid retrieval without re-ranking is completely insensitive to rrf_k (identical 0.451 across all values). Lower rrf_k improves Hybrid + Re-ranker by weighting FAISS results more heavily, but even the best (0.633 at rrf_k=20) cannot surpass pure Baseline + Re-ranker (0.653).

### Experiment 4: Fetch-k Multiplier (Section 3.4)

Sweeps fetch_k_multiplier = {2, 3, 5, 8} for Baseline + Re-ranker and Hybrid + Re-ranker. The multiplier controls how many candidates the cross-encoder re-ranks (fetch_k = top_k × multiplier).

| Multiplier | fetch_k | Baseline + RR MRR | Baseline + RR HR | Hybrid + RR MRR | Hybrid + RR HR |
|------------|---------|-------------------|------------------|-----------------|----------------|
| 2x         | 10      | 0.620             | 0.640            | 0.523           | 0.600          |
| 3x (default) | 15   | 0.653             | 0.680            | 0.553           | 0.600          |
| 5x         | 25      | 0.653             | 0.720            | 0.633           | 0.720          |
| **8x**     | **40**  | 0.693             | 0.760            | **0.753**       | **0.840**      |

With a sufficiently large candidate pool (8x), Hybrid + Re-ranker (0.753 MRR, 0.840 HR) surpasses Baseline + Re-ranker (0.693/0.760) for the first time. At high multipliers, BM25's additional candidates give the re-ranker more material to work with, overcoming the noise that harms lower-pool configurations.

### Experiment 5: Chunking Parameter Sweep

Tests different chunk sizes and overlap values using baseline configuration only (to isolate the chunking variable):

| Chunk Size | Overlap Lines |
|-----------|---------------|
| 200 | 0 |
| 400 (default) | 0 |
| 600 | 0 |
| 400 | 1 |
| 400 | 2 |

Each combination requires re-running `pipeline/chunking.py` and `pipeline/embedding.py`. The notebook provides the exact commands and auto-detects available indices.

### Experiment 6: LLM Synthesis Comparison

Qualitative (not metric-based) comparison of Claude Haiku vs. Llama 3 on 5 representative queries (one per query type). Responses are displayed side by side for manual inspection. Gracefully skips Ollama if not running.

### Experiment 7: LLM-as-a-Judge (RAG Quality Evaluation)

Goes beyond retrieval metrics to assess the full RAG pipeline (retrieval + generation) using the DeepEval framework with **two independent judges**: Claude Haiku (Anthropic) and GPT-4o-mini (OpenAI).

**Metrics:**

| Metric | What It Measures | Why It Matters for Lyrics RAG |
|--------|-----------------|-------------------------------|
| Faithfulness | Whether the LLM response only contains claims supported by retrieved context | Detects hallucinated lyrics, fabricated metadata, or invented song attributions |
| Answer Relevancy | Whether the response actually answers the user's query | Detects off-topic or generic responses that ignore the query intent |
| Context Precision | Whether relevant chunks are ranked higher than irrelevant ones | Validates that retrieval ordering is useful (not just recall) |
| Context Recall | Whether the retrieved context covers all information needed to answer | Identifies cases where important chunks were missed by retrieval |

**Setup:**
- Uses 5 representative queries (one per query type from the test set)
- **Two judges run independently**: Claude Haiku (`claude-haiku-4-5-20251001`) via `AnthropicModel` and GPT-4o-mini via `GPTModel` — both using DeepEval's built-in model wrappers
- Both judges read API keys from `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
- Each query is run through the best-performing pipeline configuration from Experiment 1
- Results displayed per judge, plus an inter-judge agreement table showing scores and delta per metric

**Why two judges:** Claude Haiku is also the generation model, so using it as the sole judge introduces self-evaluation bias. GPT-4o-mini provides an independent assessment. Agreement between judges strengthens result credibility; disagreement surfaces where evaluation is subjective.

**Why this matters for the thesis:** Retrieval metrics (Precision@k, MRR) only tell you whether the right chunks were found. LLM-as-judge metrics tell you whether the final answer is faithful, relevant, and well-grounded — completing the evaluation picture from retrieval quality through to generation quality.

## Pipeline Parameters

The evaluation notebook exercises these configurable parameters of the `QueryPipeline`:

| Parameter | Default | Swept Values | Description |
|-----------|---------|-------------|-------------|
| `top_k` | 5 | 3, 5, 10, 20 | Number of song results to return |
| `use_hybrid` | True | True/False | Combine FAISS dense + BM25 sparse via RRF |
| `use_hyde` | False | True/False | Generate hypothetical lyrics before retrieval |
| `use_reranker` | False | True/False | Re-score top candidates with cross-encoder |
| `rrf_k` | 60 | 20, 40, 60, 80, 100 | RRF fusion constant (lower = more weight to top-ranked method) |
| `fetch_k_multiplier` | 3 | 2, 3, 5, 8 | Candidate pool size for re-ranking (fetch_k = top_k × multiplier) |
| `reranker_model` | `ms-marco-MiniLM-L-6-v2` | L-6-v2, L-12-v2, bge-reranker-base | Cross-encoder model for re-ranking |

## Visualizations

| Plot | File | Description |
|------|------|-------------|
| Technique heatmap | `notebooks/technique_heatmap.png` | Heatmap of all metrics across 8 technique combinations |
| MRR bar chart | `notebooks/mrr_comparison.png` | Horizontal bar chart comparing MRR across all configurations |
| Score distribution | `notebooks/score_distribution.png` | Histograms of correct vs. incorrect match scores (baseline vs. best) |
| Top-k sensitivity | `notebooks/topk_sensitivity.png` | Line chart of MRR/HR vs. top_k for Baseline and Baseline + Re-ranker |
| RRF tuning | `notebooks/rrf_tuning.png` | Line chart of MRR vs. rrf_k with Baseline MRR reference line |
| Fetch-k multiplier | `notebooks/fetchk_multiplier.png` | Line chart of MRR/HR vs. multiplier for Baseline+RR and Hybrid+RR |
| Chunking sweep table | `notebooks/chunking_sweep.png` | Rendered table of chunking parameter results |
| LLM-as-judge scores | `notebooks/llm_judge_scores.png` | Side-by-side heatmaps of all four DeepEval metrics for Claude Haiku and GPT-4o-mini judges |
| RAG vs. Pure LLM | `notebooks/rag_vs_llm_baseline.png` | Overall and per-query-type Hit Rate across 5 conditions: Claude (no ctx), GPT (no ctx), RAG retrieval only, RAG+Claude, RAG+GPT |
| Per-query-type breakdown | `notebooks/per_query_type_breakdown.png` | MRR and Hit Rate grouped bar charts by query type |
| Latency profile | `notebooks/latency_profile.png` | Stacked bar chart of per-stage latency breakdown |

## Output

The final section produces:
- Best configuration identification with metric scores
- Improvement over baseline (delta for each metric)
- Per-technique contribution analysis
- Parameter sensitivity findings
- A clean summary table formatted for thesis appendix inclusion

## Key Findings

1. **FAISS + Cross-encoder re-ranking is the optimal pipeline.** Baseline + Re-ranker (0.653 MRR at k=5, 0.767 MRR at k=20) is the best configuration. The re-ranker was only discovered to be effective in isolation after expanding the grid from 6 to 8 combinations.

2. **The re-ranker's benefit scales with candidate pool size.** Whether increased via higher top-k (+0.182 MRR at k=20) or higher fetch multiplier (+0.040 MRR at 8x), giving the cross-encoder more candidates consistently improves performance.

3. **Contextual embeddings make dense retrieval a strong baseline.** Prepending metadata headers achieves MRR 0.563 with no additional techniques.

4. **HyDE consistently degrades performance (-57% to -86% MRR).** Lyrics queries already contain document-space text, making HyDE's hypotheses redundant or harmful.

5. **Hybrid search harms at default settings but helps at scale.** At 8x fetch multiplier, Hybrid + Re-ranker (0.753 MRR) surpasses Baseline + Re-ranker (0.693 MRR).

6. **Query specificity is the dominant factor.** Improving test queries from generic descriptions to lyrics-grounded formulations improved baseline MRR by +51%, a larger gain than any technique combination.

## Design Decisions

- **Full combinatorial grid (2^3 = 8 configurations)** — testing all combinations of {Hybrid, HyDE, Re-ranker} ensures no interaction effects are missed. The original 6-combo grid failed to test the re-ranker in isolation, masking its true effectiveness.
- **`skip_llm=True` for retrieval evaluation** — the technique grid skips LLM synthesis and only runs retrieval + grouping. This is ~10x faster and evaluates retrieval quality independently of LLM output quality.
- **Log suppression in sweep cells** — verbose per-query retrieval logging is suppressed during grid and parameter sweeps to prevent output truncation in Jupyter HTML exports. The `pipeline.query` logger is temporarily set to `WARNING` level during sweeps and restored afterward.
- **Normalized string matching for ground truth** — titles and artists are lowercased and stripped of punctuation before comparison, with partial match fallback for formatting differences (e.g., "Don't Stop Believin'" vs "Dont Stop Believin").
- **Separate LLM comparison section** — LLM output quality is subjective and not reducible to a single metric, so it's evaluated qualitatively in its own section.
- **Auto-detection of available indices** — the notebook doesn't fail if optional indices (mpnet, chunking variants) aren't built. It runs on whatever is available and prints instructions for building missing ones.
- **Plots saved to disk** — all visualizations are saved as PNG files for easy copy-paste into a thesis document.
- **Parameter sweeps use best-performing base configuration** — top-k and fetch-k sweeps use the best configuration from the technique grid (Baseline + Re-ranker) as the starting point, isolating the effect of each parameter.

### Experiment 8: Combined Parameter & Re-ranker Sweep (Section 7.5)

Full combinatorial sweep across all tunable parameters and re-ranker models. Previous experiments (3.2–3.4) swept parameters independently — this experiment tests every combination together to find the true global optimum.

**Dimensions swept:**

| Dimension | Values | Count |
|-----------|--------|-------|
| `top_k` | 3, 5, 10, 20 | 4 |
| `fetch_k_multiplier` | 2, 3, 5, 8 | 4 |
| Technique stack | Baseline+RR, Hybrid+RR | 2 |
| Re-ranker model | L-6-v2, L-12-v2, bge-reranker-base | 3 |
| **Re-ranker runs** | | **96** |
| No-reranker runs | Baseline, Hybrid × 4 top_k values | **8** |
| **Total runs** | | **104** |

Non-reranker configs (Baseline, Hybrid) are swept over top_k only since fetch_k_multiplier and reranker model have no effect without a re-ranker.

**Visualizations produced:**
| Plot | File | Description |
|------|------|-------------|
| Re-ranker model comparison | `notebooks/reranker_model_comparison.png` | Bar charts: mean MRR, Hit Rate, and latency per re-ranker model |
| Combined sweep heatmap | `notebooks/combined_sweep_heatmap.png` | Heatmap of MRR across top_k × fetch_k_mult for best re-ranker |
| Top-k curves by re-ranker | `notebooks/combined_sweep_topk_curves.png` | MRR vs top_k for each re-ranker model at best fetch multiplier |

**Analysis tables:**
- Top 10 configurations by MRR (overall)
- Best configuration per re-ranker model
- Best configuration per technique stack

### Experiment 9: Per-Query-Type Breakdown (Section 3.1.1)

Breaks down MRR and Hit Rate by query type (direct lyric, semantic, partial recall, ambiguous) for each technique configuration on the default embedding model. Produces pivot tables and a grouped bar chart showing which techniques help which query types.

**Why it matters:** Aggregate metrics can mask important patterns — e.g., hybrid search might excel at direct lyric queries (exact keywords) while hurting semantic queries. Per-type analysis provides actionable insight into when each technique should be enabled.

### Experiment 10: Latency Profiling (Section 7.7)

End-to-end latency breakdown by pipeline stage: query embedding, FAISS search, BM25 search (if hybrid), RRF fusion (if hybrid), and cross-encoder re-ranking (if enabled). Tests four configurations (Baseline, Baseline+RR, Hybrid, Hybrid+RR) using 5 representative queries.

**Output:**
| Plot | File | Description |
|------|------|-------------|
| Latency stacked bar | `notebooks/latency_profile.png` | Per-stage latency breakdown for each configuration |

**Why it matters:** The accuracy-latency tradeoff is central to choosing a production configuration. A larger re-ranker candidate pool may improve MRR but add significant latency — the profiling data quantifies this tradeoff.

### Experiment 12: RAG vs. Pure LLM Baseline (Section 8)

Answers the fundamental question: **does retrieval actually help?** Compares five conditions on all 75 test queries, scored identically (Hit Rate: does the model's final response name the correct song?).

| Condition | How it works |
|-----------|--------------|
| Claude Haiku (no retrieval) | Raw query → Claude, no retrieved context |
| GPT-4o-mini (no retrieval) | Raw query → GPT-4o-mini, no retrieved context |
| RAG retrieval only | Best config from Experiment 1, `skip_llm=True` — reused free of charge |
| RAG + Claude Haiku | Retrieval → Claude synthesis with context |
| RAG + GPT-4o-mini | Reuses retrieval results from RAG+Claude run, calls GPT for synthesis — no second retrieval pass |

**Implementation note:** The RAG+GPT condition reuses `retrieval_results` already computed for RAG+Claude, calling GPT-4o-mini directly with the formatted context. No extra FAISS/BM25 search is performed. The `QueryPipeline` only supports `anthropic` and `ollama` backends for synthesis; GPT synthesis is handled inline in the notebook cell.

**Outputs:** Overall Hit Rate comparison bar chart + per-query-type grouped bar chart (5 bars per type) + thesis-ready table with RAG lift deltas over both pure-LLM baselines.

**Why this matters for the thesis:** Without this comparison, the evaluation only shows which retrieval technique is best — not whether retrieval is worth doing at all. The RAG lift metric directly quantifies the value added by the pipeline.

### Experiment 11: Statistical Significance Testing (Section 7.8)

Tests whether MRR improvements are statistically significant using two methods:

1. **Wilcoxon signed-rank test** — non-parametric paired test on per-query reciprocal ranks (baseline vs. best configuration). Reports p-value.
2. **Paired bootstrap resampling** — 10,000 iterations, computes 95% confidence interval for the MRR delta.

**Why it matters:** With 75 test queries, observed MRR differences could still be noise. P-values and confidence intervals preempt the "is this improvement significant?" question from thesis reviewers.

## Future Experiments

The contextual embedding ablation requires building a non-contextual index first (Phase 3 `--skip-contextual`) and has not yet been run.

### Contextual Embedding Ablation

Requires building a non-contextual index first:
```bash
.venv\Scripts\python pipeline\embedding.py --skip-contextual
```

Then re-run the full 8-configuration technique grid on the non-contextual index (`faiss_noctx.index`, `metadata_noctx.json`, `bm25_noctx.pkl`). Compare contextual vs. non-contextual MRR for each configuration. The delta directly measures the isolated contribution of contextual embeddings. Anthropic reported 35% reduction in retrieval failure; the actual gain in this lyrics domain may differ.

**Data integrity:** Both contextual and non-contextual indices must be built from the same `chunks.jsonl`. Do not re-run the chunking or ingestion pipeline between builds. Verify with `certutil -hashfile data\processed\chunks.jsonl MD5` before and after — same hash guarantees the only variable is the metadata header. The embedding script reads every chunk in file order with no filtering or randomness.
