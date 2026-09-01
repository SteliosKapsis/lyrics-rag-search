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

`TEST_SET` (notebooks/evaluation.ipynb, Section 1.1) has 95 entries total — 75 answerable with ground truth (`expected_title`, `expected_artist`, `expected_chunk_index`) plus 20 deliberately unanswerable, covering five query types:

| Type | Count | Description |
|------|-------|-------------|
| Direct lyric | 19 | User quotes or closely paraphrases actual lyrics |
| Semantic/descriptive | 19 | User describes the song's theme, mood, or genre |
| Partial recall | 19 | User vaguely remembers a lyric fragment |
| Ambiguous | 18 | Query could plausibly match multiple songs |
| Unanswerable | 20 | No correct answer exists in this corpus — tests whether the system abstains instead of guessing |

Songs in the test set are drawn from the actual indexed dataset (confirmed present in `chunks.jsonl`). Retrieval-quality metrics (Recall@N, MRR, Hit Rate) are computed only over `ANSWERABLE_TEST_SET` (the 75 answerable entries) — the unanswerable subset has no correct chunk/song to rank against, and is instead scored separately via `evaluate_abstention()` (Section 7.9), which reports what fraction of them the system correctly declines to answer.

**Version history** (`TEST_SET_VERSION`, printed with a content hash at Section 1.1's setup cell so a stale cached result can be detected by comparison):
- *Undated, pre-versioning*: expanded from 25 to 75 queries for more robust statistical significance testing and reduced variance from individual query difficulty. Queries were iteratively refined during evaluation — generic descriptions were replaced with lyrics-grounded formulations, non-English songs were removed, and duplicate entries were replaced.
- **v2.0.0**: added `expected_chunk_index` to all 75 answerable entries (chunk-level, not just song-level, ground truth — proposed by `scripts/backfill_chunk_ids.py`, a semi-automated fuzzy-match against the actual corpus, then manually reviewed against each candidate chunk's full text; 2 of the 71 non-"direct" proposals needed correcting from what the fuzzy match alone picked). Added 15 unanswerable entries (6 real songs confirmed absent from this corpus, 6 fabricated song/artist names, 3 off-topic non-song queries) so hallucination/abstention behavior is measurable for the first time — previously this golden set could not detect it at all, since every entry had a real answer.
- **v2.0.1**: replaced the mislabeled "purple rain by prince" unanswerable entry with "hey jude by the beatles" — see the "History" note under Experiment 14 below for why.
- **v2.1.0**: added 5 more real-song unanswerable entries (Comfortably Numb, Highway to Hell, Sweet Home Alabama, Piano Man, Born to Run), each verified absent by both exact-title and fuzzy/substring-title checks against `data/processed/metadata.json` — see the "Results (TEST_SET v2.1.0)" note under Experiment 14 for the check methodology and result.

Neither versioned set has an enforcement mechanism preventing someone from editing `TEST_SET` without bumping `TEST_SET_VERSION` — the version string and hash are a convention, not a guarantee.

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

**Key result:** Hybrid + Re-ranker (0.698 MRR with MiniLM) is the best configuration at default k=5, edging out Baseline + Re-ranker (0.674 MRR) and surpassing all other combinations including those with HyDE.

### Experiment 2: Top-k Sensitivity (Section 3.2)

Sweeps top_k = {3, 5, 10, 20} for Baseline and Baseline + Re-ranker to measure how the number of retrieved results affects retrieval quality.

| top_k | Baseline MRR | Baseline HR | + Re-ranker MRR | + Re-ranker HR |
|-------|-------------|-------------|-----------------|----------------|
| 3     | 0.556       | 0.587       | 0.593           | 0.600          |
| 5     | 0.571       | 0.627       | 0.674           | 0.707          |
| 10    | 0.586       | 0.693       | 0.729           | 0.773          |
| **20**| **0.596**   | **0.773**   | **0.762**       | **0.813**      |

The re-ranker's benefit scales with k: at k=20, it achieves **0.762 MRR and 81.3% Hit Rate** — the highest result in this experiment (Experiment 8's combined parameter sweep achieves higher results still — see Section 7.5). The re-ranker advantage grows from +0.037 MRR at k=3 to +0.166 at k=20, demonstrating that giving the cross-encoder more candidates consistently improves performance.

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

Goes beyond retrieval metrics to assess the full RAG pipeline (retrieval + generation) using the DeepEval framework with **two independent judges**: Claude Sonnet (Anthropic) and GPT-4o-mini (OpenAI).

**Metrics:**

| Metric | What It Measures | Why It Matters for Lyrics RAG |
|--------|-----------------|-------------------------------|
| Faithfulness | Whether the LLM response only contains claims supported by retrieved context | Detects hallucinated lyrics, fabricated metadata, or invented song attributions |
| Answer Relevancy | Whether the response actually answers the user's query | Detects off-topic or generic responses that ignore the query intent |
| Context Precision | Whether relevant chunks are ranked higher than irrelevant ones | Validates that retrieval ordering is useful (not just recall) |
| Context Recall | Whether the retrieved context covers all information needed to answer | Identifies cases where important chunks were missed by retrieval |

**Setup:**
- Uses 10 representative queries selected by type: 3 direct lyric, 3 semantic/descriptive, 2 partial recall, 2 ambiguous
- **Two judges run independently**: Claude Sonnet (`claude-sonnet-4-6`) via `AnthropicModel` and GPT-4o-mini via `GPTModel` — both using DeepEval's built-in model wrappers
- Both judges read API keys from `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
- Each query is run through the best-performing pipeline configuration from Experiment 1
- Results displayed per judge, plus an inter-judge agreement table showing scores and delta per metric

**Why two judges:** The Anthropic judge (Claude Sonnet) is a different model from the pipeline's generation model (Claude Haiku, `claude-haiku-4-5-20251001`), so there's no shared-model self-evaluation risk to begin with. Running a second judge from another vendor (GPT-4o-mini) still adds value: agreement between Claude Sonnet and GPT-4o-mini strengthens result credibility, while disagreement surfaces metrics where evaluation is more subjective or vendor-dependent.

**Why this matters for the thesis:** Retrieval metrics (Precision@k, MRR) only tell you whether the right chunks were found. LLM-as-judge metrics tell you whether the final answer is faithful, relevant, and well-grounded — completing the evaluation picture from retrieval quality through to generation quality.

## Pipeline Parameters

The evaluation notebook exercises these configurable parameters of the `QueryPipeline`. Defaults below are `QueryPipeline.query()`'s actual defaults (the ones used by the deployed app and worker) — not the CLI script's defaults, which differ (`--top-k` defaults to 5 and `--use-reranker` is off unless passed):

| Parameter | Default | Swept Values | Description |
|-----------|---------|-------------|-------------|
| `top_k` | 20 | 3, 5, 10, 20 | Number of song results to return |
| `use_hybrid` | True | True/False | Combine FAISS dense + BM25 sparse via RRF |
| `use_hyde` | False | True/False | Generate hypothetical lyrics before retrieval |
| `use_reranker` | True | True/False | Re-score top candidates with cross-encoder |
| `rrf_k` | 60 | 20, 40, 60, 80, 100 | RRF fusion constant (lower = more weight to top-ranked method) |
| `fetch_k_multiplier` | 5 | 2, 3, 5, 8 | Candidate pool size for re-ranking (fetch_k = top_k × multiplier) |
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
| LLM-as-judge scores | `notebooks/llm_judge_scores.png` | Side-by-side heatmaps of all four DeepEval metrics for Claude Sonnet and GPT-4o-mini judges |
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

1. **Cross-encoder re-ranking is the single most impactful technique.** At default k=5, Hybrid + Re-ranker (0.698 MRR) is the best configuration in the technique grid, ahead of Baseline + Re-ranker (0.674 MRR). Re-ranking alone also scales well with k — Baseline + Re-ranker reaches 0.762 MRR at k=20 in the top-k sweep. The re-ranker was only discovered to be effective in isolation after expanding the grid from 6 to 8 combinations. (Experiment 8's combined parameter sweep later finds still higher results — see Section 7.5.)

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
- **Parameter sweeps build on the technique grid's core configurations** — the top-k sweep (3.2) tests Baseline and + Re-ranker; the fetch-k sweep (3.4) tests Baseline + Re-ranker and Hybrid + Re-ranker. This isolates the effect of each parameter without re-sweeping the full 8-configuration grid every time.

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

**Why this matters for the thesis:** Without this comparison, the evaluation only shows which retrieval technique is best — not whether retrieval is worth doing at all. The RAG lift metric directly quantifies the value added by the pipeline. This experiment was added specifically to pre-empt an obvious objection to the whole project: Billboard Hot 100 lyrics are public data an LLM may have seen during pretraining, so before trusting any retrieval result it's worth asking whether retrieval is doing anything the LLM couldn't already do from memory. Rather than assume an answer, this experiment measures it directly.

**Results** (full 75-query run, `all-MiniLM-L6-v2` + Hybrid + Re-ranker as the RAG configuration):

| Condition | Hit Rate |
|---|---|
| Claude Haiku (no retrieval) | 0.133 |
| GPT-4o-mini (no retrieval) | 0.147 |
| RAG retrieval only | 0.840 |
| RAG + Claude Haiku | **0.853** |
| RAG + GPT-4o-mini | 0.800 |

RAG+Claude lift over Claude alone: **+0.720**. RAG+GPT lift over GPT alone: **+0.653**.

Per-query-type breakdown:

| Type | Claude (no ctx) | GPT (no ctx) | RAG only | RAG+Claude | RAG+GPT |
|---|---|---|---|---|---|
| direct (exact lyric fragment) | 0.316 | 0.316 | 0.895 | 0.895 | 0.895 |
| semantic | 0.105 | 0.158 | 0.789 | 0.842 | 0.737 |
| partial | 0.053 | 0.105 | 0.947 | 1.000 | 0.947 |
| ambiguous | 0.056 | 0.000 | 0.722 | 0.667 | 0.611 |

**Interpretation — the memorization concern, tested directly:** if this corpus were meaningfully present in the LLMs' training data, the no-retrieval baselines would score far higher than this, especially on "direct" queries (literal lyric fragments — the type most vulnerable to memorization). Instead, pure Claude/GPT-4o-mini identify the correct song only ~13-15% of the time overall, and only 31.6% even on direct fragments; GPT-4o-mini scores exactly 0.000 on "ambiguous" queries without retrieval. A 70-plus point gap between the no-retrieval and RAG conditions is too large to be explained by output-format mismatches in the no-retrieval scoring (`_is_hit_raw` requires an exact `Title by Artist` match, so some of the 13-15% floor may undercount near-misses — but not by 70 points). This was a deliberate, pre-registered check, not an afterthought: the concern that a public-lyrics corpus might be "too easy" for an LLM to already know was anticipated before this experiment was designed, and the experiment was built specifically to test it empirically rather than argue about it. That the result is a strong, clean rejection of the concern is itself worth stating plainly in the thesis's Results/Discussion chapter, framed exactly that way — an anticipated risk that was tested, not assumed away.

### Experiment 11: Statistical Significance Testing (Section 7.8)

Tests whether MRR improvements are statistically significant using two methods:

1. **Wilcoxon signed-rank test** — non-parametric paired test on per-query reciprocal ranks (baseline vs. best configuration). Reports p-value.
2. **Paired bootstrap resampling** — 10,000 iterations, computes 95% confidence interval for the MRR delta.

**Why it matters:** With 75 test queries, observed MRR differences could still be noise. P-values and confidence intervals preempt the "is this improvement significant?" question from thesis reviewers.

### Experiment 13: Contextual Embedding Ablation (Section 7.6)

Compares retrieval quality between **contextual** (metadata header prepended before embedding) and **non-contextual** (raw lyrics only) embeddings, using a non-contextual index built via:
```bash
.venv\Scripts\python pipeline\embedding.py --skip-contextual
```
producing `faiss_noctx.index`, `metadata_noctx.json`, and `bm25_noctx.pkl`. The full technique grid (Baseline, + Re-ranker, + Hybrid, + Hybrid + Re-ranker) was run on both the contextual and non-contextual indices for all-MiniLM-L6-v2 and bge-small-en-v1.5 to isolate the contribution of contextual embeddings.

**Results (MRR at k=5):**

| Embedding Model | Technique | Contextual | Non-contextual | Delta |
|-----------------|-----------|-----------|-----------------|-------|
| all-MiniLM-L6-v2 | Baseline (FAISS only) | 0.571 | 0.471 | +0.100 |
| all-MiniLM-L6-v2 | + Re-ranker | 0.674 | 0.598 | +0.077 |
| all-MiniLM-L6-v2 | + Hybrid | 0.590 | 0.513 | +0.077 |
| all-MiniLM-L6-v2 | + Hybrid + Re-ranker | 0.698 | 0.684 | +0.013 |
| bge-small-en-v1.5 | Baseline (FAISS only) | 0.476 | 0.558 | -0.082 |
| bge-small-en-v1.5 | + Re-ranker | 0.580 | 0.683 | -0.103 |
| bge-small-en-v1.5 | + Hybrid | 0.549 | 0.556 | -0.007 |
| bge-small-en-v1.5 | + Hybrid + Re-ranker | 0.709 | 0.653 | +0.056 |

Average MRR delta (contextual − non-contextual): **+0.067** for all-MiniLM-L6-v2 — contextual embeddings help consistently, most strongly at the unassisted baseline. For bge-small-en-v1.5, the average delta is **-0.034**: contextual embeddings actually *hurt* the Baseline and + Re-ranker configurations, and only pull ahead once combined with Hybrid + Re-ranker. Unlike Anthropic's reported 35% reduction in retrieval failure, the benefit of contextual embeddings in this lyrics domain is model- and configuration-dependent rather than a uniform win.

**Data integrity:** Both indices were built from the same `chunks.jsonl` without re-running chunking or ingestion between builds, verified via `certutil -hashfile data\processed\chunks.jsonl MD5` before and after (identical hash confirms the metadata header was the only variable). The embedding script reads every chunk in file order with no filtering or randomness.

Plot saved to `notebooks/contextual_ablation.png`.

### Experiment 14: Abstention Rate on Unanswerable Queries (Section 7.9)

Measures whether `pipeline/query.py`'s retrieval-gated abstention (added in the thesis-critical fix pass — see `Documentation/phase_4_query_pipeline.md`'s "gated by retrieval confidence" section) actually fires on the `UNANSWERABLE_QUERIES` entries added to the golden set (15 in v2.0.0, expanded to 20 in v2.1.0).

Unlike every other experiment in this notebook, this one calls `pipeline.query()` in full (not `skip_llm=True`) for each unanswerable entry, since the abstention decision is made inside `query()`/`query_stream()` itself — a retrieval-only run has nothing to observe it with. Reports `abstention_rate = fraction of unanswerable queries where LLMResponse.abstained is True`, plus a per-query breakdown flagging every case that did **not** abstain (a false negative for the gate — the pipeline confidently or not answered a query with no correct answer in this corpus).

**Why it matters for the thesis:** this is the first experiment in this notebook that can say anything about hallucination behavior at all. Every other experiment's queries have a real answer; `abstention_rate` is the one number that reflects whether the system knows what it doesn't know. It is also the concrete measurement needed to actually tune `abstain_score_threshold` (currently a documented placeholder, `0.3`) — sweeping the threshold against this experiment's output is the follow-up this notebook makes possible but does not itself perform.

**Results (TEST_SET v2.0.1, re-run after the golden-set fix below):** 15 of 15 unanswerable queries triggered abstention — **abstention rate 1.00**. The mix covers fabricated song/artist names, real well-known songs confirmed absent from this Billboard-derived corpus (e.g. "Bohemian Rhapsody" attributed to the wrong artist, "Stairway to Heaven" by Led Zeppelin specifically, "Hotel California," "Hey Jude"), and out-of-scope non-song queries ("what's the weather like today"). No false negatives in this run.

**Results (TEST_SET v2.1.0):** the unanswerable set was expanded from 15 to 20 entries — 5 more real, well-known songs (Comfortably Numb, Highway to Hell, Sweet Home Alabama, Piano Man, Born to Run) verified absent from the corpus. Each was checked two ways before being added: an exact title match against `data/processed/metadata.json` under any artist, and a fuzzy/substring title check specifically to catch a cover or medley smuggling the song's actual lyrics in under a different artist credit — the same failure mode that invalidated the original "purple rain by prince" entry. That second check caught a real near-miss and kept it out of the set: a Will to Power track titled "Baby I Love Your Way/Freebird Medley" contains genuine "Free Bird" (Lynyrd Skynyrd) lyrics across three of its chunks, so "Free Bird" was excluded as not actually unanswerable rather than added. Re-running the full 20-query abstention experiment produced **20 of 20 abstentions — abstention rate 1.00**, with no false negatives, including on all 5 new entries.

**History — a corrected golden-set entry (post-deployment verification):** the original v2.0.0 run reported 14/15 (0.933), with the one non-abstaining case — **"purple rain by prince"** — documented at the time as a threshold-tuning false negative. It wasn't. Direct lookup against `data/processed/metadata.json`, confirmed live against the running app, showed "Purple Rain" genuinely exists in this corpus — credited to **"Prince and the Revolution,"** the actual official artist credit for that single, not "Prince" alone. The pipeline had retrieved a real match and correctly did not abstain; the golden-set entry's `expected_artist: "Prince"` was what was wrong, not the gate. This is the same class of issue "Stairway to Heaven" surfaced during manual testing (real songs charting under a credited-artist name that doesn't match the name most people would guess) — a golden-set labeling risk worth checking title-only across all artist credits, not just the exact expected title+artist tuple, when curating unanswerable entries. The entry was replaced with "hey jude by the beatles" (verified to have zero title matches under any artist) in TEST_SET v2.0.1, and re-running the experiment against the corrected set produced the clean 1.00 above — confirming the gate itself had no real false negative, only the golden set did. This doesn't remove the need to eventually tune `abstain_score_threshold` against real data on a larger unanswerable set — a 15-query set hitting 1.00 is a small-sample result, not proof the threshold is well-calibrated — but this specific data point is no longer evidence that the placeholder is failing.

**Additional live-testing evidence for the threshold-tuning case ("Stan" by Eminem):** a four-query manual test sequence against the same known-present song (confirmed present in the corpus, 11 total chunks) shows the reranker score — and therefore the abstain/answer decision — swinging sharply based on query *phrasing style*, not just topical correctness:

| # | Query style | Reranker score | Outcome |
|---|---|---|---|
| 1 | Vague narrative paraphrase, no artist named | -3.726 | Abstained (FAISS still found the right song at 0.510, but reranker score fell well below the 0.3 threshold) |
| 2 | Same paraphrase + "by eminem" | -2.777 | Still abstained — closer to the threshold, but adding the correct artist name alone didn't close the gap |
| 3 | Direct verbatim lyric quote | high confidence | Confident, correct match |
| 4 | Detailed narrative paraphrase (no quote, but rich plot-specific detail) | high confidence | Confident, correct match |

The pattern: a cross-encoder reranker scores *text similarity* to the chunk, not narrative/thematic accuracy — a heavily paraphrased description of a song's plot can correctly identify the song semantically while still scoring far below threshold on the reranker, because the query text and the chunk's actual lyric text don't overlap much lexically. Naming the artist alone (query 2) didn't fix this; the query needed either a verbatim quote or heavy specific plot detail to score well. This is concrete, reproducible evidence for the same conclusion as the honest-limits note above: the placeholder threshold is calibrated for nothing in particular, and a real sweep would need to account for phrasing style, not just query type (direct/semantic/partial/ambiguous), as a variable that affects reranker score.

**A second, independent finding from the same test sequence — a `verify_citations()` false flag:** query 3's response was a correct, high-confidence match with a verbatim-quoted `relevant_lyric`, yet the UI still showed "chunk #11 — could not verify this citation against the source chunk." This is a false negative for the citation-verification heuristic itself (see `Documentation/phase_4_query_pipeline.md`), not the retrieval or the answer — worth citing alongside the documented "citation linking, not full verification by entailment" limitation as a concrete instance where the `difflib`-based fuzzy match under-scored a citation that was, on inspection, accurate.
