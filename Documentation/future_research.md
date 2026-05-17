# Future Research Directions

Additional experiments and improvements identified during development that would strengthen the thesis evaluation or extend the pipeline's capabilities.

## Implemented Items

The following items have been fully implemented in code and are ready to produce results once the required index builds complete.

### 1. OpenAI `text-embedding-3-small` Comparison

**Status:** Implemented. Code added to `embedding.py` (`--openai` flag), `query.py` (OpenAI query-time embedding), `app.py` (model selector), and `evaluation.ipynb` (auto-detection of `faiss_openai.index`). Requires running `embedding.py --openai` to build the index (~2-3 hours, API-bound).

**What:** OpenAI's cloud-based embedding model (`text-embedding-3-small`, 1536 dimensions) as a third embedding option alongside the two local sentence-transformer models (MiniLM 384d, BGE 384d). The evaluation notebook auto-detects the OpenAI index and runs the full technique grid against it.

**Why:** Provides a local vs. cloud embedding quality comparison. OpenAI's model is trained on a much larger corpus and produces higher-dimensional vectors, but requires API access, costs money per token, and adds latency.

### 2. Per-Query-Type Breakdown

**Status:** Implemented. Added as Section 3.1.1 in the evaluation notebook.

**What:** MRR and Hit Rate broken down by query type (direct lyric, semantic, partial recall, ambiguous) for each technique configuration. Displayed as pivot tables and heatmaps.

**Why:** Reveals which techniques help which kinds of queries.

### 3. Statistical Significance Testing

**Status:** Implemented. Added as Section 7.8 in the evaluation notebook.

**What:** Wilcoxon signed-rank test (non-parametric, paired) and paired bootstrap resampling (10,000 iterations) comparing the baseline vs. best configuration. Reports p-values and 95% confidence intervals for the MRR delta.

**Why:** Determines whether observed MRR improvements are statistically significant or could be due to chance.

### 4. Latency Profiling

**Status:** Implemented. Added as Section 7.7 in the evaluation notebook.

**What:** End-to-end latency breakdown by pipeline stage (query embedding, FAISS search, BM25 search, RRF fusion, cross-encoder re-ranking) across four configurations (Baseline, Baseline+RR, Hybrid, Hybrid+RR). Uses 5 representative queries and produces a stacked bar chart.

**Why:** The accuracy-latency tradeoff is central to choosing a production configuration.

### 5. Larger Test Set

**Status:** Implemented. Test set expanded from 25 to 75 queries in the evaluation notebook (cell 4).

**What:** 75 evaluation queries balanced across four types: 19 direct lyric, 19 semantic, 19 partial recall, 18 ambiguous. All songs verified to exist in the indexed dataset.

**Why:** 75 queries provides much more robust statistical significance testing and reduces variance from individual query difficulty.

## Remaining Items (Requires Index Builds)

### 6. Chunk Size Experiments

**Status:** Framework implemented in evaluation notebook (Experiment 5, cell 23). Requires building variant indices.

**What:** Test different chunk size and overlap configurations to find the optimal chunking parameters for lyrics retrieval. The evaluation notebook already has the sweep structure and auto-detects available indices.

**Why:** The current default (80-400 character chunks, section-aware splitting) was chosen based on lyrics structure heuristics, not empirical evaluation. Different chunk sizes affect retrieval differently: smaller chunks are more precise but lose context, larger chunks capture more context but dilute the embedding signal.

**Implementation notes:**
- Requires re-running `pipeline/chunking.py` with different `--max-chunk-size` parameters, then `pipeline/embedding.py` for each
- Convention for output files: `faiss_c{size}_o{overlap}.index`, `metadata_c{size}_o{overlap}.json`, `bm25_c{size}_o{overlap}.pkl`
- Suggested configurations: chunk sizes 200, 400 (default), 600 characters; overlap 0, 1, 2 lines
- Each configuration requires a full re-chunk + re-embed cycle (~12 hours per configuration for embedding)
- Total compute: 5 configurations x ~12 hours = ~60 hours (run in parallel if possible)
- The evaluation notebook auto-detects available chunking indices and runs the sweep on those present

**Build commands:**
```bash
# c200_o0
.venv\Scripts\python pipeline\chunking.py --max-chunk-size 200 --overlap-lines 0 --output data/processed/chunks_c200_o0.jsonl
.venv\Scripts\python pipeline\embedding.py --input data/processed/chunks_c200_o0.jsonl --index-output data/processed/faiss_c200_o0.index --metadata-output data/processed/metadata_c200_o0.json --bm25-output data/processed/bm25_c200_o0.pkl

# c600_o0
.venv\Scripts\python pipeline\chunking.py --max-chunk-size 600 --overlap-lines 0 --output data/processed/chunks_c600_o0.jsonl
.venv\Scripts\python pipeline\embedding.py --input data/processed/chunks_c600_o0.jsonl --index-output data/processed/faiss_c600_o0.index --metadata-output data/processed/metadata_c600_o0.json --bm25-output data/processed/bm25_c600_o0.pkl

# c400_o1
.venv\Scripts\python pipeline\chunking.py --max-chunk-size 400 --overlap-lines 1 --output data/processed/chunks_c400_o1.jsonl
.venv\Scripts\python pipeline\embedding.py --input data/processed/chunks_c400_o1.jsonl --index-output data/processed/faiss_c400_o1.index --metadata-output data/processed/metadata_c400_o1.json --bm25-output data/processed/bm25_c400_o1.pkl

# c400_o2
.venv\Scripts\python pipeline\chunking.py --max-chunk-size 400 --overlap-lines 2 --output data/processed/chunks_c400_o2.jsonl
.venv\Scripts\python pipeline\embedding.py --input data/processed/chunks_c400_o2.jsonl --index-output data/processed/faiss_c400_o2.index --metadata-output data/processed/metadata_c400_o2.json --bm25-output data/processed/bm25_c400_o2.pkl
```

**Relevant phases:** Phase 2 (chunking), Phase 3 (embedding), Phase 6 (evaluation)
