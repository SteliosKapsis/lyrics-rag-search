# API Reference

Complete reference for all pipeline classes, methods, and CLI scripts.

---

## `pipeline/query.py`

### Pydantic Models

#### `SongMatch`

Represents a single matched song in the LLM response.

| Field | Type | Description |
|---|---|---|
| `title` | `str` | Song title |
| `artist` | `str` | Artist name |
| `album` | `str \| None` | Album name (optional) |
| `release_date` | `str \| None` | ISO-format date string (optional) |
| `relevant_lyric` | `str` | Excerpt from the retrieved chunk most relevant to the query |
| `explanation` | `str` | LLM-generated explanation of why this song matches |

#### `LLMResponse`

Structured output from LLM synthesis.

| Field | Type | Description |
|---|---|---|
| `matches` | `list[SongMatch]` | Ranked list of matched songs |
| `confidence` | `"high" \| "medium" \| "low"` | Overall retrieval confidence |
| `summary` | `str` | 1–2 sentence answer summarizing the result |

---

### `class QueryPipeline`

Main class for the end-to-end RAG pipeline. Instantiate once and call `query()` or `query_stream()` repeatedly.

#### `__init__`

```python
QueryPipeline(
    index_path: str | None = None,
    metadata_path: str | None = None,
    bm25_path: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str | None = None,
    llm_backend: str = "anthropic",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
```

| Parameter | Default | Description |
|---|---|---|
| `index_path` | `data/processed/faiss.index` | Path to FAISS index file |
| `metadata_path` | `data/processed/metadata.json` | Path to metadata JSON (positionally aligned with FAISS) |
| `bm25_path` | `data/processed/bm25.pkl` | Path to BM25 pickle (optional; hybrid degrades gracefully if missing) |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Model name for query-time embedding. Must match the model used at index time |
| `llm_model` | `None` | Specific LLM model. Defaults: `claude-haiku-4-5-20251001` (Anthropic), `llama3` (Ollama) |
| `llm_backend` | `"anthropic"` | LLM backend: `"anthropic"` or `"ollama"` |
| `reranker_model` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model for re-ranking |

Loads FAISS index, metadata JSON, and BM25 index on initialization. Cross-encoder is loaded lazily on first use.

---

#### `query`

Full non-streaming RAG pipeline.

```python
def query(
    query: str,
    top_k: int = 20,
    use_hyde: bool = False,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    rrf_k: int = 60,
    fetch_k_multiplier: int = 5,
) -> dict
```

| Parameter | Default | Description |
|---|---|---|
| `query` | — | Natural language search query |
| `top_k` | 20 | Number of songs to return |
| `use_hyde` | False | Generate hypothetical lyric before retrieval (not recommended for lyrics) |
| `use_hybrid` | True | Combine FAISS dense + BM25 sparse via RRF fusion |
| `use_reranker` | True | Re-score candidates with cross-encoder |
| `rrf_k` | 60 | RRF fusion constant. Lower = more weight to top-ranked items |
| `fetch_k_multiplier` | 5 | Candidate pool multiplier for re-ranking (`fetch_k = top_k × multiplier`) |

**Returns**:
```python
{
    "query": str,
    "retrieval_results": list[dict],    # grouped by song, sorted by best_score
    "llm_response": LLMResponse,        # Pydantic model
    "hyde_hypothesis": str | None,
}
```

Each element of `retrieval_results`:
```python
{
    "title": str,
    "artist": str,
    "album": str | None,
    "release_date": str | None,
    "best_score": float,                # max score across all chunks
    "chunks": [
        {
            "text": str,
            "score": float,             # FAISS inner product
            "chunk_index": int,
            "bm25_score": float,        # present if hybrid enabled
            "rrf_score": float,         # present if hybrid enabled
            "cross_encoder_score": float # present if reranker enabled
        }
    ]
}
```

---

#### `query_stream`

Streaming version of the full RAG pipeline. Yields retrieval results immediately, then streams LLM tokens.

```python
def query_stream(
    query: str,
    top_k: int = 20,
    use_hyde: bool = False,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    rrf_k: int = 60,
    fetch_k_multiplier: int = 5,
) -> Generator[dict | str, None, None]
```

Same parameters as `query()`.

**Yields** (in order):
1. A `dict` with keys `retrieval_results` and `hyde_hypothesis` (same structure as `query()` return)
2. `str` tokens — JSON string fragments from the LLM. Concatenate all tokens and parse with `LLMResponse.model_validate_json()` or `json.loads()`

---

#### `retrieve`

Dense-only FAISS retrieval.

```python
def retrieve(query: str, top_k: int = 5) -> list[dict]
```

Returns chunk dicts with `score` (FAISS inner product) field added.

---

#### `retrieve_bm25`

Sparse keyword retrieval.

```python
def retrieve_bm25(query: str, top_k: int = 5) -> list[dict]
```

Returns chunk dicts with `bm25_score` field added. Returns empty list if BM25 index not loaded.

---

#### `retrieve_hybrid`

RRF-fused dense + sparse retrieval.

```python
def retrieve_hybrid(query: str, top_k: int = 5, rrf_k: int = 60) -> list[dict]
```

Returns chunk dicts with `score`, `bm25_score`, and `rrf_score` fields added.

---

#### `group_by_song`

Groups a flat list of chunks into per-song aggregates.

```python
def group_by_song(results: list[dict]) -> list[dict]
```

Returns song group dicts sorted by `best_score` descending.

---

#### `synthesize`

Calls the LLM with retrieved context and returns a structured response.

```python
def synthesize(query: str, grouped_results: list[dict]) -> LLMResponse
```

---

### CLI Usage

```bash
.venv\Scripts\python pipeline\query.py "I stay out too late got nothin in my brain" [OPTIONS]

Options:
  --top-k INT           Number of chunks to retrieve (default: 5)
  --embedding-model STR Embedding model name (default: all-MiniLM-L6-v2)
  --use-hyde            Enable HyDE hypothesis generation
  --no-hybrid           Disable BM25 + RRF (use FAISS only)
  --use-reranker        Enable cross-encoder re-ranking
  --reranker-model STR  Cross-encoder model (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
  --llm-backend STR     LLM backend: anthropic or ollama (default: anthropic)
  --llm-model STR       Specific LLM model override
```

---

## `pipeline/reranker.py`

### `class Reranker`

Cross-encoder re-ranker as a second retrieval stage.

#### `__init__`

```python
Reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2")
```

Loads cross-encoder model on instantiation. Supported models:

| Model | Size | Speed | Quality |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~130MB | Medium | Better |
| `BAAI/bge-reranker-base` | ~280MB | Slow | Best |

#### `rerank`

```python
def rerank(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]
```

| Parameter | Description |
|---|---|
| `query` | The original user query (not HyDE hypothesis) |
| `chunks` | List of chunk dicts, each must have a `"text"` field |
| `top_k` | If set, return only the top N chunks after re-ranking |

**Returns**: Same chunk dicts with `cross_encoder_score` field added, sorted by score descending.

---

## `pipeline/chunking.py`

### Functions

#### `chunk_song`

```python
def chunk_song(
    lyrics: str,
    max_chunk_size: int = 400,
    min_chunk_size: int = 80,
    overlap_lines: int = 0,
) -> list[str]
```

Splits a single song's lyrics into chunks. Returns list of chunk text strings (with section headers prepended).

#### `process_songs`

```python
def process_songs(
    songs: list[dict],
    max_chunk_size: int,
    min_chunk_size: int,
    overlap_lines: int,
) -> list[dict]
```

Processes all songs in the dataset. Returns list of chunk dicts with full metadata.

### CLI Usage

```bash
.venv\Scripts\python pipeline\chunking.py [OPTIONS]

Options:
  --input PATH          Input cleaned lyrics JSON (default: data/raw/cleaned_lyrics.json)
  --output PATH         Output chunks JSONL (default: data/processed/chunks.jsonl)
  --max-chunk-size INT  Max chunk size in characters (default: 400)
  --min-chunk-size INT  Min chunk size before merging (default: 80)
  --overlap-lines INT   Lines from previous chunk to prepend as overlap (default: 0)
```

**Variant indices** (for Experiment 5 — chunking sweep):
```bash
.venv\Scripts\python pipeline\chunking.py --max-chunk-size 200 --output data/processed/chunks_c200_o0.jsonl
.venv\Scripts\python pipeline\chunking.py --max-chunk-size 600 --output data/processed/chunks_c600_o0.jsonl
.venv\Scripts\python pipeline\chunking.py --overlap-lines 1   --output data/processed/chunks_c400_o1.jsonl
.venv\Scripts\python pipeline\chunking.py --overlap-lines 2   --output data/processed/chunks_c400_o2.jsonl
```

---

## `pipeline/embedding.py`

### Functions

#### `build_contextual_text`

```python
def build_contextual_text(chunk: dict) -> str
```

Prepends a metadata header to the chunk text:
```
Song: 'Title' by Artist. Album: Album. Released: 2021-01-01. Section: [Chorus].
[Chorus]
<original chunk text>
```

#### `embed_chunks`

```python
def embed_chunks(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray
```

Returns L2-normalized embedding matrix of shape `(n_texts, embedding_dim)`.

#### `embed_chunks_openai`

```python
def embed_chunks_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,
) -> np.ndarray
```

Requires `OPENAI_API_KEY` env var. Truncates texts to 8192 tokens via tiktoken. Returns L2-normalized embeddings.

#### `build_faiss_index`

```python
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP
```

Creates and populates an exact inner-product FAISS index.

#### `build_bm25_index`

```python
def build_bm25_index(texts: list[str]) -> BM25Okapi
```

Tokenizes texts as lowercase + whitespace split, builds BM25Okapi index.

### CLI Usage

```bash
.venv\Scripts\python pipeline\embedding.py [OPTIONS]

Options:
  --input PATH            Input JSONL (default: data/processed/chunks.jsonl)
  --index-output PATH     FAISS index output (default: data/processed/faiss.index)
  --metadata-output PATH  Metadata JSON output (default: data/processed/metadata.json)
  --bm25-output PATH      BM25 pickle output (default: data/processed/bm25.pkl)
  --model STR             Embedding model (default: all-MiniLM-L6-v2)
                          Options: all-mpnet-base-v2, BAAI/bge-small-en-v1.5
  --batch-size INT        Embedding batch size (default: 64)
  --skip-contextual       Embed raw text without metadata header (ablation)
  --openai                Use OpenAI text-embedding-3-small instead of local model
```

**Output file auto-suffixing**:

The script automatically adjusts output filenames based on model and flags, so you can build all indices without manually specifying paths:

```bash
# Builds: faiss.index, metadata.json, bm25.pkl (MiniLM, contextual)
.venv\Scripts\python pipeline\embedding.py

# Builds: faiss_bge.index, metadata_bge.json, bm25_bge.pkl
.venv\Scripts\python pipeline\embedding.py --model BAAI/bge-small-en-v1.5

# Builds: faiss_openai.index, metadata_openai.json, bm25_openai.pkl
.venv\Scripts\python pipeline\embedding.py --openai

# Builds: faiss_noctx.index, metadata_noctx.json, bm25_noctx.pkl
.venv\Scripts\python pipeline\embedding.py --skip-contextual

# Builds: faiss_bge_noctx.index, ...
.venv\Scripts\python pipeline\embedding.py --model BAAI/bge-small-en-v1.5 --skip-contextual
```

---

## `collection/ingest_lyrics.py`

### CLI Usage

```bash
.venv\Scripts\python collection\ingest_lyrics.py [OPTIONS]

Required:
  --input PATH      CSV file with song/title and artist columns

Options:
  --output PATH     Output lyrics JSON (default: data/raw/lyrics.json)
  --failures PATH   Failed fetches CSV (default: data/failed/failed_fetches.csv)
  --token STR       Genius API token (or set GENIUS_API_TOKEN env var)
  --delay FLOAT     Seconds between requests (default: 1.5)
```

**Resume behavior**: On startup, reads existing `--output` file and skips already-fetched songs. Safe to interrupt and restart.

---

## `collection/clean_lyrics.py`

### CLI Usage

```bash
.venv\Scripts\python collection\clean_lyrics.py [OPTIONS]

Options:
  --input PATH    Raw lyrics JSON (default: data/raw/lyrics.json)
  --output PATH   Cleaned lyrics JSON (default: data/raw/cleaned_lyrics.json)
  --report PATH   Validation report CSV (default: data/failed/validation_report.csv)
```

---

## Environment Variables

| Variable | Required | Used by | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | `query.py`, `worker.py` | Anthropic API key for Claude |
| `OPENAI_API_KEY` | For OpenAI model | `embedding.py`, `query.py`, `worker.py` | OpenAI API key for `text-embedding-3-small` |
| `GENIUS_API_TOKEN` | For ingestion | `ingest_lyrics.py` | Genius API token for lyrics fetching |
| `RABBITMQ_URL` | For queue path | `app.py`, `worker.py` | AMQP connection string. If unset, app uses direct streaming |
| `OLLAMA_HOST` | For Ollama backend | `query.py` | Ollama server URL (default: `http://localhost:11434`) |
| `HF_HOME` | Docker only | sentence-transformers | HuggingFace model cache directory |

---

## Data File Reference

| File | Produced by | Consumed by | Description |
|---|---|---|---|
| `charts.csv` | (manual) | `ingest_lyrics.py` | Billboard Hot 100 chart data |
| `data/raw/lyrics.json` | `ingest_lyrics.py` | `clean_lyrics.py` | Raw lyrics + metadata from Genius |
| `data/raw/cleaned_lyrics.json` | `clean_lyrics.py` | `chunking.py` | Cleaned and validated lyrics |
| `data/processed/chunks.jsonl` | `chunking.py` | `embedding.py` | Lyrics chunks with metadata (JSONL) |
| `data/processed/faiss*.index` | `embedding.py` | `query.py` | FAISS dense vector index |
| `data/processed/metadata*.json` | `embedding.py` | `query.py` | Chunk metadata (positionally aligned with FAISS) |
| `data/processed/bm25*.pkl` | `embedding.py` | `query.py` | BM25 sparse index (pickle) |
| `data/failed/failed_fetches.csv` | `ingest_lyrics.py` | (manual review) | Songs that could not be fetched |
| `data/failed/validation_report.csv` | `clean_lyrics.py` | (manual review) | Flagged entries after cleaning |
