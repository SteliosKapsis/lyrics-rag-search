# RabbitMQ Worker Integration

## Overview

The RabbitMQ integration decouples query processing from the Streamlit frontend using the AMQP RPC pattern. In Docker deployment, the app publishes queries to a message queue and a separate worker container processes them asynchronously. This architecture enables horizontal scaling (multiple workers) and isolates the heavy ML workload from the web server.

In local development, the RabbitMQ path is bypassed entirely — the app calls `QueryPipeline` directly via streaming. No configuration change is needed to switch between modes.

---

## When Each Path Is Used

| Environment | `RABBITMQ_URL` set? | Path used |
|---|---|---|
| Local dev (`.venv\Scripts\streamlit run app\app.py`) | No | Direct streaming (`query_stream`) |
| Docker (`docker compose up`) | Yes (injected by docker-compose) | RabbitMQ worker queue |
| Local dev with RabbitMQ | Yes (in `.env`) | RabbitMQ worker queue |

---

## RPC Pattern

The integration uses AMQP's standard RPC (Remote Procedure Call) pattern:

```
App                              RabbitMQ                         Worker
 │                                  │                                │
 │── publish(query) ───────────────►│                                │
 │   routing_key: "rag.queries"     │                                │
 │   reply_to: "amq.gen-XYZ"       │                                │
 │   correlation_id: "uuid-1234"    │── deliver ────────────────────►│
 │                                  │                                │
 │                                  │   (worker runs pipeline)       │
 │                                  │                                │
 │                                  │◄── publish(result) ────────────│
 │                                  │   routing_key: "amq.gen-XYZ"  │
 │◄── deliver(result) ─────────────│   correlation_id: "uuid-1234"  │
 │   (matched by correlation_id)    │                                │
```

Key properties:
- **`reply_to`**: An exclusive, auto-delete queue created per request. Only the requesting app instance reads from it.
- **`correlation_id`**: UUID generated per request. Used to match responses to requests, enabling multiple in-flight requests from the same process.
- **`delivery_mode: 2`** on requests (persistent): Survives RabbitMQ restart.
- **`delivery_mode: 1`** on responses (non-persistent): Ephemeral reply, no need to persist.

---

## Message Formats

### Request (App → Queue)

```json
{
  "query": "I stay out too late got nothin in my brain",
  "embedding_model": "text-embedding-3-small",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "top_k": 20,
  "use_hyde": false,
  "use_hybrid": true,
  "use_reranker": true,
  "rrf_k": 60,
  "fetch_k_multiplier": 5
}
```

### Response (Worker → Reply Queue)

On success:
```json
{
  "query": "I stay out too late...",
  "retrieval_results": [...],
  "hyde_hypothesis": null,
  "llm_response": {
    "matches": [
      {
        "title": "Shake It Off",
        "artist": "Taylor Swift",
        "album": "1989",
        "release_date": "2014-10-27",
        "relevant_lyric": "[Verse 1]\nI stay out too late...",
        "explanation": "The query directly quotes the opening verse."
      }
    ],
    "confidence": "high",
    "summary": "This is 'Shake It Off' by Taylor Swift (2014)."
  }
}
```

Response AMQP headers:
- `status: "ok"` — successful processing
- `status: "error"` — pipeline raised an exception (body contains `{"error": "..."}`)

---

## Worker Design

### Pipeline Caching

The `QueryPipeline` is expensive to instantiate (~5–30s depending on model):
- FAISS index load: reads the full index from disk into memory
- Sentence-transformer model load: downloads (first time) and loads ~80–420MB model weights

The worker caches pipeline instances in a module-level dict keyed by `(embedding_model, reranker_model)`. The first query for a given model combination pays the load cost; subsequent queries reuse the cached instance.

```python
_pipeline_cache: dict[tuple, QueryPipeline] = {}

def get_pipeline(embedding_model, reranker_model) -> QueryPipeline:
    key = (embedding_model, reranker_model)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = QueryPipeline(...)  # expensive
    return _pipeline_cache[key]
```

### QoS: One Query at a Time

```python
channel.basic_qos(prefetch_count=1)
```

The worker processes one message at a time. A second query is not delivered until the first is acknowledged. This prevents out-of-memory errors when multiple large FAISS searches run concurrently. To scale throughput, run additional worker containers — each maintains its own pipeline cache.

### Serialization

`QueryPipeline.query()` returns `LLMResponse` as a Pydantic model. Before publishing, the worker converts it to a plain dict:

```python
"llm_response": result["llm_response"].model_dump()
```

The app reconstructs it on receipt:

```python
llm_response = LLMResponse.model_validate(result["llm_response"])
```

### Connection Retry on Startup

The worker retries the RabbitMQ connection up to 15 times with 3-second delays. This handles the container startup race condition where the worker starts before RabbitMQ has fully initialized (even with healthcheck, there's a brief window).

---

## Docker Compose Configuration

```yaml
rabbitmq:
  image: rabbitmq:3-management
  ports:
    - "5672:5672"     # AMQP protocol
    - "15672:15672"   # Management UI
  healthcheck:
    test: ["CMD", "rabbitmq-diagnostics", "ping"]
    interval: 10s
    timeout: 5s
    retries: 10
    start_period: 20s

worker:
  build: .
  entrypoint: ["python", "worker/worker.py"]   # overrides Streamlit entrypoint
  env_file: .env
  environment:
    - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
  volumes:
    - ./data/processed:/app/data/processed:ro
    - hf-cache:/app/.cache/huggingface
  depends_on:
    rabbitmq:
      condition: service_healthy

app:
  # ...
  environment:
    - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
  depends_on:
    rabbitmq:
      condition: service_healthy
    worker:
      condition: service_started
```

Both `app` and `worker` are built from the **same Docker image**. The worker overrides the entrypoint to run `worker/worker.py` instead of the Streamlit server.

---

## Running Locally with RabbitMQ

To test the RabbitMQ path without Docker:

**1. Start RabbitMQ:**
```bash
docker run -d --name rabbit \
  -p 5672:5672 -p 15672:15672 \
  rabbitmq:3-management
```

**2. Add to `.env`:**
```
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
```

**3. Start worker (Terminal 1):**
```bash
.venv\Scripts\python worker\worker.py
```

**4. Start app (Terminal 2):**
```bash
.venv\Scripts\streamlit run app\app.py
```

The app will detect `RABBITMQ_URL` and route queries through the worker.

---

## Monitoring

The RabbitMQ management UI is available at `http://localhost:15672` (credentials: `guest` / `guest`).

Useful views:
- **Queues → `rag.queries`**: Monitor message rate, depth, consumer count
- **Connections**: Verify app and worker connections
- **Overview**: Throughput charts

The `rag.queries` queue is declared with `durable=True` so it survives RabbitMQ restart. Reply queues are exclusive and auto-delete when the app's connection closes.

---

## Scaling Workers

To run multiple workers (for higher concurrent throughput):

```bash
# In docker-compose.yml, scale the worker service:
docker compose up --scale worker=3
```

Each worker instance maintains its own pipeline cache. The first query to each instance still pays the load cost; subsequent queries to the same instance are fast. RabbitMQ distributes messages round-robin across consumers.

Note: Multiple workers each load a full FAISS index into memory. With the default MiniLM index (~175K × 384 = ~270MB), 3 workers consume ~810MB for FAISS alone plus sentence-transformer weights.

---

## Timeout Behavior

The app waits up to **120 seconds** for a worker response before raising `TimeoutError`. This covers:
- Cold start (first query to a worker that hasn't loaded its pipeline yet): ~5–30s
- Normal query processing: ~1–5s
- Network overhead: negligible (same Docker network)

If the worker is not running or RabbitMQ is unavailable, the app shows `"Worker queue error: ..."` in the UI.
