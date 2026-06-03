"""
RabbitMQ worker for the lyrics RAG pipeline.

Consumes query requests from the 'rag.queries' queue, runs the full
QueryPipeline, and publishes the result back via the RPC reply queue.

Usage (from project root):
    .venv/Scripts/python worker/worker.py
"""

import json
import os
import sys
import time
from pathlib import Path

import pika

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.query import QueryPipeline  # noqa: E402

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
QUEUE = "rag.queries"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "",
    "BAAI/bge-small-en-v1.5": "_bge",
    "all-MiniLM-L6-v2 (no context)": "_noctx",
    "BAAI/bge-small-en-v1.5 (no context)": "_bge_noctx",
    "text-embedding-3-small": "_openai",
}

# Cache pipelines by (embedding_model, reranker_model) — loading is expensive
_pipeline_cache: dict[tuple, QueryPipeline] = {}


def get_pipeline(embedding_model: str, reranker_model: str) -> QueryPipeline:
    key = (embedding_model, reranker_model)
    if key not in _pipeline_cache:
        suffix = EMBEDDING_MODELS.get(embedding_model, "")
        print(f"[Worker] Loading pipeline: {embedding_model} / {reranker_model}")
        _pipeline_cache[key] = QueryPipeline(
            index_path=str(DATA_DIR / f"faiss{suffix}.index"),
            metadata_path=str(DATA_DIR / f"metadata{suffix}.json"),
            bm25_path=str(DATA_DIR / f"bm25{suffix}.pkl"),
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
        print(f"[Worker] Pipeline loaded: {embedding_model}")
    return _pipeline_cache[key]


def on_request(ch, method, props, body):
    status = "ok"
    try:
        req = json.loads(body)
        query_text = req.pop("query")
        embedding_model = req.pop("embedding_model", "text-embedding-3-small")
        reranker_model = req.pop("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        session_id = req.pop("session_id", props.correlation_id)

        print(f"[Worker] Query: '{query_text[:60]}...' model={embedding_model}")

        pipeline = get_pipeline(embedding_model, reranker_model)
        result = pipeline.query(query_text, session_id=session_id, **req)

        # Serialize: LLMResponse is a Pydantic model, convert to plain dict
        payload = json.dumps({
            "query": result["query"],
            "retrieval_results": result["retrieval_results"],
            "hyde_hypothesis": result["hyde_hypothesis"],
            "llm_response": result["llm_response"].model_dump(),
        }, default=str)

        print(f"[Worker] Done. Songs found: {len(result['retrieval_results'])}")

    except Exception as e:
        print(f"[Worker] Error: {e}")
        payload = json.dumps({"error": str(e)})
        status = "error"

    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,
        properties=pika.BasicProperties(
            correlation_id=props.correlation_id,
            headers={"status": status},
            delivery_mode=1,  # non-persistent reply
        ),
        body=payload,
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


def connect_with_retry(url: str, retries: int = 15, delay: int = 3) -> pika.BlockingConnection:
    for attempt in range(retries):
        try:
            return pika.BlockingConnection(pika.URLParameters(url))
        except Exception as e:
            if attempt < retries - 1:
                print(f"[Worker] RabbitMQ not ready ({e}), retry {attempt + 1}/{retries} in {delay}s...")
                time.sleep(delay)
            else:
                raise


def main():
    print(f"[Worker] Connecting to RabbitMQ at {RABBITMQ_URL.split('@')[-1]}...")
    connection = connect_with_retry(RABBITMQ_URL)
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)  # one query at a time per worker
    channel.basic_consume(queue=QUEUE, on_message_callback=on_request)

    print(f"[Worker] Ready. Listening on queue '{QUEUE}'. Press Ctrl+C to stop.")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    connection.close()


if __name__ == "__main__":
    main()
