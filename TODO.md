# TODO — Testing Checklist

## 1. Langfuse — Core Pipeline (CLI)
- [ ] Run a query via CLI and confirm trace appears in Langfuse dashboard
  ```
  .venv/Scripts/python pipeline/query.py "your query here" --top-k 5
  ```
- [ ] Confirm trace has correct structure: `rag-query` → `retrieval` → `anthropic-structured`
- [ ] Confirm token usage (input/output tokens) is visible on the generation span
- [ ] Confirm session metadata is present on the trace
- [ ] Test with `--use-reranker` and confirm reranker span appears inside `retrieval`
- [ ] Test with `--use-hyde` and confirm `hyde-hypothesis` generation span appears

## 2. Langfuse — Streamlit App (Direct Streaming)
- [ ] Start the app: `.venv/Scripts/streamlit run app/app.py`
- [ ] Run a query and confirm a new trace appears in Langfuse with `session_id` set
- [ ] Run multiple queries in the same browser tab — confirm they share the same `session_id`
- [ ] Open a new browser tab — confirm a different `session_id` is assigned
- [ ] Check the Sessions view in Langfuse dashboard

## 3. Langfuse — Evaluation Notebook
The `_judge_inputs` checkpoint was created before Langfuse was added, so it has no `trace_id` values.
The `if trace_id:` guard means the score upload silently skips — no crash, results still print locally.

**Preferred approach — upload scores without touching the checkpoint:**
- [ ] Add a new cell at the end of the judge section that uploads `judge_results_anthropic` directly to Langfuse as a Dataset, no trace_ids needed:
  ```python
  from langfuse import get_client
  lf = get_client()
  dataset = lf.create_dataset(name="lyrics-rag-judge-eval")
  for entry in judge_results_anthropic:
      item = lf.create_dataset_item(
          dataset_name="lyrics-rag-judge-eval",
          input=entry["query"],
          expected_output=entry["expected"],
          metadata={k: v for k, v in entry.items() if k not in ("query", "expected")},
      )
  lf.flush()
  ```
- [ ] Confirm the dataset appears in Langfuse under **Datasets**
- [ ] Repeat for `judge_results_openai` if available

**Future runs only (if checkpoint is ever cleared for other reasons):**
- [ ] The trace_id capture and per-trace score logging will work automatically — no extra steps needed

## 4. Docker — Build & Run
- [ ] Build the Docker image: `docker compose build`
- [ ] Start all services: `docker compose up`
- [ ] Confirm RabbitMQ management UI is accessible at `http://localhost:15672`
- [ ] Confirm Streamlit app is accessible at `http://localhost:8501`
- [ ] Confirm worker container starts without errors (`docker compose logs worker`)
- [ ] Add Langfuse env vars to `docker-compose.yml` or a `.env` file used by Compose

## 5. RabbitMQ / Worker Path
- [ ] With Docker running, send a query via the Streamlit UI (it should route through RabbitMQ automatically when `RABBITMQ_URL` is set)
- [ ] Confirm the worker processes the query (`docker compose logs worker`)
- [ ] Confirm the Langfuse trace for the worker query has `session_id` matching the `correlation_id` from the app
- [ ] Test worker failure recovery: stop the worker mid-query and confirm the app shows a timeout error gracefully

## 6. Ollama (Local LLM) Path
- [ ] Start Ollama and pull the model: `ollama pull llama3`
- [ ] Switch the Streamlit app to "Ollama (Llama 3 local)" backend
- [ ] Run a query and confirm it completes
- [ ] Confirm a Langfuse trace appears with `llm_backend: ollama` in metadata (note: no token counts expected for Ollama)

## 7. General Regression Check
- [ ] All 5 embedding model variants still load correctly in the Streamlit sidebar (requires the corresponding FAISS indices to exist)
- [ ] HyDE toggle works without errors
- [ ] Hybrid search toggle works without errors
- [ ] Re-ranker model dropdown switches correctly
- [ ] Similarity threshold slider filters results as expected
