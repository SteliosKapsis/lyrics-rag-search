"""
LLM backend calling for the RAG query pipeline.

Owns all direct calls to Anthropic Claude and Ollama/Llama 3 — plain-text
(HyDE), structured (tool-use / JSON-schema-constrained), and streaming —
separated from QueryPipeline's retrieval/grouping/orchestration logic.
Composed into QueryPipeline the same way pipeline/reranker.py's Reranker
already is.

This is a structural extraction only: every method here is a verbatim move
from pipeline/query.py, not a rewrite. Same prompts, same API calls, same
parsing, same error handling — QueryPipeline's public methods (.query(),
.query_stream(), etc.) and their observable behavior are unchanged.

LLMResponse/RESPONSE_TOOL are imported lazily inside the methods that need
them (not at module level) to avoid a circular import with pipeline.query,
which imports LLMBackend.
"""

import json
import os
from typing import Generator

import anthropic
import requests
from langfuse import get_client, observe


class LLMBackend:
    """Talks to whichever LLM backend (Anthropic or Ollama) a QueryPipeline
    is configured for. Holds no retrieval/index state of its own."""

    def __init__(self, llm_backend: str, llm_model: str, anthropic_client=None):
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.anthropic_client = anthropic_client

    @observe(as_type="generation", name="hyde-hypothesis")
    def call_anthropic(self, system: str, user_message: str) -> str:
        """Call Claude via the Anthropic API (plain text, used for HyDE)."""
        if not self.anthropic_client:
            return "[Error: ANTHROPIC_API_KEY not configured]"

        response = self.anthropic_client.messages.create(
            model=self.llm_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        get_client().update_current_generation(
            model=self.llm_model,
            usage_details={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
        )
        return response.content[0].text

    @observe(as_type="generation", name="anthropic-structured")
    def call_anthropic_structured(self, system: str, user_message: str):
        """
        Call Claude with tool_use to get structured JSON output.

        Uses tool_use: defines a tool whose input_schema matches the Pydantic
        model. Claude returns structured JSON as the tool call arguments, which
        we parse and validate with Pydantic.
        """
        from pipeline.query import LLMResponse, RESPONSE_TOOL

        if not self.anthropic_client:
            return LLMResponse(
                matches=[], confidence="low",
                summary="Error: ANTHROPIC_API_KEY not configured",
            )

        response = self.anthropic_client.messages.create(
            model=self.llm_model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            tools=[RESPONSE_TOOL],
            tool_choice={"type": "tool", "name": "format_response"},
        )

        # Extract tool call arguments
        result = LLMResponse(
            matches=[], confidence="low",
            summary="Failed to parse structured response from LLM.",
        )
        for block in response.content:
            if block.type == "tool_use":
                result = LLMResponse.model_validate(block.input)
                break

        get_client().update_current_generation(
            model=self.llm_model,
            input=user_message,
            output=result.model_dump(),
            usage_details={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
        )
        return result

    def _get_ollama_url(self) -> str:
        """Get the Ollama API base URL from env or default."""
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return f"{host}/api/chat"

    @observe(as_type="generation", name="hyde-hypothesis-ollama")
    def call_ollama(self, system: str, user_message: str) -> str:
        """
        Call a local LLM via the Ollama REST API (plain text, used for HyDE).

        Ollama is a tool for running open-source LLMs locally. It downloads,
        manages, and serves models via a simple REST API on localhost:11434.

        Install: https://ollama.com/download
        Pull Llama 3: ollama pull llama3
        It runs automatically after install — no manual start needed.
        """
        url = self._get_ollama_url()
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }

        get_client().update_current_generation(model=self.llm_model)
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.ConnectionError:
            return (
                "[Error: Cannot connect to Ollama. "
                "Make sure Ollama is installed and running. "
                "Install: https://ollama.com/download | "
                "Pull model: ollama pull llama3]"
            )
        except requests.Timeout:
            return "[Error: Ollama request timed out after 120s]"
        except Exception as e:
            return f"[Error: Ollama request failed: {e}]"

    @observe(as_type="generation", name="ollama-structured")
    def call_ollama_structured(self, system: str, user_message: str):
        """
        Call Ollama with the format parameter to get structured JSON output.

        Uses the format parameter with the Pydantic model's JSON schema,
        which forces Ollama to return conforming JSON.
        """
        from pipeline.query import LLMResponse

        get_client().update_current_generation(model=self.llm_model, input=user_message)
        url = self._get_ollama_url()
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": LLMResponse.model_json_schema(),
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            result = LLMResponse.model_validate_json(content)
            get_client().update_current_generation(output=result.model_dump())
            return result
        except requests.ConnectionError:
            return LLMResponse(
                matches=[], confidence="low",
                summary="Error: Cannot connect to Ollama. Install: https://ollama.com/download",
            )
        except Exception as e:
            return LLMResponse(
                matches=[], confidence="low",
                summary=f"Error: Ollama request failed: {e}",
            )

    @observe(as_type="generation", name="anthropic-stream")
    def stream_anthropic(self, system: str, user_message: str) -> Generator[str, None, None]:
        """
        Stream tokens from Claude via the Anthropic API using tool_use.

        Streams the JSON being built as tool input. The caller collects all
        tokens and parses the final JSON into LLMResponse after stream completes.
        """
        from pipeline.query import RESPONSE_TOOL

        get_client().update_current_generation(model=self.llm_model)
        if not self.anthropic_client:
            yield "[Error: ANTHROPIC_API_KEY not configured]"
            return

        try:
            with self.anthropic_client.messages.stream(
                model=self.llm_model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                tools=[RESPONSE_TOOL],
                tool_choice={"type": "tool", "name": "format_response"},
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                        yield event.delta.partial_json
        except anthropic.APIStatusError as e:
            error_msg = e.body.get("error", {}).get("message", str(e)) if isinstance(e.body, dict) else str(e)
            if "content filtering" in error_msg.lower():
                yield "[Error: Response blocked by content filter. Try rephrasing your query.]"
            else:
                yield f"[Error: Anthropic API error: {error_msg}]"
        except anthropic.APIConnectionError:
            yield "[Error: Cannot connect to Anthropic API. Check your network.]"
        except anthropic.RateLimitError:
            yield "[Error: Rate limit exceeded. Please wait and retry.]"
        except anthropic.APITimeoutError:
            yield "[Error: Request timed out. Please retry.]"

    @observe(as_type="generation", name="ollama-stream")
    def stream_ollama(self, system: str, user_message: str) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama REST API with structured format.

        Streams the JSON being built token by token. The caller collects all
        tokens and parses the final JSON into LLMResponse after stream completes.
        """
        from pipeline.query import LLMResponse

        get_client().update_current_generation(model=self.llm_model)
        url = self._get_ollama_url()
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "stream": True,
            "format": LLMResponse.model_json_schema(),
        }

        try:
            resp = requests.post(url, json=payload, timeout=120, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except requests.ConnectionError:
            yield (
                "[Error: Cannot connect to Ollama. "
                "Install: https://ollama.com/download | Pull model: ollama pull llama3]"
            )
        except Exception as e:
            yield f"[Error: Ollama streaming failed: {e}]"
