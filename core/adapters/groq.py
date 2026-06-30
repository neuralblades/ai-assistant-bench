"""
GROQ ADAPTER
============
Implements BaseAdapter for Llama 3.3 70B via Groq's OpenAI-compatible API.

Same shape as QwenAdapter — Groq exposes an OpenAI-compatible endpoint,
so we reuse the openai SDK pointed at Groq's base_url, exactly like
the HuggingFace router pattern.

Why this adapter exists:
Used as a free stand-in for Claude in eval runs when no Anthropic
credits are available. Llama 3.3 70B is a strong enough model to give
a meaningful "frontier-tier" comparison point against Qwen 7B.
"""

import os
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from .base import BaseAdapter, AdapterResponse


class GroqAdapter(BaseAdapter):
    """
    Adapter for Llama 3.3 70B via Groq's API.

    Args:
        model      : Groq model string. Defaults to llama-3.3-70b-versatile.
        api_key    : Groq API key. Reads from GROQ_API_KEY env var if not provided.
        max_tokens : Max tokens to generate. Default 1024.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url=self.GROQ_BASE_URL,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "groq"

    def generate(self, messages: list[dict], system_prompt: str) -> AdapterResponse:
        """
        Same OpenAI-compatible shape as QwenAdapter — system prompt
        stays inside the messages list, response extraction matches
        the OpenAI response schema.
        """
        start = self._start_timer()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,
            )

            text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            return AdapterResponse(
                text=text,
                model=self._model,
                provider=self.provider,
                latency_ms=self._elapsed_ms(start),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True,
            )

        except RateLimitError as e:
            return self._error_response(
                Exception(f"Groq rate limit: {e}"),
                self._elapsed_ms(start)
            )

        except APIConnectionError as e:
            return self._error_response(
                Exception(f"Groq connection error: {e}"),
                self._elapsed_ms(start)
            )

        except APIError as e:
            return self._error_response(
                Exception(f"Groq API error: {e}"),
                self._elapsed_ms(start)
            )

        except Exception as e:
            return self._error_response(e, self._elapsed_ms(start))