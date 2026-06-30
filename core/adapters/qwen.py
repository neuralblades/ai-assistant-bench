"""
QWEN ADAPTER
============
Implements BaseAdapter for Qwen2.5 via the HuggingFace Inference API.

HuggingFace uses the OpenAI-compatible chat completions endpoint,
so we use the `openai` SDK pointed at HF's base URL — this is the
standard pattern for HF inference.

HuggingFace-specific quirks this adapter handles:
1. System prompt goes INSIDE the messages list as {"role": "system", ...}
   — unlike Anthropic which takes it separately
2. Response text lives at response.choices[0].message.content
3. Token counts at response.usage.prompt_tokens / completion_tokens
   (OpenAI naming convention, different from Anthropic)
4. Model string includes the org prefix: "Qwen/Qwen2.5-7B-Instruct"
"""

import os
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from .base import BaseAdapter, AdapterResponse


class QwenAdapter(BaseAdapter):
    """
    Adapter for Qwen2.5 models via HuggingFace Inference API.

    Uses the OpenAI-compatible endpoint that HuggingFace exposes,
    which means we can use the openai Python SDK with a custom base_url.
    This pattern works for any OpenAI-compatible API (HF, Together, Groq, etc.)

    Args:
        model     : HF model identifier. Defaults to Qwen2.5-7B-Instruct.
        api_key   : HuggingFace API token. Reads from HF_API_TOKEN env var if not provided.
        max_tokens: Max tokens to generate. Default 1024.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # HuggingFace's OpenAI-compatible inference endpoint
    HF_BASE_URL = "https://router.huggingface.co/v1"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._max_tokens = max_tokens

        # Point the OpenAI SDK at HuggingFace's endpoint
        # This is the elegant trick — we reuse openai's battle-tested SDK
        # instead of writing raw HTTP requests
        self._client = OpenAI(
            api_key=api_key or os.getenv("HF_API_TOKEN"),
            base_url=self.HF_BASE_URL,
        )

    # ── ABSTRACT METHOD IMPLEMENTATIONS ──────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "huggingface"

    def generate(self, messages: list[dict], system_prompt: str) -> AdapterResponse:
        """
        Send messages to Qwen via HuggingFace and return AdapterResponse.

        The messages list from ConversationManager already includes the
        system message at index 0 — for HuggingFace this is correct,
        no stripping needed. The system prompt goes in the messages list
        as {"role": "system", "content": "..."}.

        So we pass messages directly as-is:
          [
            {"role": "system",    "content": "You are..."},   ← keep this
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user",      "content": "How are you?"}
          ]
        """
        start = self._start_timer()

        try:
            # HuggingFace OpenAI-compatible endpoint — messages include system
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=messages,          # ← pass directly, system msg included
            )

            # OpenAI/HF response structure — different from Anthropic
            text = response.choices[0].message.content

            # HF uses OpenAI token naming: prompt_tokens / completion_tokens
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
                Exception(f"HuggingFace rate limit: {e}"),
                self._elapsed_ms(start)
            )

        except APIConnectionError as e:
            return self._error_response(
                Exception(f"HuggingFace connection error: {e}"),
                self._elapsed_ms(start)
            )

        except APIError as e:
            return self._error_response(
                Exception(f"HuggingFace API error: {e}"),
                self._elapsed_ms(start)
            )

        except Exception as e:
            return self._error_response(e, self._elapsed_ms(start))
