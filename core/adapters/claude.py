"""
CLAUDE ADAPTER
==============
Implements BaseAdapter for Anthropic's Claude API.

Anthropic-specific quirks this adapter handles:
1. System prompt is a SEPARATE parameter, not inside the messages list
2. Response text lives at response.content[0].text
3. Token counts are at response.usage.input_tokens / output_tokens
4. The messages list passed in contains a system message at index 0
   (from ConversationManager) — we strip it out before sending to Anthropic
   since Anthropic takes it separately
"""

import os
from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError
from .base import BaseAdapter, AdapterResponse


class ClaudeAdapter(BaseAdapter):
    """
    Adapter for Claude models via the Anthropic SDK.

    Args:
        model   : Claude model string. Defaults to claude-sonnet-4-20250514.
        api_key : Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        max_tokens : Max tokens to generate in response. Default 1024.
    """

    # Default model — easy to change in one place
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._max_tokens = max_tokens

        # The Anthropic client handles connection pooling, retries, etc.
        # If api_key is None, the SDK automatically reads ANTHROPIC_API_KEY
        # from environment — standard 12-factor app pattern.
        self._client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    # ── ABSTRACT METHOD IMPLEMENTATIONS ──────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "anthropic"

    def generate(self, messages: list[dict], system_prompt: str) -> AdapterResponse:
        """
        Send messages to Claude and return standardized AdapterResponse.

        The messages list from ConversationManager looks like:
          [
            {"role": "system", "content": "You are..."},  ← index 0, strip this
            {"role": "user",   "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user",   "content": "How are you?"}
          ]

        Anthropic's API wants:
          system = "You are..."           ← separate param
          messages = [                    ← no system message here
            {"role": "user", ...},
            {"role": "assistant", ...},
            {"role": "user", ...},
          ]
        """
        start = self._start_timer()

        try:
            # Strip the system message from the list — Anthropic takes it separately.
            # Filter by role rather than slicing [1:] to be safe against edge cases.
            api_messages = [m for m in messages if m["role"] != "system"]

            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,       # ← Anthropic-specific: system as top-level param
                messages=api_messages,
            )

            # Extract the text — Anthropic returns a list of content blocks
            # (could be text, tool_use, etc.) — we want the first text block
            text = response.content[0].text

            return AdapterResponse(
                text=text,
                model=self._model,
                provider=self.provider,
                latency_ms=self._elapsed_ms(start),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                success=True,
            )

        except RateLimitError as e:
            # Rate limit hit — log it clearly so the eval runner can handle it
            return self._error_response(
                Exception(f"Rate limit exceeded: {e}"),
                self._elapsed_ms(start)
            )

        except APIConnectionError as e:
            return self._error_response(
                Exception(f"Connection error: {e}"),
                self._elapsed_ms(start)
            )

        except APIError as e:
            # Catch-all for other Anthropic API errors (invalid request, auth, etc.)
            return self._error_response(
                Exception(f"Anthropic API error [{e.status_code}]: {e.message}"),
                self._elapsed_ms(start)
            )

        except Exception as e:
            # Last resort — unexpected errors (SDK bugs, network issues, etc.)
            return self._error_response(e, self._elapsed_ms(start))
