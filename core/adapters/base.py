"""
ABSTRACT MODEL ADAPTER + RESPONSE TYPE
=======================================
This module defines the contract that every model adapter must fulfill.

Key principle: The rest of the system (UI, eval, conversation manager)
only ever talks to this interface — never to Claude or Qwen directly.
This is what makes the system swappable and testable.
"""
import os
import time
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from dataclasses import dataclass

load_dotenv()

# ─────────────────────────────────────────────
# RESPONSE TYPE
# ─────────────────────────────────────────────

@dataclass
class AdapterResponse:
    """
    Standardized response object returned by every adapter.

    Why not just return a string?
    Because we need metadata for the eval report — latency, token counts,
    cost estimation. If we just return a string now, we'd have to rebuild
    the whole system later to collect this data.

    Build it right once.

    Fields:
        text          : the model's actual response text
        model         : exact model identifier used (e.g. "claude-sonnet-4-20250514")
        provider      : "anthropic" | "huggingface" | etc.
        latency_ms    : wall-clock time for the API call in milliseconds
        input_tokens  : tokens consumed by the prompt (from API response)
        output_tokens : tokens generated in the response (from API response)
        success       : False if the call failed, True otherwise
        error         : error message if success=False, else None
    """
    text: str
    model: str
    provider: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        """Convenience property — total tokens used in this call."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        """Serialize to dict — used when writing eval results to JSON."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "error": self.error,
        }


# ─────────────────────────────────────────────
# ABSTRACT BASE ADAPTER
# ─────────────────────────────────────────────

class BaseAdapter(ABC):
    """
    Abstract base class for all model adapters.

    Any class that inherits from BaseAdapter MUST implement:
    - generate()   : the core method — takes messages, returns AdapterResponse
    - model_name   : property returning the model identifier string
    - provider     : property returning the provider name

    If a subclass doesn't implement these, Python raises a TypeError
    at instantiation time — not at runtime when it's too late.
    This is the value of ABC (Abstract Base Class).

    Usage:
        class ClaudeAdapter(BaseAdapter):
            def generate(self, messages, system_prompt): ...
            @property
            def model_name(self): return "claude-sonnet-4-20250514"
            @property
            def provider(self): return "anthropic"
    """

    @abstractmethod
    def generate(self, messages: list[dict], system_prompt: str) -> AdapterResponse:
        """
        Send messages to the model and return a standardized response.

        Args:
            messages      : list of {"role": ..., "content": ...} dicts
                            from ConversationManager.get_messages()
                            NOTE: this list INCLUDES the system message at index 0.
                            Each adapter handles it according to its API's requirements.
            system_prompt : the raw system prompt string, passed separately
                            for APIs (like Anthropic) that take it as a distinct param.

        Returns:
            AdapterResponse with text, latency, token counts, etc.

        Must never raise — catch all exceptions internally and return
        AdapterResponse(success=False, error=str(e), ...) instead.
        This keeps the UI and eval runner clean from try/except noise.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The exact model string used in API calls."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """The provider name — 'anthropic', 'huggingface', etc."""
        ...

    # ── SHARED UTILITIES ─────────────────────────────────────────────
    # These are implemented here so every adapter gets them for free.

    def _start_timer(self) -> float:
        """Call at the start of an API request. Returns start time."""
        return time.time()

    def _elapsed_ms(self, start: float) -> int:
        """Call after API response. Returns elapsed time in milliseconds."""
        return int((time.time() - start) * 1000)

    def _error_response(self, error: Exception, latency_ms: int = 0) -> AdapterResponse:
        """
        Build a consistent failure response.
        Adapters call this in their except blocks instead of re-raising.

        Why return instead of raise?
        The eval runner compares two models across many prompts.
        If one model throws an exception on prompt #7, we don't want
        the entire eval run to crash — we want to record the failure
        and continue to prompt #8.
        """
        return AdapterResponse(
            text="",
            model=self.model_name,
            provider=self.provider,
            latency_ms=latency_ms,
            success=False,
            error=str(error),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, provider={self.provider})"
