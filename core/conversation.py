"""
CONVERSATION MANAGER
====================
This module handles all conversational state — history, memory,
context window limits, and session management.

Key principle: This class is completely model-agnostic.
It doesn't know about Claude or Qwen. It just manages a list of messages
in the standard OpenAI-compatible chat format that all modern LLMs accept.
"""

import time
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────
# TYPE DEFINITIONS
# ─────────────────────────────────────────────

# A "Role" is one of three string literals.
# Using Literal here means Python will catch typos like "usr" at dev time.
Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """
    Represents a single turn in the conversation.

    Why a dataclass?
    - Cleaner than a plain dict — you get attribute access (msg.role vs msg["role"])
    - Self-documenting — the fields tell you exactly what a message contains
    - Easy to convert to dict when the API needs it (see to_dict())

    Fields:
        role      : who sent this message
        content   : the text content
        timestamp : unix timestamp, useful for logs and observability
    """
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)  # auto-set to now if not provided

    def to_dict(self) -> dict:
        """
        Convert to the dict format that LLM APIs expect.
        Most APIs want: {"role": "user", "content": "hello"}
        We exclude timestamp — APIs don't need it, it's internal metadata.
        """
        return {"role": self.role, "content": self.content}


# ─────────────────────────────────────────────
# CONVERSATION MANAGER
# ─────────────────────────────────────────────

class ConversationManager:
    """
    Manages the full state of a conversation session.

    Responsibilities:
    1. Store messages in the correct order
    2. Apply a sliding window to prevent context overflow
    3. Always prepend the system prompt (the model's personality/behavior)
    4. Provide clean reset/session management

    Args:
        system_prompt : The instruction that defines the assistant's behavior.
                        This is always message #1, never removed by the window.
        max_turns     : Max number of conversation TURNS (user+assistant pairs)
                        to keep in the window. Default 10 = 20 messages max.
    """

    def __init__(self, system_prompt: str, max_turns: int = 10):
        self.system_prompt = system_prompt
        self.max_turns = max_turns

        # The history list stores only user+assistant messages.
        # The system prompt is stored separately and always prepended.
        # Why separate? Because the system prompt is immutable — it never
        # gets dropped by the sliding window. Mixing it into history
        # would complicate the windowing logic.
        self._history: list[Message] = []

        # Track session metadata — useful for observability/logging
        self._session_start = time.time()
        self._turn_count = 0

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        """
        Append a user turn to history.
        Call this BEFORE sending to the model.
        """
        if not content.strip():
            raise ValueError("User message cannot be empty.")
        self._history.append(Message(role="user", content=content.strip()))

    def add_assistant_message(self, content: str) -> None:
        """
        Append the model's response to history.
        Call this AFTER receiving the response from the model.

        Why enforce this order? Because history must always alternate:
        user → assistant → user → assistant
        Violating this order causes undefined behavior in most LLM APIs.
        """
        if not content.strip():
            raise ValueError("Assistant message cannot be empty.")
        self._history.append(Message(role="assistant", content=content.strip()))
        self._turn_count += 1  # a "turn" completes when assistant responds

    def get_messages(self) -> list[dict]:
        """
        Returns the full message list ready to send to any LLM API.

        This is the core method — it:
        1. Applies the sliding window to _history
        2. Prepends the system prompt
        3. Returns everything as a list of plain dicts

        The sliding window keeps the last (max_turns * 2) messages.
        We multiply by 2 because each turn = 1 user msg + 1 assistant msg.

        Example with max_turns=2:
          Full history: [u1, a1, u2, a2, u3, a3, u4]  (7 messages)
          Window keeps: [u3, a3, u4]                   (last 4 + incomplete turn)
          Final output: [system, u3, a3, u4]
        """
        # Apply sliding window — keep last N turn-pairs
        window_size = self.max_turns * 2
        windowed = self._history[-window_size:] if len(self._history) > window_size else self._history

        # Build final list: system prompt first, then windowed history
        system_message = {"role": "system", "content": self.system_prompt}
        return [system_message] + [msg.to_dict() for msg in windowed]

    def reset(self) -> None:
        """
        Clear conversation history and start a fresh session.
        The system prompt is preserved — only the history is wiped.

        When to call this:
        - User clicks "New Chat"
        - Starting a new eval run
        - Session timeout
        """
        self._history.clear()
        self._session_start = time.time()
        self._turn_count = 0

    # ── OBSERVABILITY HELPERS ─────────────────────────────────────────

    def token_estimate(self) -> int:
        """
        Rough estimate of current context size in tokens.

        Rule of thumb: 1 token ≈ 4 characters in English.
        This is NOT exact — use it for monitoring, not hard limits.
        For hard limits, use the adapter's actual token counts from API responses.
        """
        total_chars = len(self.system_prompt)
        for msg in self._history:
            total_chars += len(msg.content)
        return total_chars // 4

    def get_stats(self) -> dict:
        """
        Returns session metadata — useful for logging and the eval report.
        """
        return {
            "turn_count": self._turn_count,
            "history_length": len(self._history),
            "estimated_tokens": self.token_estimate(),
            "session_duration_s": round(time.time() - self._session_start, 2),
            "window_applied": len(self._history) > self.max_turns * 2,
        }

    def __repr__(self) -> str:
        return (
            f"ConversationManager("
            f"turns={self._turn_count}, "
            f"history={len(self._history)} messages, "
            f"~{self.token_estimate()} tokens)"
        )
