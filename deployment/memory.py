"""
PERSISTENT MEMORY
=================
Stores facts about the user across sessions using SQLite.

The problem this solves:
  ConversationManager gives the model memory WITHIN a session.
  Once the session ends, everything is gone.
  A user who says "my name is Nullity" in session 1 has to repeat
  themselves in session 2.

This module gives the model memory ACROSS sessions.

How it works:
  1. After each model response, we extract key facts from the conversation
     (name, preferences, context) and store them in SQLite.
  2. At the start of each new session, we retrieve stored facts and
     inject them into the system prompt — so the model "remembers."

Architecture:
  SQLite database (facts.db)
  └── table: memories
        ├── key       (e.g. "user_name", "user_location")
        ├── value     (e.g. "Nullity", "Chennai")
        ├── source    (which conversation turn created this)
        └── updated_at

Why SQLite?
  - Zero configuration — no database server needed
  - Single file — easy to inspect, backup, and reset
  - Built into Python's standard library — no extra dependencies
  - Sufficient for a personal assistant with one user
  - For multi-user production: swap for PostgreSQL, same interface
"""

import sqlite3
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime


DB_PATH = Path(__file__).parent / "data" / "memory.db"


class Memory:
    """
    SQLite-backed persistent memory store.

    Stores facts as key-value pairs. Keys are semantic labels
    ("user_name", "user_job", "user_interests") and values are
    whatever the model extracted from the conversation.

    Usage:
        mem = Memory()

        # Store a fact
        mem.store("user_name", "Nullity")

        # Retrieve all facts as a formatted string for the system prompt
        context = mem.get_context_string()
        # → "User's name is Nullity. User works in cryptography."

        # Let the model decide what to store from a conversation turn
        mem.extract_and_store(user_message, assistant_response)
    """

    # Facts the model should try to remember when it encounters them
    EXTRACTABLE_KEYS = [
        "user_name",
        "user_location",
        "user_job",
        "user_interests",
        "user_preferences",
    ]

    def __init__(self, db_path: str | None = None):
        self._db_path = Path(db_path or DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the memories table if it doesn't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    source      TEXT,
                    updated_at  TEXT NOT NULL
                )
            """)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection with row factory for dict-like access."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────

    def store(self, key: str, value: str, source: str = "manual") -> None:
        """
        Store or update a fact.

        Uses INSERT OR REPLACE — if the key already exists,
        it updates the value rather than creating a duplicate.
        """
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories (key, value, source, updated_at)
                VALUES (?, ?, ?, ?)
            """, (key, value, source, datetime.now().isoformat()))

    def retrieve(self, key: str) -> str | None:
        """Retrieve a single fact by key. Returns None if not found."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM memories WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def retrieve_all(self) -> dict[str, str]:
        """Retrieve all stored facts as a dict."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key, value FROM memories").fetchall()
            return {row["key"]: row["value"] for row in rows}

    def get_context_string(self) -> str:
        """
        Format all stored facts into a string for injection
        into the system prompt.

        Example output:
          "Known facts about the user:
           - Name: Nullity
           - Location: Chennai
           - Job: cryptography engineer"
        """
        facts = self.retrieve_all()
        if not facts:
            return ""

        lines = ["Known facts about the user:"]
        label_map = {
            "user_name":        "Name",
            "user_location":    "Location",
            "user_job":         "Job / profession",
            "user_interests":   "Interests",
            "user_preferences": "Preferences",
        }
        for key, value in facts.items():
            label = label_map.get(key, key.replace("_", " ").title())
            lines.append(f"  - {label}: {value}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Wipe all stored memories. Used for 'forget me' or testing."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM memories")

    def extract_and_store(self, user_msg: str, assistant_msg: str) -> list[str]:
        """
        Simple rule-based extraction of facts from a conversation turn.

        Looks for patterns like:
          "my name is X" → store user_name = X
          "I live in X"  → store user_location = X
          "I work as X"  → store user_job = X

        Returns list of keys that were stored this turn.

        Why rule-based and not LLM-based extraction?
        LLM extraction is more powerful but adds latency and cost on every
        single turn. Rule-based catches the most common patterns for free.
        For production: use LLM extraction on turns where rule-based found nothing.
        """
        stored = []
        text = user_msg.lower()

        # Name extraction
        name_match = re.search(
            r"my name is ([A-Za-z][A-Za-z0-9_\-]{1,30})", text, re.IGNORECASE
        )
        if name_match:
            self.store("user_name", name_match.group(1).strip().title(), source="auto")
            stored.append("user_name")

        # Location extraction
        location_match = re.search(
            r"i (?:live|am|'m) (?:in|from) ([A-Za-z ]{2,40})", text, re.IGNORECASE
        )
        if location_match:
            self.store("user_location", location_match.group(1).strip().title(), source="auto")
            stored.append("user_location")

        # Job extraction
        job_match = re.search(
            r"i (?:work as|am a|am an) ([A-Za-z ]{2,40})", text, re.IGNORECASE
        )
        if job_match:
            self.store("user_job", job_match.group(1).strip(), source="auto")
            stored.append("user_job")

        return stored
