"""
OBSERVABILITY
=============
Logs every request and response to SQLite.
Provides a metrics summary for the dashboard tab.

Why observability matters:
Without this, you're flying blind. You don't know:
  - How many requests your Space is handling
  - What the real-world latency distribution looks like
  - Which prompts are hitting guardrails
  - Whether the model is failing silently

With this, you can answer all of the above from a simple DB query.

What we log per request:
  - timestamp
  - user message
  - model response
  - latency (ms)
  - token counts
  - guardrail result (was it blocked? which tier?)
  - tool use (was a tool called? which one?)
  - success/failure

This is the raw material for the cost + latency table in your eval report.
"""

import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


DB_PATH = Path(__file__).parent / "data" / "observability.db"


# ─────────────────────────────────────────────
# LOG ENTRY TYPE
# ─────────────────────────────────────────────

@dataclass
class RequestLog:
    """
    A single logged request-response cycle.
    Every field corresponds to a column in the SQLite table.
    """
    timestamp: str
    user_message: str
    model_response: str
    model_name: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    guardrail_passed: bool
    guardrail_tier: str | None      # "rule", "classifier", or None (if passed)
    tool_called: str | None         # tool name, or None if no tool was used
    tool_result: str | None         # tool output, or None
    success: bool
    error: str | None


# ─────────────────────────────────────────────
# OBSERVABILITY CLASS
# ─────────────────────────────────────────────

class Observability:
    """
    Request logger and metrics aggregator.

    Usage:
        obs = Observability()

        # Log a completed request
        obs.log(RequestLog(
            timestamp=datetime.now().isoformat(),
            user_message="What is 2+2?",
            model_response="4",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            latency_ms=843,
            input_tokens=45,
            output_tokens=3,
            guardrail_passed=True,
            guardrail_tier=None,
            tool_called="calculator",
            tool_result="4",
            success=True,
            error=None,
        ))

        # Get metrics for the dashboard
        metrics = obs.get_metrics()
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = Path(db_path or DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create the logs table if it doesn't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT NOT NULL,
                    user_message      TEXT NOT NULL,
                    model_response    TEXT,
                    model_name        TEXT,
                    latency_ms        INTEGER,
                    input_tokens      INTEGER DEFAULT 0,
                    output_tokens     INTEGER DEFAULT 0,
                    guardrail_passed  INTEGER DEFAULT 1,
                    guardrail_tier    TEXT,
                    tool_called       TEXT,
                    tool_result       TEXT,
                    success           INTEGER DEFAULT 1,
                    error             TEXT
                )
            """)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── PUBLIC INTERFACE ──────────────────────────────────────────────

    def log(self, entry: RequestLog) -> None:
        """Write a request log entry to the database."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO request_logs (
                    timestamp, user_message, model_response, model_name,
                    latency_ms, input_tokens, output_tokens,
                    guardrail_passed, guardrail_tier,
                    tool_called, tool_result, success, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp,
                entry.user_message,
                entry.model_response,
                entry.model_name,
                entry.latency_ms,
                entry.input_tokens,
                entry.output_tokens,
                int(entry.guardrail_passed),
                entry.guardrail_tier,
                entry.tool_called,
                entry.tool_result,
                int(entry.success),
                entry.error,
            ))

    def get_metrics(self) -> dict:
        """
        Compute aggregate metrics from all logged requests.
        Used to populate the observability dashboard tab.
        """
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) as n FROM request_logs").fetchone()["n"]

            if total == 0:
                return {"total_requests": 0}

            stats = conn.execute("""
                SELECT
                    COUNT(*)                          as total,
                    AVG(latency_ms)                   as avg_latency,
                    MAX(latency_ms)                   as max_latency,
                    MIN(latency_ms)                   as min_latency,
                    AVG(input_tokens + output_tokens) as avg_tokens,
                    SUM(CASE WHEN success=0 THEN 1 ELSE 0 END)          as errors,
                    SUM(CASE WHEN guardrail_passed=0 THEN 1 ELSE 0 END) as blocked,
                    SUM(CASE WHEN tool_called IS NOT NULL THEN 1 ELSE 0 END) as tool_calls
                FROM request_logs
            """).fetchone()

            # p95 latency — need to compute manually in SQLite
            latencies = [
                row[0] for row in
                conn.execute("SELECT latency_ms FROM request_logs ORDER BY latency_ms").fetchall()
            ]
            p95_idx = int(len(latencies) * 0.95)
            p95_latency = latencies[p95_idx] if latencies else 0

            # Most recent 5 requests for the live log panel
            recent = conn.execute("""
                SELECT timestamp, user_message, latency_ms, success, guardrail_passed, tool_called
                FROM request_logs
                ORDER BY id DESC
                LIMIT 5
            """).fetchall()

            return {
                "total_requests":   stats["total"],
                "avg_latency_ms":   round(stats["avg_latency"] or 0),
                "p95_latency_ms":   p95_latency,
                "max_latency_ms":   stats["max_latency"] or 0,
                "avg_tokens":       round(stats["avg_tokens"] or 0),
                "error_count":      stats["errors"],
                "blocked_count":    stats["blocked"],
                "tool_call_count":  stats["tool_calls"],
                "recent_requests": [
                    {
                        "timestamp":        row["timestamp"][:19],
                        "message_preview":  row["user_message"][:50] + "…"
                                            if len(row["user_message"]) > 50
                                            else row["user_message"],
                        "latency_ms":       row["latency_ms"],
                        "success":          bool(row["success"]),
                        "blocked":          not bool(row["guardrail_passed"]),
                        "tool_used":        row["tool_called"],
                    }
                    for row in recent
                ],
            }

    def format_metrics_markdown(self) -> str:
        """
        Format metrics as a markdown string for the Gradio dashboard tab.
        """
        m = self.get_metrics()

        if m.get("total_requests", 0) == 0:
            return "### 📊 No requests logged yet."

        lines = [
            "### 📊 Live Metrics\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total requests | {m['total_requests']} |",
            f"| Avg latency | {m['avg_latency_ms']}ms |",
            f"| p95 latency | {m['p95_latency_ms']}ms |",
            f"| Max latency | {m['max_latency_ms']}ms |",
            f"| Avg tokens / turn | {m['avg_tokens']} |",
            f"| Guardrail blocks | {m['blocked_count']} |",
            f"| Tool calls | {m['tool_call_count']} |",
            f"| Errors | {m['error_count']} |",
            "",
            "#### Recent Requests",
        ]

        for r in m.get("recent_requests", []):
            status = "🚫" if r["blocked"] else ("🔧" if r["tool_used"] else "✅")
            lines.append(
                f"- {status} `{r['timestamp']}` | "
                f"`{r['latency_ms']}ms` | "
                f"{r['message_preview']}"
            )

        return "\n".join(lines)
