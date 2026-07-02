"""
LLM-AS-ROUTER
=============
Approach 2: Uses a fast LLM to classify retrieval intent
and rewrite the search query before RAG retrieval.

Why this beats keyword matching:
- Resolves pronouns: "verify that" → "Eiffel Tower height"
- Handles paraphrases: "look it up online" → source="web"
- Understands context: reads full conversation history
- Returns concrete search query, not raw user message

Model choice: llama-3.1-8b-instant fast and accurate 
Router task is simple enough that we don't need a separate
smaller model — the latency overhead is ~200-400ms which is
acceptable for the quality improvement.

If you want to optimize later: swap for a faster/cheaper model.
"""

import json
import os
import re
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


ROUTER_SYSTEM_PROMPT = """You are a routing assistant. Your job is to analyze a conversation and decide how to retrieve information for the user's latest message.

Available retrieval sources:
- "upload": user's uploaded documents (session-specific files they provided)
- "document": fixed knowledge base (pre-loaded facts)  
- "web": live internet search (current events, verification, recent info)
- "none": no retrieval needed (simple conversation, math, greetings)

Your job:
1. Decide which source to use
2. Write a concrete search query (resolve any pronouns like "that", "it", "this" using conversation history)

Rules:
- Use "none" for: greetings, math, simple questions the model knows, tool calls
- Use "web" for: current events, news, verification requests, "check online", "surf the web"
- Use "upload" if: user asks about something from their uploaded file
- Use "document" for: factual questions that might be in the knowledge base
- Use "document" as default when unsure between document and web

IMPORTANT: Resolve pronouns. If user says "verify that" after discussing the Eiffel Tower, query = "Eiffel Tower height".

Respond ONLY with valid JSON, no other text:
{"source": "web", "query": "Eiffel Tower height meters"}"""


class Router:
    """
    LLM-based retrieval router.

    Takes the full conversation history + current message,
    returns the optimal retrieval source and a concrete search query.

    Args:
        model   : Groq model to use for routing
        api_key : Groq API key
    """

    # llama-3.1-8b-instant chosen over larger models for routing:
    # - Tested 8B vs 20B vs 70B: all three scored identically on routing quality (5/5)
    # - 8B is faster (~330ms vs ~550ms for 70B)
    # - 8B has 14,400 RPD free quota vs 70B's 1,000 RPD
    # - Routing is a classification task, not complex reasoning — 8B is sufficient
    ROUTER_MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str | None = None):
        self._client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def route(
        self,
        user_message: str,
        conversation_history: list[dict],
    ) -> dict:
        """
        Decide retrieval source and query for the current message.

        Args:
            user_message         : the user's current message
            conversation_history : list of {"role", "content"} dicts
                                   (recent turns for context)

        Returns:
            dict with keys:
                source  : "upload" | "document" | "web" | "none"
                query   : concrete search query string
                latency_ms : how long the router call took
        """
        start = time.time()

        # Build context from recent conversation (last 4 turns = 8 messages)
        # We don't need the full history — just enough for pronoun resolution
        recent = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history

        # Format conversation for the router
        # Strip RAG context noise from history if present
        formatted_history = []
        for msg in recent:
            content = msg["content"]
            # If the message contains RAG context, extract just the actual question
            if "Using ONLY the above context" in content:
                # Extract the real question after the RAG injection
                parts = content.split("Using ONLY the above context, answer:")
                if len(parts) > 1:
                    content = parts[-1].strip()
            formatted_history.append(f"{msg['role'].upper()}: {content[:200]}")

        history_text = "\n".join(formatted_history)

        router_user_message = f"""Conversation so far:
{history_text}

Latest user message: "{user_message}"

Decide the retrieval source and search query."""

        try:
            response = self._client.chat.completions.create(
                model=self.ROUTER_MODEL,
                max_tokens=100,
                temperature=0,   # deterministic — routing should be consistent
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user",   "content": router_user_message},
                ],
            )

            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            result = json.loads(clean)

            return {
                "source":     result.get("source", "document"),
                "query":      result.get("query", user_message),
                "latency_ms": int((time.time() - start) * 1000),
                "error":      None,
            }

        except json.JSONDecodeError as e:
            # Router returned invalid JSON — fall back to auto
            return {
                "source":     "auto",
                "query":      user_message,
                "latency_ms": int((time.time() - start) * 1000),
                "error":      f"JSON parse error: {e}",
            }

        except Exception as e:
            # Router call failed — fall back gracefully, don't crash main flow
            return {
                "source":     "auto",
                "query":      user_message,
                "latency_ms": int((time.time() - start) * 1000),
                "error":      str(e),
            }
