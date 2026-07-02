"""
AGENTIC TOOL USE
================
Approach 3: The model itself decides which tools to call,
executes them in a loop, and synthesizes a final answer.

This uses OpenAI-compatible function calling via Groq's API.
The model returns structured tool_calls objects — no regex needed.

The agentic loop:
  1. Send conversation + tool definitions to model
  2. Model returns either text (done) or tool_calls (wants to use a tool)
  3. If tool_calls: execute each tool, append results, go to step 1
  4. If text: return as final response

Key difference from Guided mode:
  - Model decides WHAT to search and WHEN
  - Model can call multiple tools in one turn
  - Model synthesizes across multiple sources naturally
  - No routing logic needed — model figures it out from tool descriptions
"""

import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# SYSTEM PROMPT ADDITION
# Llama 70B sometimes uses wrong function call format.
# This explicit instruction forces the correct format.
# ─────────────────────────────────────────────

AGENT_SYSTEM_ADDITION = """
TOOL USE INSTRUCTIONS:
- Use the tools provided via the API when you need information
- Do NOT use <function=name{...}> format — use the API tools parameter only
- Call each tool only ONCE — do not repeat the same tool call
- After receiving tool results, synthesize them into a final answer
- Do not call a tool if you already have the result from a previous call
"""


# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# Descriptions matter enormously — the model picks tools
# based entirely on reading these descriptions.
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the pre-loaded knowledge base for factual information. "
                "Use this for general factual questions, historical facts, "
                "or information that might be in a pre-loaded document store."
                "Always try this before search_web for factual questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query to find relevant information"
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for current or recent information. "
                "Use ONLY when: the knowledge base returned no results, "
                "or the question is about current events, news, or recent developments. "
                "Do not use for general factual questions that the knowledge base can answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for web search"
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_uploaded_docs",
            "description": (
                "Search documents that the user uploaded in this session. "
                "Use this when the user asks about content from their uploaded file, "
                "or when they refer to 'my document', 'the file I uploaded', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for uploaded documents"
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression accurately. "
                "ALWAYS use this tool for any arithmetic — never calculate mentally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2847 * 3921')"
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ─────────────────────────────────────────────
# AGENT CLASS
# ─────────────────────────────────────────────

class Agent:
    """
    Agentic executor — runs the tool-use loop until the model
    produces a final text response.

    Args:
        rag            : RAGPipeline instance
        max_iterations : safety limit on tool call rounds
        api_key        : Groq API key
    """

    MODEL = "qwen/qwen3.6-27b"

    def __init__(
        self,
        rag,
        max_iterations: int = 5,
        api_key: str | None = None,
    ):
        self._rag = rag
        self._max_iterations = max_iterations
        self._client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def run(
        self,
        messages: list[dict],
        system_prompt: str,
        session_id: str | None = None,
    ) -> dict:
        """
        Run the agentic loop until the model produces a final answer.

        Args:
            messages      : conversation history from ConversationManager
            system_prompt : system prompt
            session_id    : for upload document search

        Returns:
            dict with:
                text         : final response text
                tools_called : list of tools the model used
                iterations   : how many loop iterations ran
                latency_ms   : total time
                error        : error message if failed
        """
        start = time.time()
        tools_called = []
        iterations = 0

        # Inject tool use instructions into system message
        # This fixes Llama 70B's tendency to use wrong function call format
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                api_messages.append({
                    "role":    "system",
                    "content": m["content"] + "\n" + AGENT_SYSTEM_ADDITION,
                })
            else:
                api_messages.append(dict(m))

        # Track seen tool calls to prevent infinite loops
        # If the model calls the same tool with the same args twice, break out
        seen_calls: set[str] = set()

        try:
            while iterations < self._max_iterations:
                iterations += 1

                # Call the model with tool definitions
                response = self._client.chat.completions.create(
                    model=self.MODEL,
                    max_tokens=1024,
                    messages=api_messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    extra_body={"reasoning_effort": "none"},
                )

                msg = response.choices[0].message

                # ── Did the model want to call tools? ────────────────
                if msg.tool_calls:
                    # Check for duplicate tool calls — prevents infinite loops
                    duplicate_detected = False
                    for tc in msg.tool_calls:
                        sig = f"{tc.function.name}:{tc.function.arguments}"
                        if sig in seen_calls:
                            print(f"[Agent] Duplicate call detected: {tc.function.name} — forcing final answer")
                            duplicate_detected = True
                            break
                        seen_calls.add(sig)

                    if duplicate_detected:
                        # Force final answer by calling without tools
                        final = self._client.chat.completions.create(
                            model=self.MODEL,
                            max_tokens=1024,
                            messages=api_messages,
                            tool_choice="none",
                            extra_body={"reasoning_effort": "none"},
                        )
                        return {
                            "text":         final.choices[0].message.content or "",
                            "tools_called": tools_called,
                            "iterations":   iterations,
                            "latency_ms":   int((time.time() - start) * 1000),
                            "error":        None,
                        }

                    # Add assistant's tool call request to message history
                    api_messages.append({
                        "role":       "assistant",
                        "content":    msg.content or "",
                        "tool_calls": [
                            {
                                "id":       tc.id,
                                "type":     "function",
                                "function": {
                                    "name":      tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in msg.tool_calls
                        ],
                    })

                    # Execute each tool and append results
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        print(f"[Agent] Calling: {tool_name}({args})")
                        result = self._execute_tool(tool_name, args, session_id)
                        tools_called.append({
                            "tool":   tool_name,
                            "args":   args,
                            "result": result[:200],
                        })

                        # Append tool result to message history
                        api_messages.append({
                            "role":         "tool",
                            "tool_call_id": tool_call.id,
                            "content":      result,
                        })

                    # Loop back — model reads results and decides next step

                else:
                    # ── Model produced a final text response ──────────
                    final_text = msg.content or ""

                    # Qwen3 sometimes returns empty text after tool use
                    # because it put everything in the thinking block.
                    # Force a synthesis call if text is empty but tools were called.
                    if not final_text.strip() and tools_called:
                        synthesis = self._client.chat.completions.create(
                            model=self.MODEL,
                            max_tokens=1024,
                            messages=api_messages + [{
                                "role": "user",
                                "content": "Please summarize the tool results and answer the original question."
                            }],
                            tool_choice="none",
                            extra_body={"reasoning_effort": "none"},
                        )
                        final_text = synthesis.choices[0].message.content or "I found information but could not format a response."

                    return {
                        "text":         final_text,
                        "tools_called": tools_called,
                        "iterations":   iterations,
                        "latency_ms":   int((time.time() - start) * 1000),
                        "error":        None,
                    }

            # Safety: exceeded max iterations
            return {
                "text":         "I reached the maximum number of reasoning steps. Please try rephrasing.",
                "tools_called": tools_called,
                "iterations":   iterations,
                "latency_ms":   int((time.time() - start) * 1000),
                "error":        "max_iterations_exceeded",
            }

        except Exception as e:
            return {
                "text":         f"Agent error: {str(e)}",
                "tools_called": tools_called,
                "iterations":   iterations,
                "latency_ms":   int((time.time() - start) * 1000),
                "error":        str(e),
            }

    def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        session_id: str | None,
    ) -> str:
        """
        Execute a tool and return its result as a string.
        All tool results must be strings — the model reads them as text.
        """
        try:
            if tool_name == "search_knowledge_base":
                chunks = self._rag._doc_retriever.retrieve(
                    args.get("query", ""),
                    min_similarity=0.5,
                )
                if not chunks:
                    return "No relevant information found in the knowledge base."
                return self._rag.format_context(chunks)

            elif tool_name == "search_web":
                from core.rag.sources.web import web_retrieve
                results = web_retrieve(
                    query=args.get("query", ""),
                    embedder=self._rag._embedder,
                    top_k=3,
                )
                if not results:
                    return "No web results found."
                lines = []
                for r in results:
                    lines.append(f"[{r['source']}] {r['text']}")
                return "\n\n".join(lines)

            elif tool_name == "search_uploaded_docs":
                if not session_id or session_id not in self._rag._upload_retrievers:
                    return "No uploaded documents found for this session."
                chunks = self._rag._upload_retrievers[session_id].retrieve(
                    args.get("query", ""),
                    min_similarity=0.5,
                )
                if not chunks:
                    return "No relevant content found in uploaded documents."
                return self._rag.format_context(chunks)

            elif tool_name == "calculator":
                from deployment.tools import ToolDispatcher
                dispatcher = ToolDispatcher()
                result = dispatcher.detect_and_execute(
                    f"TOOL_CALL: calculator({args.get('expression', '')})"
                )
                if result and result.success:
                    # Explicit format makes it harder for model to substitute wrong answer
                    return f"CALCULATOR RESULT: {args.get('expression', '')} = {result.output}"
                return f"Calculator error: {result.error if result else 'unknown'}"

            elif tool_name == "get_datetime":
                from datetime import datetime
                return f"Current date and time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}"

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            return f"Tool execution error: {str(e)}"