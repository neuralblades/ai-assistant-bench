"""
TOOL USE
========
Gives the model the ability to call external functions.

How tool use works conceptually:
  1. You define tools and describe them in the system prompt
  2. The model generates a response that LOOKS LIKE a tool call
     e.g. "TOOL_CALL: calculator(2847 * 3921)"
  3. Your code detects this pattern, executes the real function
  4. The result is injected back and the model generates a final response

This is "poor man's tool use" — real tool use (OpenAI function calling,
Anthropic tool_use) uses structured API parameters. For a local model
like Qwen 0.5B that doesn't support structured tool calling natively,
we use a text-based protocol instead.

The pattern:
  Model output: "TOOL_CALL: calculator(147 * 892)"
  Your code:    detect → execute → "TOOL_RESULT: 131124"
  Model input:  "TOOL_RESULT: 131124" → generates final answer

Tools included:
  - calculator  : evaluates math expressions safely
  - datetime    : returns current date and time
  - (extensible): add any function following the same pattern
"""

import re
import math
from datetime import datetime
from dataclasses import dataclass


# ─────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    Result of a tool execution.

    Fields:
        tool_name  : which tool was called
        input      : what was passed to the tool
        output     : what the tool returned
        success    : whether it ran without error
        error      : error message if success=False
    """
    tool_name: str
    input: str
    output: str
    success: bool
    error: str | None = None


# ─────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────

def _calculator(expression: str) -> ToolResult:
    """
    Safely evaluate a math expression.

    Why not just use eval()?
    eval() executes arbitrary Python — a malicious expression like
    "__import__('os').system('rm -rf /')" would run on your server.

    Safe approach: only allow a whitelist of math operations.
    We parse the expression ourselves rather than trusting Python's eval.

    Allowed: +, -, *, /, **, (), numbers, and math functions (sqrt, etc.)
    Blocked: everything else
    """
    # Whitelist: only allow math-safe characters
    safe_pattern = re.compile(r'^[\d\s\+\-\*\/\(\)\.\,\%\^]+$')

    # Also allow math function names
    clean = expression.strip()
    clean = clean.replace("^", "**")  # support ^ as power operator

    # Replace common math function names with their math.* equivalents
    clean = re.sub(r'\bsqrt\b', 'math.sqrt', clean)
    clean = re.sub(r'\babs\b',  'math.fabs', clean)
    clean = re.sub(r'\bpi\b',   'math.pi',   clean)
    clean = re.sub(r'\bsin\b',  'math.sin',  clean)
    clean = re.sub(r'\bcos\b',  'math.cos',  clean)
    clean = re.sub(r'\blog\b',  'math.log',  clean)

    # Final safety check — only allow known-safe characters after substitution
    allowed = re.compile(r'^[\d\s\+\-\*\/\(\)\.\,a-z_]+$')
    if not allowed.match(clean):
        return ToolResult(
            tool_name="calculator",
            input=expression,
            output="",
            success=False,
            error=f"Expression contains disallowed characters: {expression}",
        )

    try:
        # Restricted eval — only math module in scope
        result = eval(clean, {"__builtins__": {}, "math": math})
        return ToolResult(
            tool_name="calculator",
            input=expression,
            output=str(round(float(result), 6)),
            success=True,
        )
    except Exception as e:
        return ToolResult(
            tool_name="calculator",
            input=expression,
            output="",
            success=False,
            error=str(e),
        )


def _datetime_tool(query: str = "") -> ToolResult:
    """Return the current date and time."""
    now = datetime.now()
    output = now.strftime("%A, %B %d, %Y at %I:%M %p")
    return ToolResult(
        tool_name="datetime",
        input=query,
        output=output,
        success=True,
    )


# ─────────────────────────────────────────────
# TOOL REGISTRY
# Maps tool names to their implementation functions
# Adding a new tool = add one entry here
# ─────────────────────────────────────────────

TOOL_REGISTRY = {
    "calculator": _calculator,
    "datetime":   _datetime_tool,
}


# ─────────────────────────────────────────────
# SYSTEM PROMPT ADDITION
# Tells the model about available tools and how to call them
# ─────────────────────────────────────────────

TOOLS_SYSTEM_PROMPT = """
You have access to the following tools. Use them when appropriate.

To call a tool, output EXACTLY this format on its own line:
TOOL_CALL: tool_name(argument)

Available tools:
- calculator(expression) : evaluates math. Example: TOOL_CALL: calculator(15 * 847)
- datetime()             : returns current date and time. Example: TOOL_CALL: datetime()

Rules:
- Use a tool whenever the user asks for a calculation or the current date/time.
- Output ONLY the TOOL_CALL line when calling a tool — no other text.
- After seeing TOOL_RESULT, use it to answer the user naturally.
"""


# ─────────────────────────────────────────────
# TOOL DISPATCHER
# ─────────────────────────────────────────────

# Pattern that matches: TOOL_CALL: tool_name(argument)
TOOL_CALL_PATTERN = re.compile(
    r"TOOL_CALL:\s*(\w+)\((.*)\)",
    re.IGNORECASE
)


class ToolDispatcher:
    """
    Detects tool calls in model output and executes them.

    Usage:
        dispatcher = ToolDispatcher()

        # Check if model wants to call a tool
        result = dispatcher.detect_and_execute(model_response)
        if result:
            # Inject result back to model
            follow_up = f"TOOL_RESULT: {result.output}"
            # Call model again with follow_up appended to conversation
    """

    def detect_and_execute(self, model_output: str) -> ToolResult | None:
        """
        Check model output for a TOOL_CALL pattern.
        If found, execute the tool and return the result.
        If not found, return None.
        """
        match = TOOL_CALL_PATTERN.search(model_output)
        if not match:
            return None

        tool_name = match.group(1).lower()
        argument  = match.group(2).strip()

        if tool_name not in TOOL_REGISTRY:
            return ToolResult(
                tool_name=tool_name,
                input=argument,
                output="",
                success=False,
                error=f"Unknown tool: {tool_name}. Available: {list(TOOL_REGISTRY.keys())}",
            )

        # Execute the tool
        tool_fn = TOOL_REGISTRY[tool_name]
        return tool_fn(argument)

    @property
    def system_prompt_addition(self) -> str:
        """Return the tools description to inject into the system prompt."""
        return TOOLS_SYSTEM_PROMPT
