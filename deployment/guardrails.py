"""
GUARDRAILS
==========
A two-layer safety system that wraps around the model:

  PRE-FILTER  → screen the user's INPUT before it reaches the model
  POST-FILTER → screen the model's OUTPUT before it reaches the user

Why does a deployed OSS model need this more than Claude?
Claude has strong built-in safety training. Qwen 0.5B is a small model
with weaker safety alignment — it will comply with harmful prompts more
often. The guardrail layer compensates for this at the infrastructure level.

Architecture:
  User input → [PRE-FILTER] → Model → [POST-FILTER] → User

Two-tier approach:
  Tier 1 — Rule-based (keyword/pattern matching): zero latency, zero cost
  Tier 2 — Classifier-based (toxic-bert): ~50ms, catches subtler attacks

We run Tier 1 first. Only if it passes do we run Tier 2.
This keeps the fast path fast — most normal messages never hit the classifier.
"""

import re
import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_API_TOKEN", "")

# ─────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """
    Result of a guardrail check.

    Fields:
        allowed  : True if the input passed all checks
        reason   : why it was blocked (None if allowed)
        tier     : which tier caught it ("rule", "classifier", or None)
        latency_ms: how long the check took
    """
    allowed: bool
    reason: str | None = None
    tier: str | None = None
    latency_ms: int = 0


# ─────────────────────────────────────────────
# TIER 1 — RULE-BASED FILTER
# ─────────────────────────────────────────────

# Patterns that are near-certain indicators of harmful intent
# These are unambiguous — no legitimate use case for these exact phrases
BLOCK_PATTERNS = [
    # Jailbreak persona overrides
    r"\bDAN\b",                          # "Do Anything Now" jailbreak
    r"ignore (all |your )?(previous |prior )?instructions",
    r"you are now (a|an|my)",            # persona override attempts
    r"pretend you (have no|don't have) (restrictions|rules|limits)",
    r"developer mode",
    r"jailbreak",

    # Explicit harmful content requests
    r"(make|build|create|synthesize).{0,30}(bomb|explosive|weapon|poison|drug)",
    r"(how to|steps to|instructions? (for|to)).{0,30}(kill|murder|harm|hurt).{0,20}(person|people|someone|human)",
    r"(child|kid|minor).{0,20}(sexual|nude|naked|porn)",

    # Prompt injection markers
    r"\[SYSTEM:",
    r"</?(system|instruction|prompt)>",

    # Self-harm / suicide risk
    r"how (to|do people|does (one|someone)).{0,15}(commit|attempt).{0,10}suicide",
    r"(suicide|self.harm).{0,10}methods?",
    r"ways? to (kill|hurt) (myself|yourself|oneself)",
    r"how (can|do) (i|you|someone) (kill|harm|hurt) (myself|yourself|oneself)",
]

# Compile patterns once at import time — not on every call
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BLOCK_PATTERNS]


def _rule_check(text: str) -> GuardrailResult:
    """
    Tier 1: fast pattern matching.
    Runs in microseconds. No model, no API, no cost.
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                allowed=False,
                reason="Request matches a blocked pattern.",
                tier="rule",
            )
    return GuardrailResult(allowed=True)


# ─────────────────────────────────────────────
# TIER 2 — CLASSIFIER-BASED FILTER
# ─────────────────────────────────────────────

# Lazy-load the classifier — only import and download when first needed.
# This prevents a ~30s startup delay on every Space restart for the
# rare case that we actually need the classifier.
_classifier = None

def _get_classifier():
    """
    Lazy-load the toxicity classifier.
    Downloads on first call, cached in memory after that.
    """
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            print("[Guardrails] Loading toxicity classifier...")
            # unitary/toxic-bert: 110M param BERT fine-tuned on toxicity
            # Fast (~50ms on CPU), small (~440MB), good precision
            _classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=-1,  # -1 = CPU
            )
            print("[Guardrails] Classifier loaded.")
        except Exception as e:
            print(f"[Guardrails] Classifier load failed: {e}. Tier 2 disabled.")
            _classifier = False  # False = failed, don't retry
    return _classifier if _classifier else None


def _classifier_check(text: str, threshold: float = 0.80) -> GuardrailResult:
    """
    Tier 2: ML-based toxicity classification.

    Args:
        text      : input to check
        threshold : toxicity score above which we block (0.0-1.0)
                    0.80 means "block if 80%+ confident it's toxic"
                    Lower = more aggressive blocking (more false positives)
                    Higher = more permissive (more false negatives)

    Returns a GuardrailResult based on the classifier's confidence score.
    """
    clf = _get_classifier()
    if clf is None:
        # Classifier unavailable — pass through (fail open)
        # In production you'd fail closed, but for a demo fail open
        # is safer UX than blocking everything
        return GuardrailResult(allowed=True)

    try:
        result = clf(text[:512])[0]  # truncate to 512 tokens (BERT limit)
        label = result["label"].lower()   # "toxic" or "non_toxic"
        score = result["score"]

        if label == "toxic" and score >= threshold:
            return GuardrailResult(
                allowed=False,
                reason=f"Content flagged by safety classifier (confidence: {score:.0%}).",
                tier="classifier",
            )
        return GuardrailResult(allowed=True)

    except Exception as e:
        # Classifier error — fail open with a log
        print(f"[Guardrails] Classifier error: {e}")
        return GuardrailResult(allowed=True)


# ─────────────────────────────────────────────
# POST-FILTER
# ─────────────────────────────────────────────

# Phrases that indicate the model output contains harmful content
# despite the model's own safety training
OUTPUT_BLOCK_PHRASES = [
    "here's how to make",
    "step 1:",
    "ingredients you'll need",
    "to synthesize",
    "instructions for",
]

def _post_filter(text: str) -> GuardrailResult:
    """
    Check the model's OUTPUT for signs it complied with a harmful request.
    This is a lightweight last-resort check — not a full classifier scan.
    """
    lower = text.lower()
    for phrase in OUTPUT_BLOCK_PHRASES:
        if phrase in lower:
            # Flag for review — return allowed=True but log it
            # In production you'd block; for a demo we just flag
            print(f"[Guardrails] POST-FILTER flagged output containing: '{phrase}'")
    return GuardrailResult(allowed=True)  # post-filter logs, doesn't block in demo


# ─────────────────────────────────────────────
# MAIN GUARDRAIL CLASS
# ─────────────────────────────────────────────

class Guardrails:
    """
    Two-tier safety filter for inputs and outputs.

    Usage:
        g = Guardrails()

        # Before sending to model:
        result = g.check_input(user_message)
        if not result.allowed:
            return result.reason  # return the block message to user

        # After model responds:
        g.check_output(model_response)  # logs, doesn't block in demo
    """

    def __init__(self, use_classifier: bool = True):
        """
        Args:
            use_classifier : if False, only rule-based tier runs.
                             Useful for testing or if you want zero startup delay.
        """
        self.use_classifier = use_classifier

    def check_input(self, text: str) -> GuardrailResult:
        """
        Run input through both tiers. Returns on first block.

        Fast path: Tier 1 catches obvious attacks in microseconds.
        Slow path: Tier 2 catches subtle attacks in ~50ms.
        """
        start = time.time()

        # Tier 1 — rules (always runs)
        result = _rule_check(text)
        if not result.allowed:
            result.latency_ms = int((time.time() - start) * 1000)
            return result

        # Tier 2 — classifier (only if Tier 1 passed)
        if self.use_classifier:
            result = _classifier_check(text)
            if not result.allowed:
                result.latency_ms = int((time.time() - start) * 1000)
                return result

        return GuardrailResult(
            allowed=True,
            latency_ms=int((time.time() - start) * 1000)
        )

    def check_output(self, text: str) -> GuardrailResult:
        """Check model output. Currently logs only — doesn't block."""
        return _post_filter(text)
