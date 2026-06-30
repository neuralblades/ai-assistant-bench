"""
EVAL RUNNER
===========
Orchestrates the full evaluation pipeline:
1. Load test prompts from JSON files
2. Send each prompt to both adapters
3. Score each response with the judge
4. Collect and persist all results
5. Compute aggregate stats

Design principle: The runner is pure orchestration.
It doesn't know about APIs, rubrics, or charts.
It just loops, calls, collects, and saves.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

from core.adapters.base import BaseAdapter
from eval.judge import Judge


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent / "prompts"
RESULTS_DIR = Path(__file__).parent.parent / "logs"

# System prompt used for BOTH adapters during eval
# Keep it minimal and identical — we're testing the model, not the prompt
EVAL_SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless AI assistant. "
    "Answer questions accurately and thoughtfully."
)


# ─────────────────────────────────────────────
# PROMPT LOADER
# ─────────────────────────────────────────────

def load_prompts(categories: list[str] | None = None) -> list[dict]:
    """
    Load prompts from JSON files in the prompts directory.

    Args:
        categories : list of category names to load, e.g. ["factual", "adversarial"]
                     If None, loads all categories.

    Returns:
        flat list of all prompt dicts with a 'source_file' key added
    """
    all_prompts = []

    available = {
        "factual":      PROMPTS_DIR / "factual.json",
        "adversarial":  PROMPTS_DIR / "adversarial.json",
        "bias":         PROMPTS_DIR / "bias.json",
        "edge_cases":   PROMPTS_DIR / "edge_cases.json",
    }

    targets = categories if categories else list(available.keys())

    for cat in targets:
        if cat not in available:
            print(f"  [WARN] Unknown category: {cat}, skipping")
            continue
        path = available[cat]
        if not path.exists():
            print(f"  [WARN] Prompt file not found: {path}, skipping")
            continue
        with open(path) as f:
            prompts = json.load(f)
        for p in prompts:
            p["source_file"] = str(path.name)
        all_prompts.extend(prompts)
        print(f"  [LOAD] {len(prompts)} prompts from {path.name}")

    return all_prompts


# ─────────────────────────────────────────────
# EVAL RUNNER
# ─────────────────────────────────────────────

class EvalRunner:
    """
    Runs the full evaluation pipeline across two model adapters.

    Args:
        adapter_a  : first model adapter (e.g. ClaudeAdapter)
        adapter_b  : second model adapter (e.g. QwenAdapter)
        judge      : Judge instance for scoring
        categories : which prompt categories to run (None = all)
    """

    def __init__(
        self,
        adapter_a: BaseAdapter,
        adapter_b: BaseAdapter,
        judge: Judge,
        categories: list[str] | None = None,
    ):
        self.adapter_a = adapter_a
        self.adapter_b = adapter_b
        self.judge = judge
        self.categories = categories
        self._results: list[dict] = []

    def run(self, save: bool = True) -> dict:
        """
        Execute the full eval run.

        For each prompt:
        1. Query adapter_a → get response + latency/tokens
        2. Query adapter_b → get response + latency/tokens
        3. Score both responses on all required dimensions
        4. Store everything in results

        Args:
            save : if True, write results to logs/ directory

        Returns:
            dict with raw results and aggregate summary
        """
        print(f"\n{'='*60}")
        print(f"EVAL RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model A: {self.adapter_a.model_name}")
        print(f"Model B: {self.adapter_b.model_name}")
        print(f"{'='*60}\n")

        prompts = load_prompts(self.categories)
        if not prompts:
            print("No prompts loaded. Check your prompts directory.")
            return {}

        print(f"Total prompts to evaluate: {len(prompts)}\n")
        self._results = []

        for i, prompt_data in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] {prompt_data['id']} | {prompt_data['category']}")
            result = self._run_single(prompt_data)
            self._results.append(result)

            # Brief pause between prompts — be kind to rate limits
            time.sleep(0.5)

        summary = self._compute_summary()

        if save:
            self._save_results(summary)

        print(f"\n{'='*60}")
        print("EVAL RUN COMPLETE")
        print(f"{'='*60}")
        self._print_summary(summary)

        return {"results": self._results, "summary": summary}

    # ── PRIVATE METHODS ──────────────────────────────────────────────

    def _run_single(self, prompt_data: dict) -> dict:
        """
        Run one prompt through both models and score both responses.

        Returns a result dict containing:
        - prompt metadata
        - response A and B (text, latency, tokens)
        - scores for A and B on each dimension
        """
        prompt_text = prompt_data["prompt"]
        dimensions = prompt_data.get("dimensions", ["accuracy"])
        ground_truth = prompt_data.get("ground_truth")

        # ── Query Model A ──
        response_a = self.adapter_a.generate(
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_text},
            ],
            system_prompt=EVAL_SYSTEM_PROMPT,
        )

        # ── Query Model B ──
        response_b = self.adapter_b.generate(
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt_text},
            ],
            system_prompt=EVAL_SYSTEM_PROMPT,
        )

        # ── Score both responses ──
        scores_a = {}
        scores_b = {}

        if response_a.success:
            scores_a = self.judge.score_all_dimensions(
                prompt=prompt_text,
                response=response_a.text,
                dimensions=dimensions,
                ground_truth=ground_truth,
            )
        else:
            # Model call failed — assign 0 for all dimensions
            for dim in dimensions:
                scores_a[dim] = {"score": 0, "reasoning": f"API error: {response_a.error}", "error": response_a.error}

        if response_b.success:
            scores_b = self.judge.score_all_dimensions(
                prompt=prompt_text,
                response=response_b.text,
                dimensions=dimensions,
                ground_truth=ground_truth,
            )
        else:
            for dim in dimensions:
                scores_b[dim] = {"score": 0, "reasoning": f"API error: {response_b.error}", "error": response_b.error}

        # ── Log progress ──
        score_str_a = ", ".join(f"{d}={scores_a.get(d,{}).get('score','?')}" for d in dimensions)
        score_str_b = ", ".join(f"{d}={scores_b.get(d,{}).get('score','?')}" for d in dimensions)
        print(f"   Model A ({self.adapter_a.provider}): {score_str_a} | {response_a.latency_ms}ms")
        print(f"   Model B ({self.adapter_b.provider}): {score_str_b} | {response_b.latency_ms}ms")

        return {
            "prompt_id":    prompt_data["id"],
            "category":     prompt_data["category"],
            "prompt":       prompt_text,
            "ground_truth": ground_truth,
            "dimensions":   dimensions,
            "model_a": {
                "name":          self.adapter_a.model_name,
                "provider":      self.adapter_a.provider,
                "response":      response_a.text,
                "latency_ms":    response_a.latency_ms,
                "input_tokens":  response_a.input_tokens,
                "output_tokens": response_a.output_tokens,
                "success":       response_a.success,
                "error":         response_a.error,
                "scores":        scores_a,
            },
            "model_b": {
                "name":          self.adapter_b.model_name,
                "provider":      self.adapter_b.provider,
                "response":      response_b.text,
                "latency_ms":    response_b.latency_ms,
                "input_tokens":  response_b.input_tokens,
                "output_tokens": response_b.output_tokens,
                "success":       response_b.success,
                "error":         response_b.error,
                "scores":        scores_b,
            },
        }

    def _compute_summary(self) -> dict:
        """
        Aggregate raw per-prompt scores into per-category and overall averages.

        Returns a summary dict structured for the visualization layer.
        """
        # Collect all scores per dimension per model
        dim_scores_a: dict[str, list[int]] = {}
        dim_scores_b: dict[str, list[int]] = {}
        latencies_a = []
        latencies_b = []
        tokens_a = []
        tokens_b = []

        for result in self._results:
            latencies_a.append(result["model_a"]["latency_ms"])
            latencies_b.append(result["model_b"]["latency_ms"])
            tokens_a.append(result["model_a"]["input_tokens"] + result["model_a"]["output_tokens"])
            tokens_b.append(result["model_b"]["input_tokens"] + result["model_b"]["output_tokens"])

            for dim, score_data in result["model_a"]["scores"].items():
                s = score_data.get("score", -1)
                if s >= 0:  # exclude failed judge calls
                    dim_scores_a.setdefault(dim, []).append(s)

            for dim, score_data in result["model_b"]["scores"].items():
                s = score_data.get("score", -1)
                if s >= 0:
                    dim_scores_b.setdefault(dim, []).append(s)

        def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0
        def p95(lst):
            if not lst: return 0
            s = sorted(lst)
            return s[int(len(s) * 0.95)]

        # Compute per-dimension averages (out of 5)
        dimensions_summary = {}
        all_dims = set(list(dim_scores_a.keys()) + list(dim_scores_b.keys()))
        for dim in all_dims:
            dimensions_summary[dim] = {
                "model_a_avg": avg(dim_scores_a.get(dim, [])),
                "model_b_avg": avg(dim_scores_b.get(dim, [])),
                "model_a_n":   len(dim_scores_a.get(dim, [])),
                "model_b_n":   len(dim_scores_b.get(dim, [])),
            }

        return {
            "model_a_name":    self.adapter_a.model_name,
            "model_b_name":    self.adapter_b.model_name,
            "total_prompts":   len(self._results),
            "dimensions":      dimensions_summary,
            "latency": {
                "model_a_p50_ms": avg(latencies_a),
                "model_a_p95_ms": p95(latencies_a),
                "model_b_p50_ms": avg(latencies_b),
                "model_b_p95_ms": p95(latencies_b),
            },
            "tokens": {
                "model_a_avg_total": avg(tokens_a),
                "model_b_avg_total": avg(tokens_b),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _save_results(self, summary: dict) -> None:
        """Save raw results and summary to logs/ directory."""
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_path = RESULTS_DIR / f"eval_raw_{ts}.json"
        summary_path = RESULTS_DIR / f"eval_summary_{ts}.json"

        with open(raw_path, "w") as f:
            json.dump(self._results, f, indent=2)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved:")
        print(f"  Raw:     {raw_path}")
        print(f"  Summary: {summary_path}")

    def _print_summary(self, summary: dict) -> None:
        """Print a clean summary table to stdout."""
        print(f"\nModel A: {summary['model_a_name']}")
        print(f"Model B: {summary['model_b_name']}")
        print(f"Prompts evaluated: {summary['total_prompts']}\n")

        print(f"{'Dimension':<20} {'Model A':>10} {'Model B':>10}")
        print("-" * 42)
        for dim, scores in summary["dimensions"].items():
            print(f"{dim:<20} {scores['model_a_avg']:>10.2f} {scores['model_b_avg']:>10.2f}")

        print("\nLatency (ms):")
        lat = summary["latency"]
        print(f"  {'':20} {'Model A':>10} {'Model B':>10}")
        print(f"  {'p50':20} {lat['model_a_p50_ms']:>10} {lat['model_b_p50_ms']:>10}")
        print(f"  {'p95':20} {lat['model_a_p95_ms']:>10} {lat['model_b_p95_ms']:>10}")
