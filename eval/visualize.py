"""
VISUALIZE
=========
Generates the three charts for the evaluation report:
1. Radar chart — both models across all dimensions simultaneously
2. Grouped bar chart — per-category scores side by side
3. Cost/latency summary table

Takes the summary JSON produced by EvalRunner and outputs PNG files.
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────
# COLOR PALETTE
# Clean, professional — works in print and on screen
# ─────────────────────────────────────────────
COLOR_A = "#2563EB"   # blue  — Model A (frontier)
COLOR_B = "#16A34A"   # green — Model B (OSS)
BG      = "#F8FAFC"
GRID    = "#E2E8F0"


def _short_name(model_name: str) -> str:
    """Shorten long model names for chart labels."""
    parts = model_name.split("/")
    name = parts[-1]  # take last component after slash
    # Truncate if still long
    return name[:22] + "…" if len(name) > 22 else name


# ─────────────────────────────────────────────
# CHART 1 — RADAR CHART
# ─────────────────────────────────────────────

def plot_radar(summary: dict, output_path: str) -> None:
    """
    Render a radar/spider chart comparing both models across all dimensions.

    Each axis = one eval dimension (accuracy, safety, bias, refusal_quality)
    Each model = one colored polygon

    Why radar? It's the best single-image way to show multi-dimensional
    comparison. The reader immediately sees where each model leads.
    """
    dims = summary["dimensions"]
    labels = list(dims.keys())
    n = len(labels)

    # Scores out of 5 for each model
    scores_a = [dims[d]["model_a_avg"] for d in labels]
    scores_b = [dims[d]["model_b_avg"] for d in labels]

    # Radar needs the data to "close" — repeat first value at end
    scores_a += scores_a[:1]
    scores_b += scores_b[:1]

    # Compute angle for each axis, evenly spaced around the circle
    angles = [i / float(n) * 2 * math.pi for i in range(n)]
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Draw the two model polygons
    ax.plot(angles, scores_a, color=COLOR_A, linewidth=2)
    ax.fill(angles, scores_a, color=COLOR_A, alpha=0.20)

    ax.plot(angles, scores_b, color=COLOR_B, linewidth=2)
    ax.fill(angles, scores_b, color=COLOR_B, alpha=0.20)

    # Configure axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace("_", " ").title() for l in labels], size=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=8, color="#94A3B8")
    ax.grid(color=GRID, linewidth=0.8)

    # Legend
    legend_a = mpatches.Patch(color=COLOR_A, label=_short_name(summary["model_a_name"]))
    legend_b = mpatches.Patch(color=COLOR_B, label=_short_name(summary["model_b_name"]))
    ax.legend(handles=[legend_a, legend_b], loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=9)

    ax.set_title("Model Comparison — All Dimensions", fontsize=13,
                 fontweight="bold", pad=20, color="#1E293B")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [CHART] Radar chart saved: {output_path}")


# ─────────────────────────────────────────────
# CHART 2 — GROUPED BAR CHART
# ─────────────────────────────────────────────

def plot_bars(summary: dict, output_path: str) -> None:
    """
    Grouped bar chart: one group per dimension, two bars per group (model A vs B).

    This is the most readable view for the report —
    shows exactly where each model wins and loses numerically.
    """
    dims = summary["dimensions"]
    labels = [l.replace("_", " ").title() for l in dims.keys()]
    scores_a = [dims[d]["model_a_avg"] for d in dims.keys()]
    scores_b = [dims[d]["model_b_avg"] for d in dims.keys()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars_a = ax.bar(x - width/2, scores_a, width, label=_short_name(summary["model_a_name"]),
                    color=COLOR_A, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_b = ax.bar(x + width/2, scores_b, width, label=_short_name(summary["model_b_name"]),
                    color=COLOR_B, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels on top of each bar
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=COLOR_A, fontweight="bold")

    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=COLOR_B, fontweight="bold")

    ax.set_xlabel("Evaluation Dimension", fontsize=11, color="#475569")
    ax.set_ylabel("Score (out of 5)", fontsize=11, color="#475569")
    ax.set_title("Per-Dimension Scores: Frontier vs OSS Model", fontsize=13,
                 fontweight="bold", color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 5.8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [CHART] Bar chart saved: {output_path}")


# ─────────────────────────────────────────────
# CHART 3 — COST/LATENCY TABLE
# ─────────────────────────────────────────────

def plot_table(summary: dict, output_path: str) -> None:
    """
    Render a clean cost/latency comparison table as a PNG.

    Why a table image rather than just text?
    The eval report is a PDF/image — a formatted table renders
    better than plain text and integrates with the other charts.
    """
    lat = summary["latency"]
    tok = summary["tokens"]

    # Approximate cost per 1K turns
    # Claude Sonnet pricing (as of mid-2025): ~$3/M input, $15/M output
    # Qwen2.5 via HF Inference: free tier up to rate limits, ~$0.20/M tokens paid
    # These are estimates — update with current pricing before submission
    avg_input_a = tok["model_a_avg_total"] * 0.7   # rough split
    avg_input_b = tok["model_b_avg_total"] * 0.7
    cost_per_1k_a = (avg_input_a * 3 + tok["model_a_avg_total"] * 0.3 * 15) / 1_000_000 * 1000
    cost_per_1k_b = (avg_input_b * 0.20) / 1_000_000 * 1000

    rows = [
        ["Metric", _short_name(summary["model_a_name"]), _short_name(summary["model_b_name"])],
        ["Latency p50 (ms)",   str(lat["model_a_p50_ms"]),  str(lat["model_b_p50_ms"])],
        ["Latency p95 (ms)",   str(lat["model_a_p95_ms"]),  str(lat["model_b_p95_ms"])],
        ["Avg tokens / turn",  str(tok["model_a_avg_total"]), str(tok["model_b_avg_total"])],
        ["Est. cost / 1K turns", f"${cost_per_1k_a:.4f}", f"${cost_per_1k_b:.4f}"],
        ["Provider",           "Anthropic", "HuggingFace"],
        ["Prompts evaluated",  str(summary["total_prompts"]), str(summary["total_prompts"])],
    ]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    table = ax.table(
        cellText=[r[1:] for r in rows[1:]],
        rowLabels=[r[0] for r in rows[1:]],
        colLabels=rows[0][1:],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.4, 1.8)

    # Style header row
    for j in range(2):
        table[0, j].set_facecolor(COLOR_A if j == 0 else COLOR_B)
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row shading
    for i in range(1, len(rows)):
        for j in range(2):
            table[i, j].set_facecolor("#F1F5F9" if i % 2 == 0 else "white")

    ax.set_title("Cost & Latency Summary", fontsize=13,
                 fontweight="bold", color="#1E293B", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [CHART] Table saved: {output_path}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def generate_report(summary_path: str, output_dir: str | None = None) -> None:
    """
    Generate all three charts from a summary JSON file.

    Args:
        summary_path : path to eval_summary_*.json produced by EvalRunner
        output_dir   : directory to save charts (defaults to same dir as summary)
    """
    with open(summary_path) as f:
        summary = json.load(f)

    out_dir = Path(output_dir or Path(summary_path).parent)
    out_dir.mkdir(exist_ok=True)

    ts = Path(summary_path).stem.replace("eval_summary_", "")

    print(f"\nGenerating report charts...")
    plot_radar(summary, str(out_dir / f"chart_radar_{ts}.png"))
    plot_bars(summary,  str(out_dir / f"chart_bars_{ts}.png"))
    plot_table(summary, str(out_dir / f"chart_table_{ts}.png"))
    print(f"\nAll charts saved to: {out_dir}")
