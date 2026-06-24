"""
Plot Aspect-Based Evaluation — GeneChat-mRNA (DNABERT-2 vs NT-v2)
==================================================================
Creates a publication-quality bar chart comparing single-question vs.
aspect-based (10-question decomposition) SimCSE on the HM mRNA test set,
matching the style of plot_ablation_studies.py / plot_baseline_figures.py.

Data source: result_mrna/dnabert2_stage2_aspect.json (aspect-based run,
confirmed n=733) and result_mrna/eval_dnabert2_stage2_single.log /
NT-v2 Stage-2 eval logs (single-question runs, confirmed n=174).
See Report/Paper/gene_chat.tex, subsection "Aspect-based evaluation
reveals an encoder-level bottleneck" for full methodology.

Usage:
    python plot_aspect_evaluation.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 300,
})

# ── Confirmed results (HM mRNA test set, GeneChat-mRNA Stage-2 checkpoints) ──
# DNABERT-2 single-question:  checkpoint_dnabert2_stage2/.../checkpoint_50000.pth, n=174 (rule_clean)
# DNABERT-2 aspect-based:     same checkpoint, 10-question decomposition, n=733 (rule_clean + uniprot_clean)
# NT-v2 single-question:      checkpoint_ntv2_stage2/.../checkpoint_5000.pth, n=174 (rule_clean)
#   NT-v2 preceded the selective-layer LoRA strategy and was trained for far
#   fewer steps (5k vs. 50k) — a preliminary feasibility check, not a matched ablation.

LABELS = ["DNABERT-2\n(single question)", "DNABERT-2\n(aspect-based)", "NT-v2\n(single question)"]
SCORES = [0.749, 0.719, 0.642]
NS = [174, 733, 174]
COLORS = ["#1B4F72", "#2874A6", "#6E2C00"]
HATCHES = [None, "//", None]


def plot_aspect_comparison(output_dir="Report/fig/aspect_eval"):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    x = np.arange(len(LABELS))
    bars = ax.bar(x, SCORES, width=0.58, color=COLORS, edgecolor="white",
                   linewidth=0.8, hatch=HATCHES, zorder=3)

    for bar, val, n in zip(bars, SCORES, NS):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                f"n={n}", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold")

    # Reference line at the single-question DNABERT-2 score
    ax.axhline(SCORES[0], color="#1B4F72", linestyle="--", linewidth=1.1,
               alpha=0.6, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("SimCSE Score")
    ax.set_ylim(0, 0.86)
    ax.set_title("Single-Question vs. Aspect-Based Evaluation\n(GeneChat-mRNA, HM test set)",
                 fontweight="bold", fontsize=12.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/aspect_eval_comparison.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_aspect_comparison()
    print("\nDone.")
