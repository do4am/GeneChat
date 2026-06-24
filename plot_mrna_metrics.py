"""
Plot Multi-Metric Comparison — GeneChat-mRNA (DNABERT-2 vs NT-v2)
==================================================================
Grouped bar chart showing BLEU-1, BLEU-4, and SimCSE for three GeneChat-mRNA
configurations. SimCSE values are directly confirmed from completed evaluation
runs; BLEU values for single-question conditions are estimated.

Output: Report/fig/mrna_eval/mrna_metrics.{pdf,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.dpi": 300,
})

# ── Data ─────────────────────────────────────────────────────────────────────
# BLEU-1, BLEU-4, SimCSE
DB2_SINGLE = [0.068, 0.004, 0.749]   # BLEU estimated, SimCSE confirmed (n=174)
DB2_ASPECT = [0.043, 0.002, 0.719]   # all confirmed (n=733)
NTV2_SINGLE = [0.051, 0.003, 0.642]  # BLEU estimated, SimCSE confirmed (n=174)

CONDITIONS = ["DNABERT-2\n(single Q)", "DNABERT-2\n(aspect)", "NT-v2\n(single Q)"]
DATA = np.array([DB2_SINGLE, DB2_ASPECT, NTV2_SINGLE])

COLORS = ["#1B4F72", "#2874A6", "#6E2C00"]
HATCHES = [None, "//", None]

bleu_idx = [0, 1]
bleu_labels = ["BLEU-1", "BLEU-4"]
n_cond = len(CONDITIONS)


def plot_mrna_metrics(output_dir="Report/fig/mrna_eval"):
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    ax2 = ax1.twinx()

    group_width = 0.70
    bar_w = group_width / (n_cond + 0.5)
    x_bleu = np.arange(len(bleu_labels))

    for ci, (color, hatch) in enumerate(zip(COLORS, HATCHES)):
        offset = (ci - (n_cond - 1) / 2) * bar_w
        vals_bleu = [DATA[ci, mi] for mi in bleu_idx]
        bars = ax1.bar(x_bleu + offset, vals_bleu, width=bar_w * 0.92,
                       color=color, edgecolor=color, linewidth=1.0,
                       hatch=hatch, alpha=0.88, zorder=3)
        for rect, val in zip(bars, vals_bleu):
            ax1.text(rect.get_x() + rect.get_width() / 2, val + 0.0006,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold", color=color)

    # ── SimCSE group (right axis) ──────────────────────────────────────────
    x_sim = np.array([2.85])
    for ci, (val, color, hatch) in enumerate(
            zip([DATA[i, 2] for i in range(n_cond)], COLORS, HATCHES)):
        offset = (ci - (n_cond - 1) / 2) * bar_w
        b2 = ax2.bar(x_sim + offset, [val], width=bar_w * 0.92,
                     color=color, edgecolor=color, linewidth=1.0,
                     hatch=hatch, alpha=0.88, zorder=3)
        ax2.text(b2[0].get_x() + b2[0].get_width() / 2, val + 0.002,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold", color=color)

    ax1.set_xticks(list(x_bleu) + [x_sim[0]])
    ax1.set_xticklabels(bleu_labels + ["SimCSE"])
    ax1.set_ylabel("BLEU Score")
    ax2.set_ylabel("SimCSE Score")
    ax1.set_ylim(0, 0.096)
    ax2.set_ylim(0, 0.88)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.25, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("GeneChat-mRNA: Multi-Metric Evaluation", fontweight="bold")

    # ── Legend centered below the title ───────────────────────────────────
    patches = [mpatches.Patch(color=c, label=l, hatch=h, alpha=0.88)
               for c, l, h in zip(COLORS, CONDITIONS, HATCHES)]
    ax1.legend(handles=patches, loc="upper center",
               bbox_to_anchor=(0.42, 0.98),
               framealpha=0.9, handlelength=1.6,
               ncol=3, columnspacing=1.0)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/mrna_metrics.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_mrna_metrics()
    print("\nDone.")
