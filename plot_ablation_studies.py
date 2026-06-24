"""
Plot Ablation Studies — Sequence Length & Encoder Comparison
============================================================
Creates publication-quality figures matching the GO classification plot style.

  1. Sequence length ablation: dual-panel line plot (lexical + semantic)
  2. Encoder ablation: dual-panel grouped bar chart (lexical + semantic)

Usage:
    python plot_ablation_studies.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Publication-quality settings — consistent across all figures
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
})

# Color palette — consistent with existing figures
PALETTE = {
    "BLEU-1":        "#1B4F72",   # dark navy
    "BLEU-2":        "#2874A6",   # medium blue
    "BLEU-3":        "#3498DB",   # blue
    "BLEU-4":        "#85C1E9",   # light blue
    "METEOR":        "#6E2C00",   # dark brown
    "BERTScore-F1":  "#1B4F72",   # dark navy
    "SimCSE":        "#6E2C00",   # dark brown
}

MARKERS = {
    "BLEU-1":        "o",
    "BLEU-2":        "s",
    "BLEU-3":        "D",
    "BLEU-4":        "^",
    "METEOR":        "v",
    "BERTScore-F1":  "o",
    "SimCSE":        "s",
}

# Line styles for B&W distinguishability
LINESTYLES = {
    "BLEU-1":        "-",
    "BLEU-2":        "--",
    "BLEU-3":        "-.",
    "BLEU-4":        ":",
    "METEOR":        "-",
    "BERTScore-F1":  "-",
    "SimCSE":        "--",
}

# Hatching for bar charts (encoder ablation)
METRIC_HATCHES = {
    "BLEU-1":        None,
    "BLEU-2":        "//",
    "BLEU-3":        "\\\\",
    "BLEU-4":        "xx",
    "METEOR":        "..",
    "BERTScore-F1":  None,
    "SimCSE":        "//",
}

# ── Sequence length ablation data ──
SEQ_LENGTHS = [1, 10, 100, 1000, 10000, 100000, 160000]
SEQ_LABELS = ["$10^0$", "$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$1.6{\\times}10^5$"]

SEQ_DATA = {
    "BLEU-1":       ([0.170, 0.176, 0.184, 0.190, 0.191, 0.193, 0.194],
                     [0.014, 0.013, 0.012, 0.011, 0.011, 0.010, 0.010]),
    "BLEU-2":       ([0.120, 0.124, 0.130, 0.135, 0.137, 0.138, 0.138],
                     [0.011, 0.010, 0.010, 0.009, 0.009, 0.009, 0.009]),
    "BLEU-3":       ([0.091, 0.095, 0.100, 0.104, 0.105, 0.106, 0.107],
                     [0.009, 0.009, 0.008, 0.008, 0.008, 0.008, 0.008]),
    "BLEU-4":       ([0.070, 0.073, 0.077, 0.080, 0.081, 0.081, 0.082],
                     [0.008, 0.008, 0.007, 0.007, 0.007, 0.007, 0.007]),
    "METEOR":       ([0.240, 0.249, 0.259, 0.267, 0.270, 0.271, 0.273],
                     [0.012, 0.011, 0.011, 0.010, 0.010, 0.010, 0.010]),
    "BERTScore-F1": ([0.584, 0.590, 0.596, 0.602, 0.609, 0.616, 0.620],
                     [0.015, 0.014, 0.013, 0.012, 0.011, 0.011, 0.010]),
    "SimCSE":       ([0.813, 0.820, 0.827, 0.835, 0.842, 0.855, 0.860],
                     [0.012, 0.011, 0.011, 0.010, 0.010, 0.009, 0.009]),
}

# ── Encoder ablation data ──
ENCODERS = ["DNABERT-2", "DNABERT", "HyenaDNA", "No Encoder"]

ENC_DATA = {
    "BLEU-1":       ([0.240, 0.241, 0.216, 0.128],
                     [0.014, 0.015, 0.018, 0.012]),
    "BLEU-2":       ([0.167, 0.147, 0.137, 0.027],
                     [0.012, 0.013, 0.014, 0.008]),
    "BLEU-3":       ([0.123, 0.097, 0.092, 0.010],
                     [0.010, 0.011, 0.012, 0.007]),
    "BLEU-4":       ([0.088, 0.056, 0.057, 0.006],
                     [0.009, 0.008, 0.009, 0.003]),
    "METEOR":       ([0.401, 0.392, 0.349, 0.114],
                     [0.013, 0.014, 0.016, 0.011]),
    "BERTScore-F1": ([0.629, 0.611, 0.611, 0.476],
                     [0.011, 0.012, 0.012, 0.014]),
    "SimCSE":       ([0.856, 0.834, 0.836, 0.669],
                     [0.010, 0.011, 0.012, 0.015]),
}

ENCODER_COLORS = {
    "DNABERT-2":  "#1B4F72",   # dark navy
    "DNABERT":    "#2874A6",   # medium blue
    "HyenaDNA":   "#6E2C00",   # dark brown
    "No Encoder": "#999999",   # gray
}


def plot_seq_length_ablation(output_dir="Report/fig"):
    """Dual-panel line plot: lexical metrics (left) + semantic metrics (right)."""

    fig, (ax_lex, ax_sem) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(SEQ_LENGTHS))

    # ── Left panel: Lexical metrics ──
    lexical_metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR"]
    for metric in lexical_metrics:
        vals, errs = SEQ_DATA[metric]
        ax_lex.errorbar(x, vals, yerr=errs,
                        marker=MARKERS[metric], markersize=6,
                        color=PALETTE[metric], linewidth=1.8,
                        linestyle=LINESTYLES[metric],
                        capsize=3, capthick=1, label=metric,
                        markeredgecolor="white", markeredgewidth=0.8,
                        zorder=3)

    ax_lex.set_xticks(x)
    ax_lex.set_xticklabels(SEQ_LABELS)
    ax_lex.set_xlabel("Sequence Length (nucleotides)")
    ax_lex.set_ylabel("Score")
    ax_lex.set_title("Lexical Metrics", fontweight="bold")
    ax_lex.set_ylim(0.04, 0.32)
    ax_lex.spines["top"].set_visible(False)
    ax_lex.spines["right"].set_visible(False)
    ax_lex.yaxis.grid(True, alpha=0.3, zorder=0)
    ax_lex.set_axisbelow(True)
    ax_lex.legend(loc="lower right", frameon=True, framealpha=0.9,
                  edgecolor="gray")

    # ── Right panel: Semantic metrics ──
    semantic_metrics = ["BERTScore-F1", "SimCSE"]
    for metric in semantic_metrics:
        vals, errs = SEQ_DATA[metric]
        ax_sem.errorbar(x, vals, yerr=errs,
                        marker=MARKERS[metric], markersize=7,
                        color=PALETTE[metric], linewidth=2.2,
                        linestyle=LINESTYLES[metric],
                        capsize=3, capthick=1, label=metric,
                        markeredgecolor="white", markeredgewidth=0.8,
                        zorder=3)

    ax_sem.set_xticks(x)
    ax_sem.set_xticklabels(SEQ_LABELS)
    ax_sem.set_xlabel("Sequence Length (nucleotides)")
    ax_sem.set_ylabel("Score")
    ax_sem.set_title("Semantic Metrics", fontweight="bold")
    ax_sem.set_ylim(0.50, 0.92)
    ax_sem.spines["top"].set_visible(False)
    ax_sem.spines["right"].set_visible(False)
    ax_sem.yaxis.grid(True, alpha=0.3, zorder=0)
    ax_sem.set_axisbelow(True)
    ax_sem.legend(loc="lower right", frameon=True, framealpha=0.9,
                  edgecolor="gray")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/seq_length_ablation.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_encoder_ablation(output_dir="Report/fig"):
    """Dual-panel grouped bar chart: lexical metrics (left) + semantic metrics (right)."""

    fig, (ax_lex, ax_sem) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left panel: Lexical metrics ──
    lexical_metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR"]
    n_groups = len(ENCODERS)
    n_bars = len(lexical_metrics)
    bar_width = 0.14
    x = np.arange(n_groups)

    for i, metric in enumerate(lexical_metrics):
        vals, errs = ENC_DATA[metric]
        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax_lex.bar(x + offset, vals, bar_width, yerr=errs,
                          color=PALETTE[metric], edgecolor="white",
                          linewidth=0.6, hatch=METRIC_HATCHES[metric],
                          label=metric,
                          capsize=2, error_kw={"linewidth": 0.8},
                          zorder=3)
        # Rotated value labels on bars — positioned above error bar cap
        for bar, val, err in zip(bars, vals, errs):
            if val > 0.005:
                ax_lex.text(bar.get_x() + bar.get_width() / 2,
                            val + err + 0.006,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=8, fontweight="bold",
                            color=PALETTE[metric], rotation=90)

    ax_lex.set_xticks(x)
    ax_lex.set_xticklabels(ENCODERS)
    ax_lex.set_ylabel("Score")
    ax_lex.set_title("Lexical Metrics", fontweight="bold")
    ax_lex.set_ylim(0, 0.55)
    ax_lex.spines["top"].set_visible(False)
    ax_lex.spines["right"].set_visible(False)
    ax_lex.yaxis.grid(True, alpha=0.3, zorder=0)
    ax_lex.set_axisbelow(True)
    ax_lex.legend(loc="upper right", frameon=True, framealpha=0.9,
                  edgecolor="gray", ncol=1)

    # ── Right panel: Semantic metrics ──
    semantic_metrics = ["BERTScore-F1", "SimCSE"]
    n_bars_sem = len(semantic_metrics)
    bar_width_sem = 0.28
    x_sem = np.arange(n_groups)

    for i, metric in enumerate(semantic_metrics):
        vals, errs = ENC_DATA[metric]
        offset = (i - (n_bars_sem - 1) / 2) * bar_width_sem
        bars = ax_sem.bar(x_sem + offset, vals, bar_width_sem, yerr=errs,
                          color=PALETTE[metric], edgecolor="white",
                          linewidth=0.6, hatch=METRIC_HATCHES[metric],
                          label=metric,
                          capsize=3, error_kw={"linewidth": 1},
                          zorder=3)
        # Rotated value labels on bars — positioned above error bar cap
        for bar, val, err in zip(bars, vals, errs):
            ax_sem.text(bar.get_x() + bar.get_width() / 2,
                        val + err + 0.008,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                        color=PALETTE[metric], rotation=90)

    ax_sem.set_xticks(x_sem)
    ax_sem.set_xticklabels(ENCODERS)
    ax_sem.set_ylabel("Score")
    ax_sem.set_title("Semantic Metrics", fontweight="bold")
    ax_sem.set_ylim(0, 1.10)
    ax_sem.spines["top"].set_visible(False)
    ax_sem.spines["right"].set_visible(False)
    ax_sem.yaxis.grid(True, alpha=0.3, zorder=0)
    ax_sem.set_axisbelow(True)
    ax_sem.legend(loc="upper right", frameon=True, framealpha=0.9,
                  edgecolor="gray")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/encoder_ablation.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    output_dir = "Report/fig/ablation"
    plot_seq_length_ablation(output_dir)
    plot_encoder_ablation(output_dir)
    print("\nDone. All ablation plots saved.")
