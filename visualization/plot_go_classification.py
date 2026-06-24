"""
Plot GO Classification Results — GeneChat vs Gemini (ProteinChat Fig. 3a style)
================================================================================
Creates:
  1. Grouped bar chart: GeneChat vs Gemini per GO task per metric
  2. Per-class F1 breakdown comparing both models side by side

Usage:
    python plot_go_classification.py \
        --genechat go_classification_results_1000.json \
        --gemini go_classification_gemini_dna.json
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

TASK_ORDER = ["molecular_function", "biological_process", "cellular_component"]
TASK_LABELS = {
    "molecular_function": "Molecular\nFunction",
    "biological_process": "Biological\nProcess",
    "cellular_component": "Cellular\nComponent",
}
CHANCE_LEVELS = {
    "molecular_function": 1 / 7,
    "biological_process": 1 / 8,
    "cellular_component": 1 / 6,
}

METRICS = ["accuracy", "macro_f1", "weighted_f1"]
METRIC_LABELS = ["Accuracy", "Macro F1", "Weighted F1"]

# Dark color palette: blue/brown tones
MODEL_COLORS = {
    "GeneChat": "#1B4F72",       # dark navy blue
    "Gemini 2.5 Flash": "#6E2C00",  # dark brown
}
MODEL_HATCHES = {
    "GeneChat": None,
    "Gemini 2.5 Flash": "//",
}


def plot_main(gc_data, gem_data, output_dir="Report/fig"):
    """Grouped bar chart: models x metrics x tasks (ProteinChat Fig. 3a style)."""
    gc_summary = {s["task"]: s for s in gc_data["summary"]}
    gem_summary = {s["task"]: s for s in gem_data["summary"]}
    tasks = [t for t in TASK_ORDER if t in gc_summary]

    models = ["GeneChat", "Gemini 2.5 Flash"]
    summaries = [gc_summary, gem_summary]
    n_models = len(models)
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(1, n_metrics, figsize=(14, 4.5), sharey=True)

    x = np.arange(len(tasks))
    bar_width = 0.30

    for mi, (metric, mlabel) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = axes[mi]
        offsets = [(j - (n_models - 1) / 2) * bar_width for j in range(n_models)]

        for j, (model, summary) in enumerate(zip(models, summaries)):
            values = [summary[t][metric] for t in tasks]
            color = MODEL_COLORS[model]
            hatch = MODEL_HATCHES[model]
            bars = ax.bar(x + offsets[j], values, bar_width, color=color,
                          edgecolor="white", linewidth=0.8, label=model if mi == 0 else "",
                          hatch=hatch, zorder=3)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color=color, rotation=90)

        # Random baseline markers
        for k, t in enumerate(tasks):
            chance = CHANCE_LEVELS[t]
            ax.hlines(chance, x[k] - 0.4, x[k] + 0.4, colors="#888888",
                      linestyles="dashed", linewidth=0.8, zorder=2)
            if mi == 0:
                ax.text(x[k] + 0.42, chance, f"{chance:.2f}", fontsize=8,
                        va="center", color="#888888")

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
        ax.set_title(mlabel, fontweight="bold")
        if mi == 0:
            ax.set_ylabel("Score")
        ax.set_ylim(0, 0.58)
        ax.set_xlim(-0.5, len(tasks) - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=n_models,
               frameon=True, framealpha=0.9, edgecolor="gray",
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/go_classification.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_perclass(gc_data, gem_data, output_dir="Report/fig"):
    """Per-class F1 comparison: GeneChat vs Gemini side by side for each GO task."""
    gc_tasks = gc_data["tasks"]
    gem_tasks = gem_data["tasks"]
    tasks = [t for t in TASK_ORDER if t in gc_tasks and t in gem_tasks]

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    gc_color = MODEL_COLORS["GeneChat"]
    gem_color = MODEL_COLORS["Gemini 2.5 Flash"]

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        gc_report = gc_tasks[task]["classification_report"]
        gem_report = gem_tasks[task]["classification_report"]

        # Get all classes from GeneChat report
        classes, gc_f1, gem_f1, supports = [], [], [], []
        for cls_name, cls_data in gc_report.items():
            if cls_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(cls_data, dict) and "f1-score" in cls_data:
                classes.append(cls_name)
                gc_f1.append(cls_data["f1-score"])
                supports.append(cls_data.get("support", 0))
                # Get matching Gemini F1
                gem_cls = gem_report.get(cls_name, {})
                gem_f1.append(gem_cls.get("f1-score", 0) if isinstance(gem_cls, dict) else 0)

        # Sort by GeneChat support descending
        order = np.argsort(supports)[::-1]
        classes = [classes[i] for i in order]
        gc_f1 = [gc_f1[i] for i in order]
        gem_f1 = [gem_f1[i] for i in order]
        supports = [supports[i] for i in order]

        x_pos = np.arange(len(classes))
        bar_w = 0.35

        bars_gc = ax.bar(x_pos - bar_w / 2, gc_f1, bar_w, color=gc_color,
                         edgecolor="white", linewidth=0.5, label="GeneChat" if idx == 0 else "")
        bars_gem = ax.bar(x_pos + bar_w / 2, gem_f1, bar_w, color=gem_color,
                          edgecolor="white", linewidth=0.5, hatch="//",
                          label="Gemini 2.5 Flash" if idx == 0 else "")

        # Value labels (rotated)
        for bar, val in zip(bars_gc, gc_f1):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(bar.get_height(), 0) + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                        color=gc_color, fontweight="bold", rotation=90)

        for bar, val in zip(bars_gem, gem_f1):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(bar.get_height(), 0) + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                        color=gem_color, fontweight="bold", rotation=90)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Score" if idx == 0 else "")
        ax.set_title(TASK_LABELS[task].replace("\n", " "), fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Accuracy annotation
        gc_acc = gc_tasks[task]["accuracy"]
        gem_acc = gem_tasks[task]["accuracy"]
        ax.text(0.98, 0.95,
                f"GeneChat: {gc_acc:.3f}\nGemini: {gem_acc:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=True, framealpha=0.9, edgecolor="gray",
               bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/go_classification_perclass.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def print_comparison(gc_data, gem_data):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("GO CLASSIFICATION COMPARISON: GeneChat vs Gemini")
    print("=" * 90)
    print(f"{'Task':<25} {'Model':<20} {'Samples':>8} {'Valid':>8} {'Accuracy':>10} {'Macro F1':>10} {'Wt. F1':>10}")
    print("-" * 90)

    for task in TASK_ORDER:
        gc_t = gc_data["tasks"].get(task)
        gem_t = gem_data["tasks"].get(task)
        if gc_t:
            print(f"{gc_t['display_name']:<25} {'GeneChat':<20} {gc_t['num_samples']:>8} "
                  f"{gc_t['num_valid']:>8} {gc_t['accuracy']:>10.4f} "
                  f"{gc_t['macro_f1']:>10.4f} {gc_t['weighted_f1']:>10.4f}")
        if gem_t:
            print(f"{'':<25} {'Gemini 2.5 Flash':<20} {gem_t['num_samples']:>8} "
                  f"{gem_t['num_valid']:>8} {gem_t['accuracy']:>10.4f} "
                  f"{gem_t['macro_f1']:>10.4f} {gem_t['weighted_f1']:>10.4f}")
        chance = CHANCE_LEVELS.get(task, 0)
        print(f"{'':<25} {'Random':<20} {'':>8} {'':>8} {chance:>10.4f} {'':>10} {'':>10}")
        print("-" * 90)
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Plot GO Classification: GeneChat vs Gemini")
    parser.add_argument("--genechat", required=True, help="GeneChat results JSON")
    parser.add_argument("--gemini", required=True, help="Gemini results JSON")
    parser.add_argument("--output-dir", default="Report/fig", help="Output directory")
    args = parser.parse_args()

    with open(args.genechat) as f:
        gc_data = json.load(f)
    with open(args.gemini) as f:
        gem_data = json.load(f)

    print_comparison(gc_data, gem_data)
    plot_main(gc_data, gem_data, args.output_dir)
    plot_perclass(gc_data, gem_data, args.output_dir)


if __name__ == "__main__":
    main()
