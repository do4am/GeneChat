"""
Plot Gene Classification Results (ProteinChat Fig. 3a style)
=============================================================
Creates grouped bar charts comparing multiple methods (GeneChat, Claude, GPT-4, etc.)
across Accuracy, Macro F1, and Weighted F1 for each classification task.

Usage:
    # Single result file
    python plot_gene_classification.py --input gene_classification_results.json

    # Compare multiple methods (like ProteinChat Fig. 3a)
    python plot_gene_classification.py \
        --input classification_genechat.json \
               classification_claude_sonnet.json \
               classification_claude_opus.json \
               classification_gpt4.json \
               classification_random.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


TASK_DISPLAY_NAMES = {
    "organism": "Organism\nPrediction",
    "locus_type": "Locus Type\nPrediction",
    "chromosome": "Chromosome\nPrediction",
    "exon_count": "Exon Count\nPrediction",
}

METRIC_NAMES = ["accuracy", "macro_f1", "weighted_f1"]
METRIC_DISPLAY = {"accuracy": "Accuracy", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"}

# Colors for different methods (ProteinChat-style palette)
METHOD_COLORS = {
    "GeneChat":         "#4A90D9",   # Blue (primary)
    "Claude Sonnet":    "#E8913A",   # Orange
    "Claude Opus":      "#D4564E",   # Red
    "GPT-4":            "#7B8D8E",   # Gray
    "GPT-4 (genename)": "#A0A0A0",  # Light gray
    "Claude (genename)":"#F0C05A",   # Yellow
    "Random":           "#C0C0C0",   # Silver
}

METHOD_SHORT_NAMES = {
    "genechat": "GeneChat",
    "claude-sonnet-4-5-20250929": "Claude Sonnet",
    "claude-opus-4-5-20251101": "Claude Opus",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "gpt-4-turbo": "GPT-4",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "random": "Random",
}


def _get_method_label(result):
    """Extract a short method label from a result file."""
    method = result.get("method", "unknown")
    model = result.get("model", method)
    short = METHOD_SHORT_NAMES.get(model, model)
    # Append (genename) if applicable
    if "genename" in method:
        short += " (genename)"
    return short


def _get_color(label):
    """Get color for a method label."""
    for key, color in METHOD_COLORS.items():
        if key.lower() in label.lower():
            return color
    # Default color cycle
    colors = ["#4A90D9", "#E8913A", "#D4564E", "#7B8D8E", "#5CB85C", "#F0C05A", "#9B59B6"]
    return colors[hash(label) % len(colors)]


def plot_comparison(all_results, output_prefix="gene_classification"):
    """
    Create ProteinChat Fig. 3a-style grouped bar charts.
    One subplot per task, bars grouped by metric, colors by method.
    """
    # Collect all tasks across all result files
    all_tasks = []
    for res in all_results:
        for t in TASK_DISPLAY_NAMES:
            if t in res.get("tasks", {}) and t not in all_tasks:
                all_tasks.append(t)

    if not all_tasks:
        print("No task results found.")
        return

    n_tasks = len(all_tasks)
    n_methods = len(all_results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5), squeeze=False)
    axes = axes[0]

    bar_width = 0.8 / max(n_methods, 1)

    for i, task_name in enumerate(all_tasks):
        ax = axes[i]

        for j, res in enumerate(all_results):
            label = _get_method_label(res)
            color = _get_color(label)
            task_data = res.get("tasks", {}).get(task_name, {})

            if not task_data:
                continue

            metrics = [task_data.get(m, 0) for m in METRIC_NAMES]
            x = np.arange(len(METRIC_NAMES))
            offset = (j - n_methods / 2 + 0.5) * bar_width

            bars = ax.bar(x + offset, metrics, bar_width * 0.9,
                         color=color, edgecolor="white", linewidth=0.5, label=label)

            # Value labels on top
            for bar, val in zip(bars, metrics):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(TASK_DISPLAY_NAMES.get(task_name, task_name), fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(METRIC_NAMES)))
        ax.set_xticklabels([METRIC_DISPLAY[m] for m in METRIC_NAMES], fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score" if i == 0 else "", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    fig.legend(unique_handles, unique_labels, loc="upper center",
               ncol=min(len(unique_labels), 6), fontsize=9,
               bbox_to_anchor=(0.5, 1.08), frameon=False)

    fig.suptitle("Gene Classification Performance", fontsize=14, fontweight="bold", y=1.12)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = f"{output_prefix}_comparison.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_single(results, output_prefix="gene_classification"):
    """Create a single-method figure with subplots for each task."""
    tasks = [t for t in TASK_DISPLAY_NAMES if t in results.get("tasks", {})]
    if not tasks:
        print("No task results found.")
        return

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4.5 * n_tasks, 4.5), squeeze=False)
    axes = axes[0]

    label = _get_method_label(results)
    bar_color = _get_color(label)

    for i, task_name in enumerate(tasks):
        ax = axes[i]
        task_data = results["tasks"][task_name]
        metrics = [task_data.get(m, 0) for m in METRIC_NAMES]
        x = np.arange(len(METRIC_NAMES))

        bars = ax.bar(x, metrics, 0.6, color=bar_color, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, metrics):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(TASK_DISPLAY_NAMES.get(task_name, task_name), fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_DISPLAY[m] for m in METRIC_NAMES], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score" if i == 0 else "", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(f"{label} Classification Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = f"{output_prefix}_combined.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_per_task(results, output_prefix="gene_classification"):
    """Create separate plots for each task showing per-class F1 scores."""
    tasks = [t for t in TASK_DISPLAY_NAMES if t in results.get("tasks", {})]

    for task_name in tasks:
        task_data = results["tasks"][task_name]
        report = task_data.get("classification_report", {})

        classes, f1_scores, supports = [], [], []
        for cls_name, cls_data in report.items():
            if cls_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(cls_data, dict) and "f1-score" in cls_data:
                classes.append(cls_name)
                f1_scores.append(cls_data["f1-score"])
                supports.append(cls_data.get("support", 0))

        if not classes:
            continue

        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), 4.5))
        x = np.arange(len(classes))
        colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
        bars = ax.bar(x, f1_scores, color=colors, edgecolor="white", linewidth=0.5)

        for bar, val, sup in zip(bars, f1_scores, supports):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}\n(n={sup})", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(f"{TASK_DISPLAY_NAMES.get(task_name, task_name).replace(chr(10), ' ')} - Per-Class F1",
                     fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            path = f"{output_prefix}_{task_name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.close(fig)


def print_summary_table(all_results):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 85}")
    print("GENE CLASSIFICATION RESULTS")
    print(f"{'=' * 85}")

    for res in all_results:
        label = _get_method_label(res)
        print(f"\n  Method: {label}")
        print(f"  {'Task':<20} {'N':>6} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
        print(f"  {'-' * 62}")
        for s in res.get("summary", []):
            print(f"  {s['task']:<20} {s['num_samples']:>6} {s['accuracy']:>10.4f} "
                  f"{s['macro_f1']:>10.4f} {s['weighted_f1']:>12.4f}")
    print(f"\n{'=' * 85}")


def main():
    parser = argparse.ArgumentParser(description="Plot Gene Classification Results")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Path(s) to result JSON file(s). Pass multiple for comparison plot.")
    parser.add_argument("--output-prefix", default="gene_classification",
                        help="Output file prefix (default: gene_classification)")
    parser.add_argument("--per-class", action="store_true",
                        help="Also generate per-class F1 plots (only for first input)")
    args = parser.parse_args()

    all_results = []
    for path in args.input:
        with open(path) as f:
            all_results.append(json.load(f))

    print_summary_table(all_results)

    if len(all_results) == 1:
        plot_single(all_results[0], args.output_prefix)
    else:
        plot_comparison(all_results, args.output_prefix)

    if args.per_class:
        plot_per_task(all_results[0], args.output_prefix)


if __name__ == "__main__":
    main()
