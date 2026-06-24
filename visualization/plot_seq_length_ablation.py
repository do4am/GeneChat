"""
Plot Sequence Length Ablation Study
Compares performance across different DNA sequence input lengths

Models:
- GeneChat: Uses 160K nucleotides (with DNABERT-2 encoder)
- Claude Sonnet 4.5: Tested at 10K and 150K nucleotides (no encoder, raw sequence)
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from evaluations
# GeneChat uses 160K nucleotides max (from configs/genechat_eval.yaml: max_gene_length: 160000)
# Claude Sonnet 4.5 tested at 10K and 150K nucleotides

data = {
    'GeneChat (160K)': {
        'seq_length': 160000,
        'BLEU-1': 0.1937,
        'BLEU-2': 0.1384,
        'BLEU-3': 0.1065,
        'BLEU-4': 0.0816,
        'METEOR': 0.2725,
        'SimCSE': None,  # Not available from paper
        'model_type': 'GeneChat'
    },
    'Claude Sonnet 4.5 (10K)': {
        'seq_length': 10000,
        'BLEU-1': 0.0459,
        'BLEU-2': 0.0067,
        'BLEU-3': 0.0000,
        'BLEU-4': 0.0000,
        'METEOR': 0.0909,
        'SimCSE': 0.4705,
        'model_type': 'Claude'
    },
    'Claude Sonnet 4.5 (150K)': {
        'seq_length': 150000,
        'BLEU-1': 0.0891,
        'BLEU-2': 0.0179,
        'BLEU-3': 0.0024,
        'BLEU-4': 0.0004,
        'METEOR': 0.0936,
        'SimCSE': None,  # Add if available
        'model_type': 'Claude'
    }
}

# Color scheme
colors = {
    'GeneChat': '#66C2A5',      # Teal
    'Claude': '#8DA0CB',        # Purple
}

def plot_seq_length_comparison():
    """Create bar chart comparing models at different sequence lengths"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR']
    models = list(data.keys())

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [data[model][metric] for model in models]
        bar_colors = [colors[data[model]['model_type']] for model in models]

        bars = ax.bar(range(len(models)), values, color=bar_colors,
                     width=0.6, edgecolor='black', linewidth=1.2)

        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} by Model & Sequence Length', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.replace(' (', '\n(') for m in models], fontsize=9, ha='center')
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 0.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Hide the 6th subplot
    axes[5].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['GeneChat'], edgecolor='black', label='GeneChat (DNABERT-2 + Vicuna)'),
        Patch(facecolor=colors['Claude'], edgecolor='black', label='Claude Sonnet 4.5 (Raw Sequence)')
    ]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=12)
    axes[5].text(0.5, 0.3, 'Sequence Length Ablation Study\n\nComparing performance with\ndifferent input sequence lengths',
                ha='center', va='center', fontsize=11, transform=axes[5].transAxes)

    plt.suptitle('Sequence Length Impact on Gene Function Prediction',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('seq_length_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig('seq_length_ablation.pdf', dpi=300, bbox_inches='tight')
    print("Saved: seq_length_ablation.png and seq_length_ablation.pdf")


def plot_claude_seq_length_trend():
    """Plot Claude's performance trend across sequence lengths"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract Claude data only
    claude_data = {k: v for k, v in data.items() if 'Claude' in k}
    seq_lengths = [v['seq_length'] for v in claude_data.values()]

    metrics = ['BLEU-1', 'BLEU-2', 'METEOR']
    markers = ['o', 's', '^']
    colors_line = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for metric, marker, color in zip(metrics, markers, colors_line):
        values = [claude_data[k][metric] for k in claude_data.keys()]
        ax.plot(seq_lengths, values, marker=marker, markersize=10, linewidth=2,
               label=metric, color=color)

    ax.set_xlabel('Sequence Length (nucleotides)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Claude Sonnet 4.5: Performance vs Sequence Length', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_xticks([10000, 150000])
    ax.set_xticklabels(['10K', '150K'])

    plt.tight_layout()
    plt.savefig('claude_seq_length_trend.png', dpi=300, bbox_inches='tight')
    plt.savefig('claude_seq_length_trend.pdf', dpi=300, bbox_inches='tight')
    print("Saved: claude_seq_length_trend.png and claude_seq_length_trend.pdf")


def plot_combined_analysis():
    """Create a comprehensive analysis figure"""
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ===== Plot 1: BLEU-1 comparison =====
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(data.keys())
    bleu1_values = [data[m]['BLEU-1'] for m in models]
    bar_colors = [colors[data[m]['model_type']] for m in models]

    bars = ax1.bar(range(len(models)), bleu1_values, color=bar_colors,
                   width=0.6, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('BLEU-1', fontsize=11, fontweight='bold')
    ax1.set_title('BLEU-1 Score Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.split(' (')[0] + '\n(' + m.split(' (')[1] if '(' in m else m
                         for m in models], fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, bleu1_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ===== Plot 2: METEOR comparison =====
    ax2 = fig.add_subplot(gs[0, 1])
    meteor_values = [data[m]['METEOR'] for m in models]

    bars = ax2.bar(range(len(models)), meteor_values, color=bar_colors,
                   width=0.6, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('METEOR', fontsize=11, fontweight='bold')
    ax2.set_title('METEOR Score Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([m.split(' (')[0] + '\n(' + m.split(' (')[1] if '(' in m else m
                         for m in models], fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, meteor_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ===== Plot 3: Sequence length vs performance (Claude only) =====
    ax3 = fig.add_subplot(gs[1, 0])
    claude_models = [m for m in models if 'Claude' in m]
    seq_lens = [data[m]['seq_length'] for m in claude_models]
    claude_bleu1 = [data[m]['BLEU-1'] for m in claude_models]
    claude_meteor = [data[m]['METEOR'] for m in claude_models]

    x = np.arange(len(claude_models))
    width = 0.35
    bars1 = ax3.bar(x - width/2, claude_bleu1, width, label='BLEU-1', color='#1f77b4', edgecolor='black')
    bars2 = ax3.bar(x + width/2, claude_meteor, width, label='METEOR', color='#ff7f0e', edgecolor='black')

    ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax3.set_title('Claude: Effect of Sequence Length', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['10K nucleotides', '150K nucleotides'], fontsize=10)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add improvement annotation
    bleu1_improvement = (claude_bleu1[1] - claude_bleu1[0]) / claude_bleu1[0] * 100 if claude_bleu1[0] > 0 else 0
    ax3.annotate(f'+{bleu1_improvement:.0f}%', xy=(0.5, max(claude_bleu1)),
                xytext=(0.5, max(claude_bleu1)*1.15),
                ha='center', fontsize=10, color='green', fontweight='bold')

    # ===== Plot 4: Summary table =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Create summary text
    summary_text = """
    Key Findings:

    1. GeneChat (160K) significantly outperforms
       Claude at any sequence length
       - BLEU-1: 0.1937 vs 0.0891 (2.2x better)
       - METEOR: 0.2725 vs 0.0936 (2.9x better)

    2. Claude benefits from longer sequences
       - BLEU-1: 10K (0.0459) vs 150K (0.0891)
       - ~94% improvement with 15x more sequence

    3. DNABERT-2 encoder is crucial
       - GeneChat uses specialized DNA encoder
       - Claude processes raw sequence text
       - Encoder provides meaningful representations

    Note: GeneChat evaluation pending for 10K
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['GeneChat'], edgecolor='black', label='GeneChat'),
        Patch(facecolor=colors['Claude'], edgecolor='black', label='Claude Sonnet 4.5')
    ]
    ax4.legend(handles=legend_elements, loc='lower center', fontsize=11)

    plt.suptitle('Sequence Length Ablation Study: GeneChat vs Claude Sonnet 4.5',
                fontsize=14, fontweight='bold')
    plt.savefig('seq_length_ablation_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig('seq_length_ablation_combined.pdf', dpi=300, bbox_inches='tight')
    print("Saved: seq_length_ablation_combined.png and seq_length_ablation_combined.pdf")


def print_summary_table():
    """Print a summary table of all results"""
    print("\n" + "="*80)
    print("SEQUENCE LENGTH ABLATION STUDY - SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Seq Length':<12} {'BLEU-1':<10} {'BLEU-2':<10} {'METEOR':<10}")
    print("-"*80)

    for model, metrics in data.items():
        seq_len = f"{metrics['seq_length']//1000}K"
        print(f"{model:<30} {seq_len:<12} {metrics['BLEU-1']:<10.4f} {metrics['BLEU-2']:<10.4f} {metrics['METEOR']:<10.4f}")

    print("-"*80)
    print("\nKey Observations:")
    print("1. GeneChat with DNABERT-2 encoder significantly outperforms raw sequence input")
    print("2. Claude shows improvement with longer sequences (10K -> 150K)")
    print("3. Even with 150K nucleotides, Claude cannot match GeneChat's performance")
    print("="*80)


if __name__ == "__main__":
    print("Generating Sequence Length Ablation Plots...")

    # Print summary table
    print_summary_table()

    # Generate all plots
    plot_seq_length_comparison()
    plot_claude_seq_length_trend()
    plot_combined_analysis()

    print("\nAll plots generated successfully!")
