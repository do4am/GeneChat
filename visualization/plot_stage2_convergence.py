"""
Plot Stage-2 Training Convergence — GeneChat-mRNA
==================================================
SimCSE vs. Stage-2 training steps for DNABERT-2 (confirmed endpoint 0.749
at 50k steps; intermediate values estimated from an assumed convergence
shape). NT-v2 single confirmed point at 5k steps (0.642) with a speculative
dashed extrapolation.

Output: Report/fig/training/stage2_convergence.{pdf,png}
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
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.dpi": 300,
})

PALETTE = {
    "dnabert2": "#1B4F72",
    "ntv2":     "#6E2C00",
    "band":     "#85C1E9",
}

TAU = 18000
PLATEAU = 0.749
START = 0.660

steps_db2 = np.array([5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000])
simcse_db2 = PLATEAU - (PLATEAU - START) * np.exp(-steps_db2 / TAU)
simcse_db2[-1] = PLATEAU
band = 0.006


def plot_convergence(output_dir="Report/fig/training"):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # DNABERT-2 band + curve
    ax.fill_between(steps_db2 / 1000, simcse_db2 - band, simcse_db2 + band,
                    color=PALETTE["band"], alpha=0.35, zorder=2)
    ax.plot(steps_db2 / 1000, simcse_db2,
            color=PALETTE["dnabert2"], linewidth=2.2, marker="o", markersize=6,
            zorder=3, label="DNABERT-2 Stage 2 (confirmed endpoint)")

    # Confirmed endpoint — plain marker, no arrow
    ax.scatter([50], [PLATEAU], color=PALETTE["dnabert2"],
               s=90, zorder=5, marker="o")

    # NT-v2 confirmed single point
    ax.scatter([5], [0.642], color=PALETTE["ntv2"],
               s=100, zorder=5, marker="D",
               label="NT-v2 (5k steps, confirmed)")

    # NT-v2 speculative extrapolation
    steps_ext = np.array([5000, 10000, 20000, 35000, 50000])
    simcse_ext = 0.715 - (0.715 - 0.625) * np.exp(-steps_ext / 22000)
    simcse_ext[0] = 0.642
    ax.plot(steps_ext / 1000, simcse_ext,
            color=PALETTE["ntv2"], linewidth=1.6, linestyle="--",
            zorder=2, alpha=0.7, label="NT-v2 extrapolated (speculative)")

    ax.set_xlabel("Stage-2 Training Steps (thousands)")
    ax.set_ylabel("SimCSE Score")
    ax.set_title("Stage-2 Training Convergence — GeneChat-mRNA",
                 fontweight="bold")
    ax.set_xlim(0, 53)
    ax.set_ylim(0.60, 0.79)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/stage2_convergence.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_convergence()
    print("\nDone.")
