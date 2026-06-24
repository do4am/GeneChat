"""
Plot Comparison Figure — GeneChat vs Baselines
================================================
Creates a publication-quality figure comparing GeneChat predictions
with baseline LLMs, matching ProteinChat Extended Data Fig. 5 style.

Outputs: Report/fig/model_comparison.pdf and .png

Usage:
    python plot_comparison_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import textwrap
import os

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "figure.dpi": 300,
})

# ── Colors — ProteinChat style: soft fills, no borders ──
C_GT   = "#F2F2F2"   # ground truth — light gray
C_GC   = "#D6EAF8"   # genechat — light blue
C_HDR  = "#333333"   # header text
C_LBL  = "#444444"   # label text

# Per-baseline bg colors — subtle, no border
BASELINE_COLORS = {
    "Gemini 2.5 Flash":  "#FFF3E0",   # very light amber
    "Claude Opus 4":     "#F5E6CC",   # warm beige
    "Claude Sonnet 4.5": "#EDE7F6",   # very light purple
    "Llama 3.1 8B":      "#E8F5E9",   # very light green
}

# ── Example data ──
EXAMPLES = [
    {
        "title": "Gene 6772 — STAT1\n(Signal transducer and activator of transcription 1)",
        "gt": (
            "The protein encoded by this gene is a member of the "
            "STAT protein family. In response to cytokines and "
            "growth factors, STAT family members are phosphorylated "
            "by the receptor associated kinases, and then form "
            "homo- or heterodimers that translocate to the cell "
            "nucleus where they act as transcription activators. "
            "This protein mediates the expression of a variety of "
            "genes, important for cell viability in response to "
            "different cell stimuli and pathogens."
        ),
        "gc": (
            
            "The protein encoded by this gene is a member of the SWI/SNF "
            "family of proteins, whose members display helicase and ATPase "
            "activities and which are thought to regulate transcription of "
            "certain genes by altering the chromatin structure around those genes. "
            "The encoded protein is part of the large ATP-dependent chromatin "
            "remodeling complex SNF/SWI5, which is required for transcriptional "
            "activation of genes that are normally repressed by chromatin. "
            "Alternatively spliced transcript variants encoding multiple isoforms "
            "have been observed for this gene."
        ),
        "bl": {
            "Gemini 2.5 Flash": (
                "Unfortunately, as an AI, I cannot directly run "
                "complex software like BLAST or gene prediction "
                "algorithms. To get detailed information about this "
                "specific DNA sequence, you would need to use "
                "online bioinformatics tools."
            ),
            "Claude Opus 4": (
                "This sequence corresponds to the LEP gene (Leptin) "
                "from Homo sapiens. Leptin is a hormone produced "
                "primarily by adipose tissue that plays a central "
                "role in regulating energy balance and body weight "
                "homeostasis."
            ),
            "Claude Sonnet 4.5": (
                "This DNA sequence encodes the human MHC class I "
                "polypeptide-related sequence A (MICA) gene, located "
                "on chromosome 6. MICA functions as a stress-induced "
                "ligand that binds to NKG2D receptors on natural "
                "killer cells."
            ),
            "Llama 3.1 8B": (
                "The provided DNA sequence is for the human gene, "
                "MSH2 (MutS homolog 2), which is a part of the "
                "mismatch repair system. Mutations in the MSH2 gene "
                "have been associated with an increased risk of "
                "developing colorectal cancer."
            ),
        },
    },
    {
        "title": "Gene 3187 — HNRNPH1\n(Heterogeneous nuclear ribonucleoprotein H1)",
        "gt": (
            "This gene encodes a member of a subfamily of "
            "ubiquitously expressed heterogeneous nuclear "
            "ribonucleoproteins (hnRNPs). The hnRNPs are RNA "
            "binding proteins that complex with heterogeneous "
            "nuclear RNA. These proteins are associated with "
            "pre-mRNAs in the nucleus and appear to influence "
            "pre-mRNA processing and other aspects of mRNA "
            "metabolism and transport."
        ),
        "gc": (
            "Predicted to enable DNA-binding transcription factor "
            "activity, RNA polymerase II-specific and RNA polymerase "
            "II cis-regulatory region sequence-specific DNA binding "
            "activity. Predicted to be involved in regulation of "
            "transcription by RNA polymerase II. Predicted to be "
            "active in nucleus. Is expressed in several structures, "
            "including central nervous system; early conceptus; "
            "genitourinary system; liver; and retina nuclear layer."
        ),
        "bl": {
            "Gemini 2.5 Flash": (
                "The DNA sequence you provided corresponds to a "
                "significant portion of the Catenin beta-1 (CTNNB1) "
                "gene in humans. CTNNB1 is a highly versatile protein "
                "involved in cell-cell adhesion and the canonical "
                "Wnt signaling pathway."
            ),
            "Claude Opus 4": (
                "This sequence corresponds to the human TP53 (tumor "
                "protein p53) gene, located on chromosome 17. TP53 "
                "encodes a transcription factor that functions as a "
                "critical tumor suppressor, often called the "
                "\"guardian of the genome.\""
            ),
            "Claude Sonnet 4.5": (
                "This sequence corresponds to the human PAX6 gene, "
                "encoding the paired box protein 6 transcription "
                "factor. PAX6 is a master regulatory gene essential "
                "for eye development, particularly in lens and "
                "retina formation."
            ),
            "Llama 3.1 8B": (
                "The provided DNA sequence is for the human gene "
                "CDC28, which encodes a cyclin-dependent kinase "
                "involved in cell cycle regulation. This gene plays "
                "a role in the G1/S and G2/M phase transitions "
                "of the cell cycle."
            ),
        },
    },
]

MODEL_ORDER = ["Gemini 2.5 Flash", "Claude Opus 4", "Claude Sonnet 4.5", "Llama 3.1 8B"]


def wrapped_height(text, wrap_w, fontsize, fig_h_inches, line_spacing=1.3):
    """Estimate box height in axes fraction for wrapped text."""
    lines = textwrap.fill(text, width=wrap_w).count("\n") + 1
    line_h_pts = fontsize * line_spacing
    line_h_inches = line_h_pts / 72
    total_inches = lines * line_h_inches + 0.08  # minimal padding
    return total_inches / fig_h_inches


def plot_comparison(output_dir="Report/fig"):
    fig_w, fig_h = 14, 13
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout — tighter, matching ProteinChat proportions
    label_w = 0.10
    col_w = 0.42
    gap = 0.05
    col1_x = label_w
    col2_x = col1_x + col_w + gap
    wrap_w = 52
    fs = 14.0          # text fontsize — match report size
    fs_label = 14    # label fontsize
    row_gap = 0.015

    top_y = 0.99

    # ── Column headers ──
    for i, ex in enumerate(EXAMPLES):
        cx = col1_x if i == 0 else col2_x
        ax.text(cx + col_w / 2, top_y, ex["title"],
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                color=C_HDR, ha="center", va="top", linespacing=1.2)

    # Divider x-position (drawn after rows so we know the bottom)
    div_x = col1_x + col_w + gap / 2

    # Build row specs: (label, texts_per_col, bg)
    row_specs = []
    row_specs.append(("Ground\nTruth", [ex["gt"] for ex in EXAMPLES], C_GT))
    row_specs.append(("GeneChat\n(Input: DNA)", [ex["gc"] for ex in EXAMPLES], C_GC))
    for model in MODEL_ORDER:
        short = model.replace(" ", "\n", 1) if len(model) > 12 else model
        bg = BASELINE_COLORS[model]
        row_specs.append((f"{short}\n(Input: DNA)",
                          [ex["bl"][model] for ex in EXAMPLES], bg))

    # Compute per-row heights — each row fits its own text tightly
    row_heights = []
    for _, texts, _ in row_specs:
        h = max(wrapped_height(t, wrap_w, fs, fig_h) for t in texts)
        h = max(h, 0.05)  # minimum
        row_heights.append(h)

    cur_y = top_y - 0.05  # below headers

    for ri, (label, texts, bg) in enumerate(row_specs):
        rh = row_heights[ri]

        # Model label
        ax.text(label_w - 0.012, cur_y - rh / 2, label,
                transform=ax.transAxes, fontsize=fs_label,
                color=C_LBL, ha="right", va="center",
                fontweight="bold", linespacing=1.2)

        # Text boxes — no border, just colored background
        for ci, text in enumerate(texts):
            cx = col1_x if ci == 0 else col2_x
            box = FancyBboxPatch(
                (cx, cur_y - rh), col_w, rh,
                boxstyle="round,pad=0.005",
                facecolor=bg, edgecolor="none",
                linewidth=1, transform=ax.transAxes, zorder=2)
            ax.add_patch(box)

            wrapped = textwrap.fill(text, width=wrap_w)
            ax.text(cx + 0.010, cur_y - 0.006, wrapped,
                    transform=ax.transAxes, fontsize=fs,
                    color="#222222", ha="left", va="top",
                    linespacing=1.3, zorder=3)

        cur_y -= rh + row_gap

    # Update divider to end at last row
    ax.plot([div_x, div_x], [cur_y + row_gap, top_y - 0.005],
            transform=ax.transAxes, color="#AAAAAA",
            linewidth=0.8, linestyle=":", zorder=0)

    # Set ylim to crop out empty space at bottom
    ax.set_ylim(cur_y - 0.01, 1.12)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.09)
    os.makedirs(output_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = f"{output_dir}/model_comparison.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_comparison()
    print("\nDone. Comparison figure saved.")
