"""
Build human + mouse filtered training, validation, and test data for GeneChat.

Steps:
  1. Read gene_info → collect all human (9606) + mouse (10090) GeneIDs with metadata
  2. Cross-reference with existing seq.json to keep only genes we have DNA for
  3. Filter qa_summary_rule.json (quality rules) and qa_kw.json to human+mouse only
  4. Write new storage directories:
       /home/namdo/applications/data_hm/train_hm/
       /home/namdo/applications/data_hm/valid_hm/
       /home/namdo/applications/data_hm/test_hm/
       /home/namdo/applications/data_hm/test_ood/  (all other organisms, zero-shot eval)

Usage:
    python prepare_human_mouse_data.py

Requirements: gene_info must be in the project root (already downloaded).
"""

import csv
import json
import os
import shutil
from collections import Counter, defaultdict

# ── Paths ───────────────────────────────────────────────────────────────────
GENE_INFO    = "gene_info"           # downloaded in project root (run on remote)

# Source: original data on remote machine
SRC_TRAIN    = "/data2/genechat/GeneChat_data/train_set"
SRC_VALID    = "/data2/genechat/GeneChat_data/valid_set"
SRC_TEST     = "/data2/genechat/GeneChat_data/test_set"

# Destination: new human+mouse filtered data
DST_TRAIN    = "/home/namdo/applications/data_hm/train_hm"
DST_VALID    = "/home/namdo/applications/data_hm/valid_hm"
DST_TEST     = "/home/namdo/applications/data_hm/test_hm"
DST_OOD      = "/home/namdo/applications/data_hm/test_ood"

# ── Filtering thresholds (same as prepare_clean_train.py) ────────────────────
MAX_TEMPLATE_COUNT = 3
MIN_WORD_COUNT     = 30
HUMAN_MOUSE_TAXIDS = {"9606", "10090"}   # Homo sapiens, Mus musculus

# ── Step 1: Read gene_info ──────────────────────────────────────────────────
print("Reading gene_info ...")
hm_genes = {}   # gene_id (str) → {symbol, description, chromosome, type, organism}

with open(GENE_INFO) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['#tax_id'] not in HUMAN_MOUSE_TAXIDS:
            continue
        gid = row['GeneID']
        # Use official full name if available, otherwise description
        full_name = row.get('Full_name_from_nomenclature_authority', '').strip()
        description = row.get('description', '').strip()
        summary_text = full_name if full_name and full_name != '-' else description

        hm_genes[gid] = {
            'symbol':      row.get('Symbol', '-').strip(),
            'description': summary_text,
            'chromosome':  row.get('chromosome', '-').strip(),
            'type':        row.get('type_of_gene', '-').strip(),
            'organism':    'Homo sapiens' if row['#tax_id'] == '9606' else 'Mus musculus',
        }

print(f"  Human + mouse genes in gene_info: {len(hm_genes):,}")

# ── Step 2: Filter function ──────────────────────────────────────────────────
def is_quality_summary(summary, counts):
    """Return True if the summary passes quality filters."""
    s = summary.strip()
    if not s:
        return False
    if s.startswith("Predicted to"):
        return False
    if len(s.split()) < MIN_WORD_COUNT:
        return False
    if counts[s] > MAX_TEMPLATE_COUNT:
        return False
    return True


def filter_split(src_dir, dst_dir, hm_gene_ids, label):
    """Filter one data split (train/valid/test) to human+mouse genes."""
    os.makedirs(dst_dir, exist_ok=True)
    print(f"\nProcessing {label} ({src_dir} → {dst_dir})")

    # ── seq.json ────────────────────────────────────────────────────────────
    seq_path = os.path.join(src_dir, "seq.json")
    seq_all  = json.load(open(seq_path)) if os.path.exists(seq_path) else {}
    seq_hm   = {gid: seq_all[gid] for gid in seq_all if gid in hm_gene_ids}
    with open(os.path.join(dst_dir, "seq.json"), "w") as f:
        json.dump(seq_hm, f)
    print(f"  seq.json:            {len(seq_all):>6,} → {len(seq_hm):>6,} genes")

    # ── gene_ids.json ────────────────────────────────────────────────────────
    gid_path = os.path.join(src_dir, "gene_ids.json")
    gids_all  = json.load(open(gid_path)) if os.path.exists(gid_path) else []
    gids_hm   = [g for g in gids_all if g in hm_gene_ids]
    with open(os.path.join(dst_dir, "gene_ids.json"), "w") as f:
        json.dump(gids_hm, f)
    print(f"  gene_ids.json:       {len(gids_all):>6,} → {len(gids_hm):>6,} genes")

    # ── qa_summary_rule.json ─────────────────────────────────────────────────
    summ_path = os.path.join(src_dir, "qa_summary_rule.json")
    summ_all  = json.load(open(summ_path)) if os.path.exists(summ_path) else []

    # First pass: count summary occurrences (for boilerplate detection)
    counts = Counter(e.get("Summary", "").strip() for e in summ_all)

    summ_hm = [
        e for e in summ_all
        if e["Gene Id"] in hm_gene_ids
        and is_quality_summary(e.get("Summary", ""), counts)
    ]
    with open(os.path.join(dst_dir, "qa_summary_rule.json"), "w") as f:
        json.dump(summ_hm, f, indent=2)
    print(f"  qa_summary_rule:     {len(summ_all):>6,} → {len(summ_hm):>6,} entries (quality filtered)")

    # ── qa_kw.json ───────────────────────────────────────────────────────────
    kw_path = os.path.join(src_dir, "qa_kw.json")
    kw_all  = json.load(open(kw_path)) if os.path.exists(kw_path) else []
    kw_hm   = [e for e in kw_all if e["Gene Id"] in hm_gene_ids]
    with open(os.path.join(dst_dir, "qa_kw.json"), "w") as f:
        json.dump(kw_hm, f, indent=2)
    print(f"  qa_kw.json:          {len(kw_all):>6,} → {len(kw_hm):>6,} entries")

    # ── qa_text_manual.json (copy if exists) ─────────────────────────────────
    manual_path = os.path.join(src_dir, "qa_text_manual.json")
    if os.path.exists(manual_path):
        manual_all = json.load(open(manual_path))
        manual_hm  = [e for e in manual_all if e.get("Gene Id") in hm_gene_ids]
        with open(os.path.join(dst_dir, "qa_text_manual.json"), "w") as f:
            json.dump(manual_hm, f, indent=2)
        print(f"  qa_text_manual:      {len(manual_all):>6,} → {len(manual_hm):>6,} entries")

    return set(g for g in seq_hm)


def filter_ood(src_dir, dst_dir, hm_gene_ids, label):
    """Keep out-of-distribution genes (not human/mouse) for zero-shot eval."""
    os.makedirs(dst_dir, exist_ok=True)
    print(f"\nProcessing {label} OOD ({src_dir} → {dst_dir})")

    seq_all = json.load(open(os.path.join(src_dir, "seq.json")))
    seq_ood = {gid: seq_all[gid] for gid in seq_all if gid not in hm_gene_ids}
    with open(os.path.join(dst_dir, "seq.json"), "w") as f:
        json.dump(seq_ood, f)

    summ_all = json.load(open(os.path.join(src_dir, "qa_summary_rule.json")))
    counts   = Counter(e.get("Summary", "").strip() for e in summ_all)
    summ_ood = [e for e in summ_all if e["Gene Id"] not in hm_gene_ids
                and is_quality_summary(e.get("Summary", ""), counts)]
    with open(os.path.join(dst_dir, "qa_summary_rule.json"), "w") as f:
        json.dump(summ_ood, f, indent=2)

    kw_all = json.load(open(os.path.join(src_dir, "qa_kw.json")))
    kw_ood = [e for e in kw_all if e["Gene Id"] not in hm_gene_ids]
    with open(os.path.join(dst_dir, "qa_kw.json"), "w") as f:
        json.dump(kw_ood, f, indent=2)

    gids_all = json.load(open(os.path.join(src_dir, "gene_ids.json")))
    gids_ood = [g for g in gids_all if g not in hm_gene_ids]
    with open(os.path.join(dst_dir, "gene_ids.json"), "w") as f:
        json.dump(gids_ood, f)

    print(f"  OOD seq: {len(seq_ood):,} genes | OOD summaries: {len(summ_ood):,}")


# ── Step 3: Find human+mouse gene IDs that exist in our seq data ─────────────
print("\nFinding human+mouse genes with DNA sequences ...")

# Union of gene IDs across all splits
all_seq_ids = set()
for src in [SRC_TRAIN, SRC_VALID, SRC_TEST]:
    sp = os.path.join(src, "seq.json")
    if os.path.exists(sp):
        all_seq_ids |= set(json.load(open(sp)).keys())

hm_with_seq = set(hm_genes.keys()) & all_seq_ids
print(f"  Total genes in our seq files:    {len(all_seq_ids):,}")
print(f"  Human+mouse genes in gene_info:  {len(hm_genes):,}")
print(f"  Overlap (have seq + hm label):   {len(hm_with_seq):,}")

# Organism breakdown
org_count = Counter(hm_genes[g]['organism'] for g in hm_with_seq)
for org, n in org_count.most_common():
    print(f"    {org}: {n:,}")

# ── Step 4: Build filtered splits ────────────────────────────────────────────
filter_split(SRC_TRAIN, DST_TRAIN, hm_with_seq, "TRAIN")
filter_split(SRC_VALID, DST_VALID, hm_with_seq, "VALID")
filter_split(SRC_TEST,  DST_TEST,  hm_with_seq, "TEST")
filter_ood(SRC_TEST,   DST_OOD,   hm_with_seq, "TEST")

# ── Step 5: Summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DONE. Update configs to use new storage paths:")
print(f"  Train:  {DST_TRAIN}")
print(f"  Valid:  {DST_VALID}")
print(f"  Test:   {DST_TEST}")
print(f"  OOD:    {DST_OOD}  (zero-shot evaluation only)")
print()
print("Next: update configs/genechat_stage3.yaml:")
print(f"  storage: /home/namdo/applications/GeneChat/{DST_TRAIN}")
