"""
Create a tiny toy dataset from 20 iconic human genes for sanity-checking the
GeneChat training pipeline.

Goal: if the model can overfit this dataset (loss → ~0.5, generation matches
targets) within ~500 iters, the encoder→adaptor→LLM chain is working.

Output: /home/namdo/applications/data_hm/toy_hm/
  - qa_summary_uniprot_clean.json  (20 entries, UniProt preferred)
  - qa_summary_rule_clean.json     (empty — not needed for toy)
  - qa_function_extracted.json     (empty)
  - seq_mrna.json                  (20 mRNA sequences)
  - seq.json                       (copied from train_hm as fallback)
"""

import json
import os

# 20 iconic human genes — very distinctive, well-known descriptions
TOY_GENE_IDS = {
    "672",    # BRCA1
    "7157",   # TP53
    "1956",   # EGFR
    "3630",   # INS (insulin)
    "348",    # APOE
    "4609",   # MYC
    "3845",   # KRAS
    "5728",   # PTEN
    "207",    # AKT1
    "7422",   # VEGFA
    "7124",   # TNF
    "3569",   # IL6
    "1029",   # CDKN2A
    "4193",   # MDM2
    "5290",   # PIK3CA
    "2475",   # MTOR
    "2064",   # ERBB2 (HER2)
    "5925",   # RB1
    "6774",   # STAT3
    "5594",   # MAPK1 (ERK2)
}

SRC_DIR  = "/home/namdo/applications/data_hm/train_hm"
TOY_DIR  = "/home/namdo/applications/data_hm/toy_hm"

os.makedirs(TOY_DIR, exist_ok=True)


def filter_json(src_path, key="Gene Id"):
    if not os.path.exists(src_path):
        return []
    data = json.load(open(src_path))
    filtered = [e for e in data if str(e.get(key, "")) in TOY_GENE_IDS]
    print(f"  {os.path.basename(src_path)}: {len(filtered)}/{len(data)} kept")
    return filtered


# UniProt summaries (preferred)
uniprot = filter_json(os.path.join(SRC_DIR, "qa_summary_uniprot_clean.json"))
with open(os.path.join(TOY_DIR, "qa_summary_uniprot_clean.json"), "w") as f:
    json.dump(uniprot, f, indent=2)

# NCBI summaries (fallback for genes not in UniProt)
uniprot_ids = {str(e["Gene Id"]) for e in uniprot}
rule = filter_json(os.path.join(SRC_DIR, "qa_summary_rule_clean.json"))
rule_extra = [e for e in rule if str(e.get("Gene Id", "")) not in uniprot_ids]
with open(os.path.join(TOY_DIR, "qa_summary_rule_clean.json"), "w") as f:
    json.dump(rule_extra, f, indent=2)
print(f"  rule_clean (extra only): {len(rule_extra)}")

# Empty extracted QA (not needed for toy)
with open(os.path.join(TOY_DIR, "qa_function_extracted.json"), "w") as f:
    json.dump([], f)

# mRNA sequences
mrna_src = os.path.join(SRC_DIR, "seq_mrna.json")
if os.path.exists(mrna_src):
    mrna = json.load(open(mrna_src))
    mrna_toy = {k: v for k, v in mrna.items() if str(k) in TOY_GENE_IDS}
    print(f"  seq_mrna.json: {len(mrna_toy)}/{len(mrna)} kept")
    with open(os.path.join(TOY_DIR, "seq_mrna.json"), "w") as f:
        json.dump(mrna_toy, f)
else:
    print("  seq_mrna.json not found — seq.json fallback will be used")
    mrna_toy = {}

# Genomic DNA fallback sequences
seq_src = os.path.join(SRC_DIR, "seq.json")
if os.path.exists(seq_src):
    seq = json.load(open(seq_src))
    seq_toy = {k: v for k, v in seq.items() if str(k) in TOY_GENE_IDS}
    print(f"  seq.json: {len(seq_toy)}/{len(seq)} kept")
    with open(os.path.join(TOY_DIR, "seq.json"), "w") as f:
        json.dump(seq_toy, f)

print(f"\nToy dataset saved to {TOY_DIR}")
print(f"UniProt entries : {len(uniprot)}")
print(f"NCBI extras     : {len(rule_extra)}")
print(f"mRNA sequences  : {len(mrna_toy)}")
print("\nGene IDs in toy set:")
for gid in sorted(TOY_GENE_IDS, key=int):
    print(f"  {gid}")
