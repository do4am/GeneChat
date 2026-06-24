"""
Extract a fixed test set from the training set for overfitting evaluation.

Samples N_SAMPLE gene IDs that appear in both seq_mrna.json and
qa_summary_uniprot_clean.json (highest quality descriptions), writes
a self-contained test directory at DST_DIR.

Usage:
    python prepare_train_test.py
"""

import json
import os
import random

# ── Config ───────────────────────────────────────────────────────────────────
SRC_DIR    = "/home/namdo/applications/data_hm/train_hm"
DST_DIR    = "/home/namdo/applications/data_hm/test_train"
N_SAMPLE   = 300      # number of genes to sample
SEED       = 42

# ── Load sources ──────────────────────────────────────────────────────────────
print("Loading training data...")
seq_mrna   = json.load(open(os.path.join(SRC_DIR, "seq_mrna.json")))
seq_dna    = json.load(open(os.path.join(SRC_DIR, "seq.json")))
uniprot    = json.load(open(os.path.join(SRC_DIR, "qa_summary_uniprot_clean.json")))
ncbi       = json.load(open(os.path.join(SRC_DIR, "qa_summary_rule_clean.json")))
extracted  = json.load(open(os.path.join(SRC_DIR, "qa_function_extracted.json"))) \
             if os.path.exists(os.path.join(SRC_DIR, "qa_function_extracted.json")) else []

# Build gene_id sets
uniprot_ids  = {str(e["Gene Id"]) for e in uniprot}
ncbi_ids     = {str(e["Gene Id"]) for e in ncbi}
mrna_ids     = set(seq_mrna.keys())

# Prefer genes that have mRNA + UniProt (highest quality)
candidates = sorted(uniprot_ids & mrna_ids)
print(f"  Candidates (mRNA + UniProt): {len(candidates):,}")

# Fill remaining slots from NCBI-only if needed
if len(candidates) < N_SAMPLE:
    extra = sorted((ncbi_ids & mrna_ids) - uniprot_ids)
    candidates += extra
    print(f"  After adding NCBI-only: {len(candidates):,}")

random.seed(SEED)
sampled = set(random.sample(candidates, min(N_SAMPLE, len(candidates))))
print(f"  Sampled: {len(sampled):,} genes (seed={SEED})")

# ── Filter each file to sampled genes ─────────────────────────────────────────
uniprot_sub   = [e for e in uniprot   if str(e["Gene Id"]) in sampled]
ncbi_sub      = [e for e in ncbi      if str(e["Gene Id"]) in sampled]
extracted_sub = [e for e in extracted if str(e.get("Gene Id", e.get("gene_id", ""))) in sampled]
seq_mrna_sub  = {k: v for k, v in seq_mrna.items() if k in sampled}
seq_dna_sub   = {k: v for k, v in seq_dna.items()  if k in sampled}

print(f"\n  UniProt entries: {len(uniprot_sub)}")
print(f"  NCBI entries:    {len(ncbi_sub)}")
print(f"  Extracted QA:    {len(extracted_sub)}")
print(f"  mRNA seqs:       {len(seq_mrna_sub)}")
print(f"  DNA seqs:        {len(seq_dna_sub)}")

# ── Write output ──────────────────────────────────────────────────────────────
os.makedirs(DST_DIR, exist_ok=True)

json.dump(uniprot_sub,   open(os.path.join(DST_DIR, "qa_summary_uniprot_clean.json"), "w"), indent=2)
json.dump(ncbi_sub,      open(os.path.join(DST_DIR, "qa_summary_rule_clean.json"),    "w"), indent=2)
json.dump(extracted_sub, open(os.path.join(DST_DIR, "qa_function_extracted.json"),    "w"), indent=2)
json.dump(seq_mrna_sub,  open(os.path.join(DST_DIR, "seq_mrna.json"),                 "w"), indent=2)
json.dump(seq_dna_sub,   open(os.path.join(DST_DIR, "seq.json"),                      "w"), indent=2)

# Write sampled gene IDs for reference
json.dump(sorted(sampled), open(os.path.join(DST_DIR, "sampled_gene_ids.json"), "w"), indent=2)

print(f"\nDone. Test set written to {DST_DIR}")
print(f"Point genechat_eval_mrna.yaml → storage: {DST_DIR}")
