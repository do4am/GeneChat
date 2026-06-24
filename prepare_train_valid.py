"""
Extract a validation set from the training set for monitoring training progress.

Samples ~300 genes from train_hm (stratified across NCBI/UniProt/extracted QA)
and writes them to valid_hm_train. Does NOT remove them from the train set —
this is an in-training validation split for loss monitoring, not held-out eval.

Output: /home/namdo/applications/data_hm/valid_hm_train/
  qa_summary_rule_clean.json
  qa_summary_uniprot_clean.json
  qa_function_extracted.json
  seq_mrna.json
  seq.json  (symlink to train)

Usage:
    python prepare_train_valid.py
"""

import json
import os
import random

TRAIN_DIR  = "/home/namdo/applications/data_hm/train_hm"
VALID_DIR  = "/home/namdo/applications/data_hm/valid_hm_train"
SEED       = 42
N_NCBI     = 60    # sample from NCBI fallback
N_UNIPROT  = 150   # sample from UniProt (larger pool)
N_EXTRACTED= 90    # sample from extracted QA

random.seed(SEED)
os.makedirs(VALID_DIR, exist_ok=True)

# ── NCBI ──────────────────────────────────────────────────────────────────────
ncbi_all = json.load(open(os.path.join(TRAIN_DIR, "qa_summary_rule_clean.json")))
ncbi_sample = random.sample(ncbi_all, min(N_NCBI, len(ncbi_all)))
with open(os.path.join(VALID_DIR, "qa_summary_rule_clean.json"), "w") as f:
    json.dump(ncbi_sample, f, indent=2)
print(f"NCBI:      {len(ncbi_sample)} / {len(ncbi_all)}")

# ── UniProt ───────────────────────────────────────────────────────────────────
uni_all = json.load(open(os.path.join(TRAIN_DIR, "qa_summary_uniprot_clean.json")))
uni_sample = random.sample(uni_all, min(N_UNIPROT, len(uni_all)))
with open(os.path.join(VALID_DIR, "qa_summary_uniprot_clean.json"), "w") as f:
    json.dump(uni_sample, f, indent=2)
print(f"UniProt:   {len(uni_sample)} / {len(uni_all)}")

# ── Extracted QA ──────────────────────────────────────────────────────────────
ext_all = json.load(open(os.path.join(TRAIN_DIR, "qa_function_extracted.json")))
ext_sample = random.sample(ext_all, min(N_EXTRACTED, len(ext_all)))
with open(os.path.join(VALID_DIR, "qa_function_extracted.json"), "w") as f:
    json.dump(ext_sample, f, indent=2)
print(f"Extracted: {len(ext_sample)} / {len(ext_all)}")

# ── Sequences — symlink to train (read-only, no duplication) ──────────────────
for fname in ("seq_mrna.json", "seq.json"):
    src = os.path.join(TRAIN_DIR, fname)
    dst = os.path.join(VALID_DIR, fname)
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print(f"Symlinked: {dst} → {src}")

total = len(ncbi_sample) + len(uni_sample) + len(ext_sample)
print(f"\nTotal valid_hm_train pairs: {total} "
      f"(+{len(uni_sample)*2} UniProt 3x → {total + len(uni_sample)*2} effective)")
print(f"Saved to: {VALID_DIR}")
