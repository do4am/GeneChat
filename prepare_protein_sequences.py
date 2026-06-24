"""
prepare_protein_sequences.py

Translate mRNA sequences → protein sequences for all data splits.

For each mRNA in seq_mrna.json:
  1. Find the longest ORF (ATG ... stop codon) using existing find_longest_orf()
  2. Translate ORF to amino acid sequence using the standard genetic code
  3. Skip genes with no ORF >= 300 nt (likely non-coding / regulatory)

Output: seq_protein.json in each split directory, same format as seq_mrna.json:
  {"gene_id": ["MKTIIALSYIFCLVFA..."]}

Usage:
  conda run -n genechat python prepare_protein_sequences.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from genechat.models.genechat import find_longest_orf

GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def translate(orf: str) -> str:
    """Translate a DNA ORF (starting with ATG) to amino acid sequence.
    Stops at first stop codon and excludes it from output."""
    aa = []
    orf = orf.upper()
    for i in range(0, len(orf) - 2, 3):
        codon = orf[i:i+3]
        aa_char = GENETIC_CODE.get(codon, 'X')
        if aa_char == '*':
            break
        aa.append(aa_char)
    return ''.join(aa)

def process_split(split_dir: str):
    seq_path = os.path.join(split_dir, 'seq_mrna.json')
    if not os.path.exists(seq_path):
        # fall back to seq.json (genomic DNA)
        seq_path = os.path.join(split_dir, 'seq.json')
    if not os.path.exists(seq_path):
        print(f"  No seq file found in {split_dir}, skipping.")
        return

    with open(seq_path) as f:
        seqs = json.load(f)

    out = {}
    skipped = 0
    for gene_id, seq_list in seqs.items():
        mrna = seq_list[0]
        orf = find_longest_orf(mrna, min_nt=300)
        if orf == mrna and not mrna.upper().startswith('ATG'):
            # find_longest_orf returned original → no ORF found
            skipped += 1
            continue
        protein = translate(orf)
        if len(protein) < 50:
            skipped += 1
            continue
        out[gene_id] = [protein]

    out_path = os.path.join(split_dir, 'seq_protein.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)

    print(f"  {split_dir}: {len(out)} proteins saved, {skipped} skipped (no ORF)")

SPLITS = [
    '/home/namdo/applications/data_hm/train_hm',
    '/home/namdo/applications/data_hm/valid_hm',
    '/home/namdo/applications/data_hm/test_train',
]

if __name__ == '__main__':
    for split in SPLITS:
        print(f"Processing {split} ...")
        process_split(split)
    print("Done.")
