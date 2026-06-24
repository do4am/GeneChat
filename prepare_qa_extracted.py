"""
Generate additional Q/A pairs from NCBI and UniProt summaries using a single
generic question — the same approach as ProteinChat.

Each cleaned summary is paired with "What is the function of this gene?",
providing a third question phrasing beyond the two used for the summary splits
("Tell me about this gene." / "Please provide a detailed description of the gene.").

Output: qa_function_extracted.json in each data directory
Format: [{"Gene Id": ..., "Q": "What is the function of this gene?", "A": "..."}, ...]

Usage:
    python prepare_qa_extracted.py
"""

import json
import os

DATA_DIRS = [
    "/home/namdo/applications/data_hm/train_hm",
    "/home/namdo/applications/data_hm/valid_hm",
    "/home/namdo/applications/data_hm/test_hm",
]

QUESTION = "Describe this gene."


def process_dir(data_dir):
    print(f"\n{'='*60}")
    print(f"Processing: {data_dir}")

    out_path = os.path.join(data_dir, "qa_function_extracted.json")
    all_pairs = []

    for fname in ["qa_summary_rule_clean.json", "qa_summary_uniprot_clean.json"]:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  Skipping {fname} (not found)")
            continue

        data = json.load(open(fpath))
        print(f"  {fname}: {len(data)} entries")

        for entry in data:
            gene_id = entry.get("Gene Id")
            summary = entry.get("Summary", "").strip()
            if not summary:
                continue
            all_pairs.append({"Gene Id": gene_id, "Q": QUESTION, "A": summary})

    # Deduplicate: one entry per gene_id (keep first occurrence)
    seen = set()
    deduped = []
    for p in all_pairs:
        gid = str(p["Gene Id"])
        if gid not in seen:
            seen.add(gid)
            deduped.append(p)

    with open(out_path, "w") as f:
        json.dump(deduped, f, indent=2)

    print(f"  Q/A pairs: {len(deduped):,}  (one per gene, question: \"{QUESTION}\")")
    print(f"  Saved to {out_path}")


def main():
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"Skipping (not found): {data_dir}")
            continue
        process_dir(data_dir)


if __name__ == "__main__":
    main()

