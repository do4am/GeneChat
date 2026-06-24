"""
Download UniProt Swiss-Prot descriptions for human and mouse genes,
map to NCBI Gene IDs, and produce qa_summary_uniprot.json in the
same format as qa_summary_rule.json.

Usage:
    python prepare_uniprot_data.py

Output:
    /home/namdo/applications/data_hm/train_hm/qa_summary_uniprot.json
    /home/namdo/applications/data_hm/valid_hm/qa_summary_uniprot.json
    /home/namdo/applications/data_hm/test_hm/qa_summary_uniprot.json

Requirements:
    pip install requests tqdm
"""

import json
import os
import requests
import time
from collections import defaultdict
from tqdm import tqdm

# UniProt REST API
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

# Data directories to update
DATA_SPLITS = {
    "train": "/home/namdo/applications/data_hm/train_hm",
    "valid": "/home/namdo/applications/data_hm/valid_hm",
    "test":  "/home/namdo/applications/data_hm/test_hm",
}


def fetch_uniprot(organism_id, batch_size=500):
    """
    Fetch all reviewed UniProt entries for an organism.
    Returns list of dicts with Gene ID and function description.
    """
    entries = []
    cursor = None
    fields = "accession,gene_names,xref_geneid,cc_function,protein_name"
    query = f"organism_id:{organism_id} AND reviewed:true"

    print(f"  Fetching UniProt reviewed entries for organism {organism_id}...")

    while True:
        params = {
            "query": query,
            "fields": fields,
            "format": "json",
            "size": batch_size,
        }
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(UNIPROT_API, params=params, timeout=60)
        if resp.status_code != 200:
            print(f"  Warning: API returned {resp.status_code}, retrying...")
            time.sleep(5)
            continue

        data = resp.json()
        results = data.get("results", [])
        entries.extend(results)

        # Check for next page
        link = resp.headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        # Extract cursor from Link header
        import re
        match = re.search(r'cursor=([^&>]+)', link)
        if not match:
            break
        cursor = match.group(1)
        time.sleep(0.5)  # be polite to UniProt API

    print(f"  Fetched {len(entries)} entries")
    return entries


def extract_function(entry):
    """Extract function description from UniProt entry."""
    comments = entry.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value", "").strip()
    return ""


def extract_gene_ids(entry):
    """Extract NCBI Gene IDs from UniProt cross-references."""
    gene_ids = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "GeneID":
            gene_ids.append(xref.get("id", ""))
    return gene_ids


def build_geneid_to_description(entries):
    """Build mapping from NCBI Gene ID to UniProt function description."""
    mapping = {}
    no_function = 0
    no_geneid = 0

    for entry in entries:
        func = extract_function(entry)
        if not func:
            no_function += 1
            continue

        gene_ids = extract_gene_ids(entry)
        if not gene_ids:
            no_geneid += 1
            continue

        for gid in gene_ids:
            # Keep longest description if multiple UniProt entries map to same Gene ID
            if gid not in mapping or len(func) > len(mapping[gid]):
                mapping[gid] = func

    print(f"  Entries without function: {no_function}")
    print(f"  Entries without Gene ID:  {no_geneid}")
    print(f"  Unique Gene IDs mapped:   {len(mapping)}")
    return mapping


def update_split(split_dir, geneid_to_desc, split_name):
    """
    For a data split directory, load existing gene IDs and create
    qa_summary_uniprot.json with UniProt descriptions.
    """
    kw_path = os.path.join(split_dir, "qa_kw.json")
    if not os.path.exists(kw_path):
        print(f"  Skipping {split_name}: no qa_kw.json found")
        return 0

    # Get all unique Gene IDs in this split
    kw_data = json.load(open(kw_path))
    gene_ids = list({str(e["Gene Id"]) for e in kw_data})
    print(f"  {split_name}: {len(gene_ids)} unique genes")

    # Build output entries
    results = []
    matched = 0
    for gid in gene_ids:
        desc = geneid_to_desc.get(gid, "")
        if desc:
            results.append({"Gene Id": gid, "Summary": desc})
            matched += 1

    out_path = os.path.join(split_dir, "qa_summary_uniprot.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Matched {matched}/{len(gene_ids)} genes with UniProt descriptions")
    print(f"  Saved to {out_path}")
    return matched


def main():
    print("=" * 60)
    print("UniProt Swiss-Prot Description Downloader")
    print("=" * 60)

    # Fetch human (9606) and mouse (10090) reviewed entries
    all_entries = []
    for organism_id, name in [(9606, "Human"), (10090, "Mouse")]:
        print(f"\n[{name}] organism_id={organism_id}")
        entries = fetch_uniprot(organism_id)
        all_entries.extend(entries)
        print(f"  Subtotal: {len(entries)} entries")

    print(f"\nTotal entries fetched: {len(all_entries)}")

    # Build Gene ID → description mapping
    print("\nBuilding Gene ID → description mapping...")
    geneid_to_desc = build_geneid_to_description(all_entries)

    # Save full mapping for reference
    mapping_path = "/home/namdo/applications/GeneChat/uniprot_geneid_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(geneid_to_desc, f, indent=2)
    print(f"Full mapping saved to {mapping_path}")

    # Update each data split
    print("\nUpdating data splits...")
    total_matched = 0
    for split_name, split_dir in DATA_SPLITS.items():
        if not os.path.exists(split_dir):
            print(f"  Skipping {split_name}: directory not found")
            continue
        print(f"\n[{split_name}] {split_dir}")
        matched = update_split(split_dir, geneid_to_desc, split_name)
        total_matched += matched

    print(f"\nDone. Total matched across all splits: {total_matched}")
    print("\nNext step: update seq_text_pair_builder.py to load qa_summary_uniprot.json")
    print("Priority: use UniProt description if available, fall back to NCBI clean description")


if __name__ == "__main__":
    main()
