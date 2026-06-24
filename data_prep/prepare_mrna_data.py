"""
prepare_mrna_data.py

Downloads canonical mRNA (NM_) sequences from NCBI for all Gene IDs in data_hm.
Outputs seq_mrna.json in the same format as seq.json: {gene_id: [mrna_sequence]}

Genes with no NM_ record (regulatory elements, pseudogenes) are skipped.
At inference time, the model falls back to genomic seq.json for these genes.

Requirements:
    pip install biopython tqdm
"""

import json
import os
import time
import logging
import random
from tqdm import tqdm
from Bio import Entrez
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Config ──────────────────────────────────────────────────────────────────
Entrez.email = "namdo@ucsd.edu"
Entrez.api_key = "a6ecdb55361a71d65bf9accaba5f3a565a09"   # 10 req/sec

DATA_DIRS = [
    "/home/namdo/applications/data_hm/train_hm",
    "/home/namdo/applications/data_hm/valid_hm",
    "/home/namdo/applications/data_hm/test_hm",
]

OUT_FILENAME   = "seq_mrna.json"   # written alongside seq.json in each data dir
FAILED_LOG     = "mrna_failed.txt" # gene IDs that failed after all retries
SLEEP_SEC      = 0.11              # 10 req/sec with API key
MAX_LENGTH     = 50000             # truncate extreme outliers (e.g. titin isoforms)
MAX_RETRIES    = 5                 # retry failed API calls up to 5 times
RETRY_BASE_SEC = 2.0               # exponential backoff base (2, 4, 8, 16, 32 sec)
# ────────────────────────────────────────────────────────────────────────────


def load_gene_ids(data_dir):
    """Only load gene IDs that appear in summary/description files.
    Excludes qa_kw.json which contains all 109k genes including ENCODE elements."""
    needed = set()

    for fname in ['qa_summary_rule.json', 'qa_summary_rule_clean.json',
                  'qa_summary_uniprot.json', 'qa_text_manual.json']:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = json.load(open(fpath))
        for entry in data:
            gid = entry.get('Gene Id') or entry.get('gene_id')
            if gid:
                needed.add(str(gid))
        logging.info(f"  {fname}: {len(data):,} entries")

    logging.info(f"  Total unique gene IDs with summaries: {len(needed):,}")
    return list(needed)


def fetch_with_retry(fn):
    """Call fn() with exponential backoff retry on any error."""
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            time.sleep(SLEEP_SEC)
            return result
        except Exception as e:
            wait = RETRY_BASE_SEC ** (attempt + 1) + random.uniform(0, 1)
            logging.warning(f"API error (attempt {attempt+1}/{MAX_RETRIES}): {e}, retrying in {wait:.1f}s")
            time.sleep(wait)
    logging.error(f"Failed after {MAX_RETRIES} retries")
    return None


def fetch_nm_accessions(gene_id):
    """Search nuccore for mRNA records (NM_/XM_) associated with a Gene ID."""
    def _call():
        # Search nuccore directly: gene ID + RefSeq mRNA filter
        handle = Entrez.esearch(
            db="nuccore",
            term=f"{gene_id}[GeneID] AND refseq[filter] AND mRNA[Filter]",
            retmax=3   # esearch already filters to mRNA, 3 is enough to find canonical
        )
        result = Entrez.read(handle)
        handle.close()
        return result["IdList"]

    result = fetch_with_retry(_call)
    return result if result is not None else []


def fetch_sequence(nuccore_id):
    """Fetch FASTA sequence for a nuccore UID. Returns (accession, sequence) or None."""
    def _call():
        handle = Entrez.efetch(
            db="nuccore", id=nuccore_id,
            rettype="fasta", retmode="text"
        )
        fasta = handle.read()
        handle.close()
        lines = fasta.strip().split("\n")
        if not lines or not lines[0].startswith(">"):
            return None
        accession = lines[0].split()[0][1:]   # e.g. NM_000546.6
        seq = "".join(lines[1:]).upper()
        return accession, seq

    return fetch_with_retry(_call)


def pick_canonical(accessions_with_seqs):
    """
    Pick the best transcript in priority order:
    1. NM_ (manually curated mRNA) — longest
    2. XM_ (predicted mRNA) — longest, fallback when no NM_ exists
    Excludes NR_ (non-coding RNA) and XR_ (predicted non-coding).
    """
    nm_records = [(acc, seq) for acc, seq in accessions_with_seqs if acc.startswith("NM_")]
    if nm_records:
        nm_records.sort(key=lambda x: len(x[1]), reverse=True)
        return nm_records[0]

    xm_records = [(acc, seq) for acc, seq in accessions_with_seqs if acc.startswith("XM_")]
    if xm_records:
        xm_records.sort(key=lambda x: len(x[1]), reverse=True)
        return xm_records[0]

    return None


def process_gene(gene_id):
    """
    Returns:
        (gene_id, mrna_sequence)  — success
        None                      — no NM_ record found (normal, e.g. non-coding gene)
        "error"                   — API call failed after all retries
    """
    accession_ids = fetch_nm_accessions(gene_id)
    if accession_ids is None:
        return "error"
    if not accession_ids:
        return None   # no mRNA transcript exists for this gene

    records = []
    for uid in accession_ids[:3]:   # esearch already filtered to mRNA, 3 is enough
        result = fetch_sequence(uid)
        if result:
            acc, seq = result
            if acc.startswith("NM_") or acc.startswith("XM_"):
                records.append((acc, seq))

    canonical = pick_canonical(records)
    if canonical is None:
        return None   # gene has only NR_/XM_ records (non-coding or predicted)

    acc, seq = canonical
    if len(seq) > MAX_LENGTH:
        seq = seq[:MAX_LENGTH]

    return gene_id, seq


def build_mrna_dict(data_dir):
    gene_ids = load_gene_ids(data_dir)
    logging.info(f"{data_dir}: {len(gene_ids)} gene IDs to process")

    out_path = os.path.join(data_dir, OUT_FILENAME)

    # Resume: load existing results if interrupted
    if os.path.exists(out_path):
        mrna_dict = json.load(open(out_path))
        logging.info(f"  Resuming: {len(mrna_dict)} already fetched")
    else:
        mrna_dict = {}

    remaining = [g for g in gene_ids if str(g) not in mrna_dict]
    logging.info(f"  Remaining: {len(remaining)} genes to fetch")

    found   = len(mrna_dict)
    skipped = 0
    failed  = []   # genes that errored after all retries
    no_nm   = []   # genes with no NM_ record

    for gene_id in tqdm(remaining, desc=os.path.basename(data_dir)):
        result = process_gene(str(gene_id))
        if result == "error":
            failed.append(gene_id)
            logging.warning(f"  API error for gene {gene_id}")
        elif result:
            gid, seq = result
            mrna_dict[gid] = [seq]   # same format as seq.json: {id: [sequence]}
            found += 1
        else:
            skipped += 1
            no_nm.append(gene_id)
            logging.info(f"  No mRNA found for gene {gene_id}")

        # Save checkpoint every 100 genes
        if (found + skipped + len(failed)) % 100 == 0:
            with open(out_path, "w") as f:
                json.dump(mrna_dict, f)

    # Final save
    with open(out_path, "w") as f:
        json.dump(mrna_dict, f)

    # Save genes with no NM_ record
    if no_nm:
        no_nm_path = os.path.join(data_dir, "mrna_no_nm.txt")
        with open(no_nm_path, "w") as f:
            f.write("\n".join(str(g) for g in no_nm))
        logging.info(f"  {len(no_nm)} genes with no NM_ record → {no_nm_path}")

    # Save API error genes for manual retry
    if failed:
        fail_path = os.path.join(data_dir, FAILED_LOG)
        with open(fail_path, "w") as f:
            f.write("\n".join(str(g) for g in failed))
        logging.warning(f"  {len(failed)} genes failed after retries → {fail_path}")

    logging.info(f"  Done: {found} mRNA fetched, {skipped} no NM_, {len(failed)} errors")
    logging.info(f"  Saved to {out_path}")
    return mrna_dict


if __name__ == "__main__":
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            logging.warning(f"Directory not found: {data_dir}, skipping")
            continue
        build_mrna_dict(data_dir)
