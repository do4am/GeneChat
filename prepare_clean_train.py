"""
Clean training data for GeneChat Stage 3 retraining.

Reads from the current training storage directory and creates a new clean directory
with a filtered qa_summary_rule.json, copying all other data files as-is.

Filtering rules applied to qa_summary_rule.json:
  1. Drop empty / whitespace summaries
  2. Drop summaries starting with "Predicted to" (RefSeq auto-annotation)
  3. Drop short summaries (< 30 words)
  4. Drop boilerplate templates (same summary text appears > 3 times)
  5. Regulatory/ENCODE elements → saved separately as qa_summary_regulatory.json
     with a clean unified description (not dropped — model learns to recognize them)

Usage:
    python prepare_clean_train.py

Output directory: exon_count/data/train_clean/
"""

import json
import os
import re
from collections import Counter, defaultdict


def strip_citations(text):
    """Remove PubMed/PMID citation markers from any source (NCBI or UniProt).

    Handles:
      (PubMed:12345678)
      (PubMed:12345678, 87654321)           ← bare second ID
      (PubMed:12345678, PubMed:87654321)    ← repeated prefix (UniProt multi-cite)
      PMID: 12345678                         ← NCBI inline format
      (PMID:12345678)
      (By similarity)
    """
    # Strip ###SOURCE:..., ###PUBMED:..., and any ### metadata sections
    # These appear in raw UniProt exports and cause the model to generate them
    if '###' in text:
        text = text[:text.index('###')].strip()
    # UniProt: (PubMed:...) — handles any mix of bare IDs and repeated PubMed: prefixes
    text = re.sub(r'\((?:PubMed:\d+)(?:,\s*(?:PubMed:)?\d+)*\)', '', text)
    # Bare PubMed:12345678 without surrounding parentheses (UniProt "According to PubMed:..." style)
    text = re.sub(r'PubMed:\d+', '', text)
    # Semicolon-separated PUBMED:ID;ID;ID format (raw UniProt export)
    text = re.sub(r'PUBMED:\d+(?:;\d+)*', '', text, flags=re.IGNORECASE)
    # NCBI inline: PMID: 12345678 (with or without parentheses)
    text = re.sub(r'\(?PMID:?\s*\d+(?:[,;]\s*\d+)*\)?', '', text)
    # Strip XML/HTML tags (e.g. <gene>...</gene>, <disease>...</disease> from UniProt XML exports)
    text = re.sub(r'<[^>]+>', '', text)
    # UniProt qualifiers
    text = re.sub(r'\(By similarity\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Microbial infection\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Probable\)\.?', '', text, flags=re.IGNORECASE)
    # NCBI attribution stamps: [provided by RefSeq, Jul 2008], [supplied by OMIM, Apr 2004]
    text = re.sub(r'\[(?:provided|supplied) by [^\]]+\]', '', text, flags=re.IGNORECASE)
    # Database timestamp artifacts: 01-01-2014, 00:00:00
    text = re.sub(r'\b\d{2}-\d{2}-\d{4},?\s*\d{2}:\d{2}:\d{2}\b', '', text)
    # GO annotation boilerplate (computationally inferred from GO terms, not experimental)
    text = re.sub(r'Predicted to be [^.]+\.', '', text)
    text = re.sub(r'Involved in several processes, including [^.]+\.', '', text)
    text = re.sub(r'Is active in [^.]+\.', '', text)
    text = re.sub(r'Is integral component of [^.]+\.', '', text)
    text = re.sub(r'Is located in [^.]+\.', '', text)
    text = re.sub(r'Likely to be non-functional[^.]*\.', '', text)
    # NCBI read-through / splicing boilerplate suffix — truncate at first occurrence
    # e.g. "Read-through transcription also exists between this gene and ..."
    #      "Naturally occurring read-through transcription ..."
    for marker in [
        "Read-through transcription",
        "Naturally occurring read-through",
        "read-through transcript",
        "Alternatively spliced transcript variants have been described",
        "Alternative splicing results in multiple transcript variants",
        "Multiple transcript variants encoding different isoforms have been found",
        "Additional alternatively spliced transcript variants have been described",
        "alternatively spliced transcript variants encoding different isoforms have been found",
        "but their biological validity has not been determined",
        "Two transcript variants encoding different isoforms have been found",
        "Three transcript variants encoding different isoforms have been found",
        "Several transcript variants encoding different isoforms have been found",
        "This gene is found in a cluster",
        "Orthologous to several human genes",
        "Orthologous to the human gene",
        # Unresolved biology suffixes — truncate the trailing hedge
        "However, it has not yet been determined",
        "however, it has not yet been determined",
        "however it remains unclear",
        "However, it remains unclear",
        "their full length nature",
        # Disease-association trailing sentences (not functional descriptions)
        "Mutations at this locus have been associated",
        "Mutations within this region",
        "Mutations within this locus",
        # Splice-variant boilerplate continuation
        "however not all variant forms",
        "However, not all variant forms",
    ]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip().rstrip('.')
    # Collapse extra whitespace left by removals
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

# Known collapse-prone gene families: if a description contains any of these keywords,
# only MAX_TEMPLATE_COUNT representatives are kept — regardless of prefix similarity.
# Maps keyword (lowercase) → family bucket ID.
# Families capped to MAX_TEMPLATE_COUNT representatives
FAMILY_BOILERPLATE = {
    "zinc finger protein":                  "znf",
    "krab domain":                          "znf",
    "keratin":                              "keratin",
    "member of the keratin":                "keratin",
}

# Families completely excluded from training (0 representatives kept).
# These cause mode collapse even with 1 representative because the model
# memorizes the single kept example as the default answer.
HARD_EXCLUDE_KEYWORDS = [
    # S100 calcium-binding protein family
    "s100 family of proteins",
    "member of the s100 family",
    "ef-hand calcium-binding motifs",
    # Serpin family
    "serine protease inhibitor (serpin)",
    "serpin family",
    # CDK / cell cycle kinase boilerplate
    "cerevisiae cell cycle control gene cdc28",
    "schizosaccharomyces pombe cdc2",
    # GPCR / olfactory — 7-TM receptor families
    "g protein-coupled receptor",
    "seven transmembrane domain",
    "olfactory receptor",
    # Interferon / immune signaling families
    "interferon-inducible double-stranded rna-activated protein kinase",
    "interferon regulatory factor",
    "interleukin 6 (il-6) cytokine",
    "il-6 cytokine family",
    "il-6/lif/cntf/osm/il-12 cytokine family",
    # Immunoglobulin superfamily
    "immunoglobulin (ig) superfamily",
    "member of the immunoglobulin superfamily",
    # Generic immune boilerplate
    "plays a role in the immune response",
    "involved in the immune response",
    # Rho family GTPases (Rac1, RhoA, Cdc42 etc.) — near-identical "member of the Rho family" entries
    "member of the rho family",
    "rho family of small gtpases",
    # Rho GTPase-activating proteins — 69 near-identical entries
    "gtpase-activating protein",
    "rhogap domain",
    # SWI/SNF chromatin remodeling complex — 37 near-identical entries
    "swi/snf",
    "swi2/snf2",
    # TP53BP1 / tumor protein p53-binding family ortholog boilerplate
    "orthologous to human tp53bp1",
    # Cytochrome P450 superfamily — 51 entries, 24 share identical prefix
    "cytochrome p450 superfamily of enzymes",
    "cytochrome p450 proteins are monooxygenases",
    # SLC25 mitochondrial carrier family — dominant collapse in second epoch
    "slc25 family",
    "member of the slc25",
    # RNA Polymerase II subunit family — repetition loop iters 140-410
    "component of the rna polymerase ii",
    "subunit of the rna polymerase ii",
    # Actin cytoskeleton generic phrase — alternating-pair repetition loop iters 570-750
    "regulation of the actin cytoskeleton and cell migration",
    # Nucleotide/thymidylate kinase family — phosphate transfer repetition loop iters 590-960
    "phosphate group from atp to the 3'-hydroxyl",
    # C-type lectin / mannose receptor family — 118 near-identical NCBI boilerplate entries
    # Specific to "member of the C-type lectin X family/superfamily" template (not UniProt mechanistic entries)
    "member of the c-type lectin",
    "c-type lectin superfamily",
    "c-type lectin/c- type mannose receptor",
    "c-type mannose receptor",
    # Peptidase families C19/S1/M10 — 111 entries with identical "member of the peptidase X family" prefix
    "peptidase c19 family",
    "peptidase s1 family",
    "peptidase m10 family",
    # TRIM / tripartite motif family — 50 near-identical entries
    "tripartite motif (trim) family",
    "tripartite motif-containing protein",
    # RNA-binding protein generic — 45 entries with no distinguishing content
    "member of the rna-binding protein family",
    # Kinesin — 27 entries with identical "member of the kinesin-like protein family" prefix
    "kinesin-like protein family",
    "member of the kinesin",
    # Cadherin family — 36 near-identical entries
    "member of the cadherin family",
    "type ii classical cadherin",
    "type i classical cadherin",
    # WD repeat — 24 entries
    "member of the wd repeat protein family",
    # ASB / ankyrin repeat — 22 entries
    "ankyrin repeat and socs box",
]

# Prefix added to all regulatory/ENCODE descriptions
# Model learns this prefix signals a non-coding regulatory element
# Original description is preserved after the prefix — keeps specific biology
REGULATORY_PREFIX = (
    "This is a cis-regulatory element, not a protein-coding gene. "
    "It does not produce a protein or functional mRNA transcript. "
)

# All data splits to clean
DATA_DIRS = [
    "/home/namdo/applications/data_hm/train_hm",
    "/home/namdo/applications/data_hm/valid_hm",
    "/home/namdo/applications/data_hm/test_hm",
]

# Filtering thresholds
MAX_TEMPLATE_COUNT = 1   # drop summaries whose exact text OR first-200-char prefix appears > this many times
MIN_WORD_COUNT     = 30  # drop summaries shorter than this many words
PREFIX_LEN         = 300 # characters used to detect near-duplicate family boilerplate

# Sentence starters that are MGI ontology boilerplate.
# These sentences are stripped from ALL summaries before saving.
# If what remains is < MIN_WORD_COUNT, the entry is dropped entirely.
# This handles mixed entries like TBX18: real sentences are kept, GO-term noise removed.
MGI_STRIP_STARTS = [
    "enables ",               # GO molecular function term
    "contributes to ",        # GO molecular function term
    "acts upstream of or within",  # GO biological process term
    "predicted to enable",
    "predicted to be located",
    "predicted to be active",
    "predicted to be involved",
    "predicted to contribute",
    "located in ",            # GO cellular component term
    "active in ",             # GO cellular component term
    "part of ",               # GO cellular component complex
    "is expressed in ",       # MGI expression pattern
    "used to study",          # MGI disease link (no mechanism)
    "orthologous to human",   # MGI ortholog boilerplate
    "human ortholog",         # MGI ortholog boilerplate
]

# Keywords indicating regulatory/ENCODE elements (not protein-coding genes)
REGULATORY_KEYWORDS = [
    # ENCODE regulatory elements
    "sharpr-mpra",
    "chip-starr-seq",
    "atac-starr-seq",
    "starr-seq",
    "chromatin state analysis",
    "h3k27ac",
    "h3k4me1",
    "candidate strong enhancer",
    "candidate weak enhancer",
    "open chromatin",
    "encode (encyclopedia",
    "encyclopedia of dna elements",
    # Long non-coding RNAs — don't encode proteins, cause mode collapse if over-represented
    "long non-coding rna",
    "long noncoding rna",
    "lncrna",
    "non-coding rna gene",
    "noncoding rna",
    "this is a long non-coding",
    "does not code for a protein",
    "does not encode a protein",
    "non-protein coding",
    "pseudogene",
]


def _split_sentences(text):
    """Split summary into sentences."""
    import re as _re
    parts = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 5]


def strip_mgi_sentences(summary):
    """Remove MGI boilerplate sentences from a summary; return remaining prose."""
    sents = _split_sentences(summary)
    kept = [
        s for s in sents
        if not any(s.lower().startswith(kw) for kw in MGI_STRIP_STARTS)
    ]
    return " ".join(kept).strip()


def load_human_gene_ids(gene_info_path="gene_info"):
    """Return set of Gene IDs belonging to Homo sapiens (tax_id=9606).
    gene_info columns: tax_id (0), GeneID (1), ... — only need first 2.
    """
    human_ids = set()
    if not os.path.exists(gene_info_path):
        print(f"  WARNING: gene_info not found at {gene_info_path}, skipping human filter.")
        return None
    with open(gene_info_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split('\t', 2)  # only split into first 2 columns
            if parts[0] == '9606':
                human_ids.add(parts[1])
    print(f"  Loaded {len(human_ids):,} human gene IDs from gene_info.")
    return human_ids


def load_biological_region_ids(kw_path):
    """Return set of Gene IDs whose locus_type is 'biological-region' (ENCODE elements)."""
    if not os.path.exists(kw_path):
        return set()
    data = json.load(open(kw_path))
    biological_ids = set()
    for entry in data:
        if entry.get("Q_Id") == 1 and "biological-region" in entry.get("A", "").lower():
            biological_ids.add(str(entry.get("Gene Id", "")))
    print(f"  Found {len(biological_ids):,} genes with locus_type=biological-region (ENCODE elements)")
    return biological_ids


def load_no_mrna_ids(data_dir):
    """Return set of Gene IDs that have no mRNA (NM_/XM_) record in NCBI.
    These are confirmed non-coding/regulatory genes from prepare_mrna_data.py output."""
    no_nm_path = os.path.join(data_dir, "mrna_no_nm.txt")
    if not os.path.exists(no_nm_path):
        return set()
    with open(no_nm_path) as f:
        ids = set(line.strip() for line in f if line.strip())
    print(f"  Found {len(ids):,} genes with no mRNA record (mrna_no_nm.txt)")
    return ids


def filter_summaries(src_path, biological_region_ids, no_mrna_ids=None, human_ids=None):
    print(f"Loading {src_path} ...")
    data = json.load(open(src_path))
    print(f"  Total entries: {len(data):,}")

    # Pass 1: count occurrences of each summary text AND first-200-char prefix
    # Use citation-stripped text so counts match what Pass 2 will compare against
    summary_counts = Counter(strip_citations(entry.get("Summary", "").strip()) for entry in data)
    prefix_counts  = Counter(strip_citations(entry.get("Summary", "").strip())[:PREFIX_LEN] for entry in data)

    # Pass 2: filter
    kept           = []
    regulatory     = []   # ENCODE/non-coding elements → relabelled, not dropped
    reason         = Counter()
    prefix_kept    = defaultdict(int)  # track how many we kept per family prefix
    family_kept    = defaultdict(int)  # track how many we kept per known boilerplate family

    for entry in data:
        summary = strip_citations(entry.get("Summary", "").strip())
        gene_id = str(entry.get("Gene Id", ""))

        # Biological-region genes → prepend regulatory prefix, keep original description
        if gene_id in biological_region_ids:
            desc = (REGULATORY_PREFIX + summary) if summary else REGULATORY_PREFIX.strip()
            regulatory.append({"Gene Id": entry.get("Gene Id"), "Summary": desc})
            reason["biological_region"] += 1
            continue

        # Genes confirmed to have no mRNA → also regulatory, use fallback if no description
        if no_mrna_ids and gene_id in no_mrna_ids:
            desc = (REGULATORY_PREFIX + summary) if summary else REGULATORY_PREFIX.strip()
            regulatory.append({"Gene Id": entry.get("Gene Id"), "Summary": desc})
            reason["no_mrna"] += 1
            continue

        # Human-only filter
        if human_ids is not None and gene_id not in human_ids:
            reason["non_human"] += 1
            continue

        if not summary:
            reason["empty"] += 1
            continue

        if summary.startswith("Predicted to"):
            reason["predicted_to"] += 1
            continue

        if summary.startswith("This genomic sequence was predicted to be a transcriptional regulatory region"):
            regulatory.append({"Gene Id": entry.get("Gene Id"), "Summary": REGULATORY_PREFIX + summary})
            reason["encode_regulatory"] += 1
            continue

        # Summaries with regulatory-element-specific language → prepend prefix, keep text
        summary_lower = summary.lower()
        if any(kw in summary_lower for kw in REGULATORY_KEYWORDS):
            regulatory.append({"Gene Id": entry.get("Gene Id"), "Summary": REGULATORY_PREFIX + summary})
            reason["encode_regulatory"] += 1
            continue

        # Hard exclusion — families that cause mode collapse even with 1 representative
        if any(kw in summary_lower for kw in HARD_EXCLUDE_KEYWORDS):
            reason["hard_exclude"] += 1
            continue

        # Strip MGI boilerplate sentences — keep only prose mechanism sentences.
        # Applies to ALL entries (handles mixed entries like TBX18).
        # Use stripped version if >= 20 words; drop if nothing meaningful remains.
        stripped = strip_mgi_sentences(summary)
        if len(stripped.split()) < 20:
            reason["pure_mgi"] += 1
            continue
        summary = stripped

        # Exact duplicate boilerplate (same text > 3 times)
        if summary_counts[summary] > MAX_TEMPLATE_COUNT:
            reason["boilerplate"] += 1
            continue

        # Near-duplicate family boilerplate — prefix-based dedup
        prefix = summary[:PREFIX_LEN]
        if prefix_counts[prefix] > MAX_TEMPLATE_COUNT:
            if prefix_kept[prefix] >= MAX_TEMPLATE_COUNT:
                reason["near_duplicate_family"] += 1
                continue
            prefix_kept[prefix] += 1

        # Explicit family cap — catches families with multiple boilerplate templates
        # that differ in their first PREFIX_LEN chars (e.g. S100, Serpin, GPCR)
        summary_lower = summary.lower()
        matched_family = None
        for kw, fid in FAMILY_BOILERPLATE.items():
            if kw in summary_lower:
                matched_family = fid
                break
        if matched_family is not None:
            if family_kept[matched_family] >= MAX_TEMPLATE_COUNT:
                reason["family_boilerplate_explicit"] += 1
                continue
            family_kept[matched_family] += 1

        # Save with the citation-stripped summary, not the original
        kept.append({"Gene Id": entry.get("Gene Id"), "Summary": summary})

    print(f"\nFiltering results:")
    print(f"  → Real gene (clean):         {len(kept):>7,}  ({100*len(kept)/len(data):.1f}% of total)")
    print(f"  → Regulatory (relabelled):   {len(regulatory):>7,}  (saved to qa_summary_regulatory.json)")
    print(f"  Breakdown of regulatory:")
    print(f"    biological-region:         {reason['biological_region']:>7,}")
    print(f"    no mRNA record:            {reason['no_mrna']:>7,}")
    print(f"    ENCODE regulatory text:    {reason['encode_regulatory']:>7,}")
    print(f"  Dropped (empty):             {reason['empty']:>7,}")
    print(f"  Dropped (hard exclude):      {reason['hard_exclude']:>7,}  (GPCR/olfactory/CYP — collapse-prone)")
    print(f"  Dropped (pure MGI ontology): {reason['pure_mgi']:>7,}  (GO-term lists, no prose mechanism)")
    print(f"  Dropped (Predicted to):      {reason['predicted_to']:>7,}")
    print(f"  Dropped (non-human):         {reason['non_human']:>7,}  (mouse/other species excluded)")
    print(f"  Dropped (too short):         {reason['too_short']:>7,}")
    print(f"  Dropped (boilerplate):       {reason['boilerplate']:>7,}")
    print(f"  Dropped (family near-dup):   {reason['near_duplicate_family']:>7,}  (e.g. S100 family, ribosomal proteins)")
    print(f"  Dropped (explicit family):   {reason['family_boilerplate_explicit']:>7,}  (S100/Serpin/GPCR/keratin variants)")

    return kept, regulatory


def process_dir(data_dir, human_ids=None):
    print(f"\n{'='*60}")
    print(f"Processing: {data_dir}")

    kw_path = os.path.join(data_dir, "qa_kw.json")
    print(f"Loading locus types from {kw_path} ...")
    biological_region_ids = load_biological_region_ids(kw_path)
    no_mrna_ids = load_no_mrna_ids(data_dir)

    src_summary = os.path.join(data_dir, "qa_summary_rule.json")
    dst_summary = os.path.join(data_dir, "qa_summary_rule_clean.json")

    if not os.path.exists(src_summary):
        print(f"  No qa_summary_rule.json found, skipping.")
        return

    ncbi_clean, regulatory_data = filter_summaries(src_summary, biological_region_ids, no_mrna_ids, human_ids)

    # Save regulatory elements with unified description
    reg_dst = os.path.join(data_dir, "qa_summary_regulatory.json")
    seen_ids = set()
    regulatory_dedup = []
    for e in regulatory_data:
        gid = str(e.get("Gene Id", ""))
        if gid not in seen_ids:
            seen_ids.add(gid)
            regulatory_dedup.append(e)
    with open(reg_dst, "w") as f:
        json.dump(regulatory_dedup, f, indent=2)
    print(f"Saved {len(regulatory_dedup):,} regulatory elements to {reg_dst}")

    # ── UniProt: filter non-coding, citations, short, family near-dups ────────
    uniprot_src = os.path.join(data_dir, "qa_summary_uniprot.json")
    uniprot_dst = os.path.join(data_dir, "qa_summary_uniprot_clean.json")
    kept_uniprot = []
    if os.path.exists(uniprot_src):
        before = json.load(open(uniprot_src))
        coding = [e for e in before
                  if str(e.get("Gene Id","")) not in biological_region_ids
                  and str(e.get("Gene Id","")) not in no_mrna_ids]
        coding_cleaned = []
        for e in coding:
            cleaned = strip_citations(e.get("Summary", "").strip())
            if len(cleaned.split()) < MIN_WORD_COUNT:
                continue
            coding_cleaned.append({"Gene Id": e.get("Gene Id"), "Summary": cleaned})
        coding = coding_cleaned

        uni_prefix_counts  = Counter(e.get("Summary","").strip()[:PREFIX_LEN] for e in coding)
        uni_prefix_kept    = defaultdict(int)
        uni_family_kept    = defaultdict(int)
        dropped_uni_dup = 0
        for e in coding:
            summary_u = e.get("Summary","").strip()
            summary_u_lower = summary_u.lower()
            if any(kw in summary_u_lower for kw in HARD_EXCLUDE_KEYWORDS):
                dropped_uni_dup += 1
                continue
            prefix = summary_u[:PREFIX_LEN]
            if uni_prefix_counts[prefix] > MAX_TEMPLATE_COUNT:
                if uni_prefix_kept[prefix] >= MAX_TEMPLATE_COUNT:
                    dropped_uni_dup += 1
                    continue
                uni_prefix_kept[prefix] += 1
            matched = None
            for kw, fid in FAMILY_BOILERPLATE.items():
                if kw in summary_u_lower:
                    matched = fid
                    break
            if matched is not None:
                if uni_family_kept[matched] >= MAX_TEMPLATE_COUNT:
                    dropped_uni_dup += 1
                    continue
                uni_family_kept[matched] += 1
            kept_uniprot.append(e)
        with open(uniprot_dst, "w") as f:
            json.dump(kept_uniprot, f, indent=2)
        print(f"UniProt: {len(before):,} → {len(coding):,} (dropped {len(before)-len(coding):,} non-coding)"
              f" → {len(kept_uniprot):,} (dropped {dropped_uni_dup:,} family near-dups)")
        print(f"Saved cleaned UniProt to {uniprot_dst}")

    # ── NCBI: keep only genes NOT covered by UniProt (one description per gene) ─
    # UniProt descriptions are manually curated and higher quality; NCBI is fallback only.
    uniprot_ids = {str(e.get("Gene Id", "")) for e in kept_uniprot}
    ncbi_fallback = [e for e in ncbi_clean if str(e.get("Gene Id", "")) not in uniprot_ids]
    print(f"Gene dedup: {len(ncbi_clean):,} NCBI → {len(ncbi_fallback):,} fallback "
          f"(removed {len(ncbi_clean)-len(ncbi_fallback):,} genes already covered by UniProt)")
    with open(dst_summary, "w") as f:
        json.dump(ncbi_fallback, f, indent=2)
    print(f"Saved NCBI fallback to {dst_summary}")


def main():
    human_ids = load_human_gene_ids()
    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"Directory not found, skipping: {data_dir}")
            continue
        process_dir(data_dir, human_ids=human_ids)


if __name__ == "__main__":
    main()
