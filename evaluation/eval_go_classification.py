"""
GO-based Gene Classification Evaluation for GeneChat
=====================================================
Evaluates GeneChat on Gene Ontology classification tasks, analogous to
ProteinChat's evaluation on UniProtKB classification tasks (Fig. 3a).

Three classification tasks following ProteinChat methodology:
1. Molecular Function (7 categories) - analogous to ProteinChat's catalytic function
2. Biological Process (8 categories) - analogous to ProteinChat's biological process
3. Cellular Component (6 categories) - analogous to ProteinChat's cellular component

Usage:
    python eval_go_classification.py \
        --cfg-path configs/genechat_eval.yaml \
        --test-seq test_data/seq.json \
        --gene2go gene2go \
        --output go_classification_results.json \
        --max-samples 100
"""

import argparse
import json
import os
import re
import time
import random
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report
import copy

# Lazy imports for GPU-dependent modules
torch = None
cudnn = None
CONV_VISION = None

def _import_genechat():
    """Import torch and genechat modules."""
    global torch, cudnn, CONV_VISION
    import torch as _torch
    import torch.backends.cudnn as _cudnn
    torch = _torch
    cudnn = _cudnn
    from genechat.common.config import Config
    from genechat.common.registry import registry
    from genechat.common.dist_utils import get_rank, init_distributed_mode
    from genechat.common.conversation import Chat, CONV_VISION as _CONV_VISION
    CONV_VISION = _CONV_VISION
    import importlib
    importlib.import_module("genechat.datasets.builders")
    importlib.import_module("genechat.models")
    importlib.import_module("genechat.runners")
    importlib.import_module("genechat.tasks")
    return Config, registry, init_distributed_mode, Chat, CONV_VISION


# ========================================
#    GO Classification Task Definitions
# ========================================

# Task 1: Molecular Function (analogous to ProteinChat's catalytic function + ligand binding)
# Categories chosen to be non-overlapping and with sufficient test samples
GO_TASKS = {
    "molecular_function": {
        "display_name": "Molecular Function",
        "prompt": "What is the molecular function of this gene? Choose from: kinase, transcription factor, receptor, hydrolase, transferase, oxidoreductase, transporter.",
        "categories": [
            "kinase", "transcription factor", "receptor",
            "hydrolase", "transferase", "oxidoreductase", "transporter",
        ],
        "go_term_mapping": {
            # Map GO terms -> our category labels
            "protein serine/threonine kinase activity": "kinase",
            "protein kinase activity": "kinase",
            "kinase activity": "kinase",
            "protein tyrosine kinase activity": "kinase",
            "DNA-binding transcription factor activity, RNA polymerase II-specific": "transcription factor",
            "DNA-binding transcription factor activity": "transcription factor",
            "G protein-coupled receptor activity": "receptor",
            "signaling receptor activity": "receptor",
            "transmembrane signaling receptor activity": "receptor",
            "hydrolase activity": "hydrolase",
            "transferase activity": "transferase",
            "oxidoreductase activity": "oxidoreductase",
            "transmembrane transporter activity": "transporter",
            "transporter activity": "transporter",
            "transmembrane transport": "transporter",
        },
        "go_category": "Function",
        "answer_aliases": {
            "kinase": "kinase", "protein kinase": "kinase", "serine/threonine kinase": "kinase",
            "transcription factor": "transcription factor", "tf": "transcription factor",
            "dna-binding transcription factor": "transcription factor",
            "receptor": "receptor", "g protein-coupled receptor": "receptor", "gpcr": "receptor",
            "hydrolase": "hydrolase",
            "transferase": "transferase",
            "oxidoreductase": "oxidoreductase",
            "transporter": "transporter", "ion channel": "transporter", "channel": "transporter",
        },
    },

    "biological_process": {
        "display_name": "Biological Process",
        "prompt": "What biological process is this gene involved in? Choose from: signal transduction, transcription regulation, proteolysis, transmembrane transport, cell differentiation, DNA repair, immune response, cell division.",
        "categories": [
            "signal transduction", "transcription regulation", "proteolysis",
            "transmembrane transport", "cell differentiation", "DNA repair",
            "immune response", "cell division",
        ],
        "go_term_mapping": {
            "signal transduction": "signal transduction",
            "intracellular signal transduction": "signal transduction",
            "G protein-coupled receptor signaling pathway": "signal transduction",
            "regulation of transcription by RNA polymerase II": "transcription regulation",
            "positive regulation of transcription by RNA polymerase II": "transcription regulation",
            "negative regulation of transcription by RNA polymerase II": "transcription regulation",
            "regulation of DNA-templated transcription": "transcription regulation",
            "positive regulation of DNA-templated transcription": "transcription regulation",
            "negative regulation of DNA-templated transcription": "transcription regulation",
            "proteolysis": "proteolysis",
            "transmembrane transport": "transmembrane transport",
            "monoatomic ion transport": "transmembrane transport",
            "cell differentiation": "cell differentiation",
            "DNA repair": "DNA repair",
            "immune response": "immune response",
            "innate immune response": "immune response",
            "adaptive immune response": "immune response",
            "cell division": "cell division",
            "mitotic cell cycle": "cell division",
            "cell cycle": "cell division",
            "mitosis": "cell division",
        },
        "go_category": "Process",
        "answer_aliases": {
            "signal transduction": "signal transduction", "signaling": "signal transduction",
            "transcription regulation": "transcription regulation",
            "transcription": "transcription regulation",
            "gene regulation": "transcription regulation",
            "proteolysis": "proteolysis", "protein degradation": "proteolysis",
            "transmembrane transport": "transmembrane transport", "transport": "transmembrane transport",
            "ion transport": "transmembrane transport",
            "cell differentiation": "cell differentiation", "differentiation": "cell differentiation",
            "dna repair": "DNA repair", "repair": "DNA repair",
            "immune response": "immune response", "immunity": "immune response",
            "cell division": "cell division", "mitosis": "cell division",
            "cell cycle": "cell division",
        },
    },

    "cellular_component": {
        "display_name": "Cellular Component",
        "prompt": "What is the cellular localization of this gene's product? Choose from: nucleus, cytoplasm, plasma membrane, mitochondrion, endoplasmic reticulum, extracellular.",
        "categories": [
            "nucleus", "cytoplasm", "plasma membrane",
            "mitochondrion", "endoplasmic reticulum", "extracellular",
        ],
        "go_term_mapping": {
            "nucleus": "nucleus",
            "nucleoplasm": "nucleus",
            "nucleolus": "nucleus",
            "chromatin": "nucleus",
            "cytoplasm": "cytoplasm",
            "cytosol": "cytoplasm",
            "plasma membrane": "plasma membrane",
            "membrane": "plasma membrane",
            "mitochondrion": "mitochondrion",
            "mitochondrial inner membrane": "mitochondrion",
            "mitochondrial matrix": "mitochondrion",
            "endoplasmic reticulum": "endoplasmic reticulum",
            "endoplasmic reticulum membrane": "endoplasmic reticulum",
            "Golgi apparatus": "endoplasmic reticulum",
            "extracellular region": "extracellular",
            "extracellular space": "extracellular",
            "secreted": "extracellular",
        },
        "go_category": "Component",
        "answer_aliases": {
            "nucleus": "nucleus", "nuclear": "nucleus",
            "cytoplasm": "cytoplasm", "cytosol": "cytoplasm", "cytoplasmic": "cytoplasm",
            "plasma membrane": "plasma membrane", "membrane": "plasma membrane",
            "cell membrane": "plasma membrane",
            "mitochondrion": "mitochondrion", "mitochondria": "mitochondrion",
            "mitochondrial": "mitochondrion",
            "endoplasmic reticulum": "endoplasmic reticulum", "er": "endoplasmic reticulum",
            "golgi": "endoplasmic reticulum", "golgi apparatus": "endoplasmic reticulum",
            "extracellular": "extracellular", "secreted": "extracellular",
            "extracellular space": "extracellular", "extracellular region": "extracellular",
        },
    },
}


def load_go_annotations(gene2go_path, gene_ids):
    """Load GO annotations for specified gene IDs, return {gene_id: {category: [terms]}}."""
    annotations = defaultdict(lambda: defaultdict(set))
    with open(gene2go_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            gid = parts[1]
            if gid in gene_ids:
                category = parts[7]  # Function, Process, Component
                go_term = parts[5]
                annotations[gid][category].add(go_term)
    return annotations


def assign_gene_to_category(gene_go_terms, task_def):
    """Assign a gene to a single category based on its GO terms.

    Priority: assign to the MOST SPECIFIC matching category.
    If a gene matches multiple categories, pick the first match in priority order.
    """
    go_category = task_def["go_category"]
    term_mapping = task_def["go_term_mapping"]

    matched_categories = set()
    for go_term in gene_go_terms.get(go_category, set()):
        if go_term in term_mapping:
            matched_categories.add(term_mapping[go_term])

    if len(matched_categories) == 1:
        return list(matched_categories)[0]
    elif len(matched_categories) > 1:
        # If multiple matches, pick the first one in category order (priority)
        for cat in task_def["categories"]:
            if cat in matched_categories:
                return cat
    return None


def build_go_test_data(annotations, task_name, task_def, max_per_class=100, seed=42):
    """Build balanced test data for a GO classification task."""
    rng = random.Random(seed)

    # Assign each gene to a category
    gene_labels = {}
    for gene_id, go_terms in annotations.items():
        label = assign_gene_to_category(go_terms, task_def)
        if label is not None:
            gene_labels[gene_id] = label

    # Group by category
    by_category = defaultdict(list)
    for gene_id, label in gene_labels.items():
        by_category[label].append(gene_id)

    print(f"\n  {task_name} ({task_def['display_name']}):")
    for cat in task_def["categories"]:
        print(f"    {cat}: {len(by_category[cat])} genes")

    # Sample up to max_per_class from each category
    test_data = []
    for cat in task_def["categories"]:
        genes = by_category[cat]
        if len(genes) == 0:
            continue
        sampled = rng.sample(genes, min(len(genes), max_per_class))
        for gene_id in sampled:
            test_data.append({
                "gene_id": gene_id,
                "true_label": cat,
                "task": task_name,
            })

    rng.shuffle(test_data)
    print(f"    Total test samples: {len(test_data)}")
    return test_data


def match_answer(raw_answer, task_def):
    """Match a generated answer to a category."""
    if raw_answer is None:
        return None

    raw = raw_answer.strip().lower()

    # Direct match
    for cat in task_def["categories"]:
        if cat.lower() in raw:
            return cat

    # Alias match
    for alias, cat in task_def["answer_aliases"].items():
        if alias.lower() in raw:
            return cat

    return None


def classify_generate(chat, seq, task_def):
    """Classify a gene using GeneChat generation."""
    # Concatenate multi-segment sequences and truncate to avoid OOM
    MAX_SEQ = 80000
    if isinstance(seq, list):
        seq = ["".join(seq)[:MAX_SEQ]]
    elif isinstance(seq, str):
        seq = seq[:MAX_SEQ]
    conv = copy.deepcopy(CONV_VISION)
    img_list = []
    chat.upload_gene(seq, conv, img_list)
    chat.ask(task_def["prompt"], conv)
    response = chat.answer(conv, img_list, max_new_tokens=64, temperature=0.1)[0]
    return response


def main():
    parser = argparse.ArgumentParser(description="GO Classification Evaluation for GeneChat")
    parser.add_argument("--cfg-path", required=True, help="Config YAML for GeneChat model")
    parser.add_argument("--test-seq", default="test_data/seq.json", help="Path to seq.json")
    parser.add_argument("--gene2go", default="gene2go", help="Path to NCBI gene2go file")
    parser.add_argument("--output", default="go_classification_results.json", help="Output JSON")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per class")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--tasks", nargs="+",
                        default=["molecular_function", "biological_process", "cellular_component"],
                        help="Which tasks to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    args = parser.parse_args()

    # Import GeneChat
    print("Loading GeneChat model...")
    Config, registry, init_distributed_mode, Chat, conv_vision = _import_genechat()

    cfg = Config(args)
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{args.gpu_id}")
    model.eval()

    chat = Chat(model, device=f"cuda:{args.gpu_id}")
    print("Model loaded.")

    # Load sequences
    print(f"Loading sequences from {args.test_seq}...")
    with open(args.test_seq) as f:
        seqs = json.load(f)
    print(f"  {len(seqs)} sequences loaded")

    # Load GO annotations
    gene_ids = set(seqs.keys())
    print(f"Loading GO annotations for {len(gene_ids)} genes...")
    annotations = load_go_annotations(args.gene2go, gene_ids)
    print(f"  {len(annotations)} genes have GO annotations")

    # Run classification for each task
    all_results = {"method": "GeneChat (Generate)", "model": "genechat-go", "tasks": {}, "summary": []}

    for task_name in args.tasks:
        if task_name not in GO_TASKS:
            print(f"Unknown task: {task_name}, skipping")
            continue

        task_def = GO_TASKS[task_name]
        print(f"\n{'='*60}")
        print(f"Task: {task_def['display_name']}")
        print(f"{'='*60}")

        # Build test data
        test_data = build_go_test_data(annotations, task_name, task_def,
                                        max_per_class=args.max_samples, seed=args.seed)

        if len(test_data) == 0:
            print(f"  No test data for {task_name}, skipping")
            continue

        # Classify each gene
        y_true, y_pred = [], []
        details = []
        start_time = time.time()

        for i, item in enumerate(test_data):
            gene_id = item["gene_id"]
            true_label = item["true_label"]

            if gene_id not in seqs:
                continue

            seq = seqs[gene_id]

            try:
                raw_response = classify_generate(chat, seq, task_def)
                predicted = match_answer(raw_response, task_def)
            except Exception as e:
                print(f"  Error on gene {gene_id}: {e}")
                raw_response = str(e)
                predicted = None

            if i < 10:
                print(f"  [{i+1}/{len(test_data)}] Gene={gene_id}, True={true_label}, "
                      f"Pred={predicted}, Raw='{raw_response[:80]}'")
            elif (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  [{i+1}/{len(test_data)}] {elapsed:.0f}s elapsed")

            if predicted is not None:
                y_true.append(true_label)
                y_pred.append(predicted)
                details.append({
                    "gene_id": gene_id,
                    "true": true_label,
                    "predicted": predicted,
                    "raw_response": raw_response[:200],
                })
            else:
                details.append({
                    "gene_id": gene_id,
                    "true": true_label,
                    "predicted": None,
                    "raw_response": raw_response[:200] if raw_response else None,
                })

        elapsed = time.time() - start_time

        # Compute metrics
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        else:
            acc = macro_f1 = weighted_f1 = 0.0
            cls_report = {}

        task_result = {
            "task": task_name,
            "display_name": task_def["display_name"],
            "method": "GeneChat (Generate)",
            "num_samples": len(test_data),
            "num_valid": len(y_true),
            "num_unparseable": len(test_data) - len(y_true),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "elapsed_seconds": elapsed,
            "classification_report": cls_report,
        }

        all_results["tasks"][task_name] = task_result
        all_results["summary"].append({
            "task": task_name,
            "display_name": task_def["display_name"],
            "method": "GeneChat (Generate)",
            "num_samples": len(test_data),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        })

        print(f"\n  Results for {task_def['display_name']}:")
        print(f"    Samples: {len(test_data)} total, {len(y_true)} valid")
        print(f"    Accuracy:    {acc:.4f}")
        print(f"    Macro F1:    {macro_f1:.4f}")
        print(f"    Weighted F1: {weighted_f1:.4f}")
        print(f"    Time: {elapsed:.1f}s")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'='*70}")
    print("GO CLASSIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<25} {'Samples':>8} {'Valid':>8} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 70)
    for s in all_results["summary"]:
        print(f"{s['display_name']:<25} {s['num_samples']:>8} "
              f"{all_results['tasks'][s['task']]['num_valid']:>8} "
              f"{s['accuracy']:>10.4f} {s['macro_f1']:>10.4f} {s['weighted_f1']:>12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
