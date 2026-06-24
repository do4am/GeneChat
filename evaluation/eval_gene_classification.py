"""
Gene Classification Evaluation for GeneChat
=============================================
Evaluates GeneChat on discrete-category classification tasks using PPL-based scoring,
following the ProteinChat methodology (Huo et al., 2024).

Supports multiple methods:
  - genechat:        GeneChat PPL-based classification (requires GPU)
  - claude-seq:      Claude API with DNA sequence input
  - claude-genename: Claude API with gene name only
  - gpt4-seq:        GPT-4 API with DNA sequence input
  - gpt4-genename:   GPT-4 API with gene name only
  - random:          Random baseline (lower bound)

Four classification tasks using NCBI gene categories:
1. Organism Classification (10 categories)
2. Locus Type Classification (6 categories)
3. Chromosome Classification (grouped to chromosome level)
4. Exon Count Classification (binned into 6 ranges)

Usage:
    # GeneChat only
    python eval_gene_classification.py \
        --cfg-path configs/genechat_eval.yaml \
        --test-qa test_data/qa_kw.json \
        --test-seq test_data/seq.json \
        --output gene_classification_results.json

    # Claude Sonnet 4.5 with DNA sequence
    python eval_gene_classification.py \
        --method claude-seq \
        --claude-api-key $ANTHROPIC_API_KEY \
        --claude-model claude-sonnet-4-5-20250929 \
        --test-qa test_data/qa_kw.json \
        --test-seq test_data/seq.json \
        --output classification_claude_sonnet.json

    # Claude Opus 4.5 with gene name only
    python eval_gene_classification.py \
        --method claude-genename \
        --claude-api-key $ANTHROPIC_API_KEY \
        --claude-model claude-opus-4-5-20251101 \
        --test-qa test_data/qa_kw.json \
        --output classification_claude_opus_genename.json

    # GPT-4 with DNA sequence
    python eval_gene_classification.py \
        --method gpt4-seq \
        --openai-api-key $OPENAI_API_KEY \
        --test-qa test_data/qa_kw.json \
        --test-seq test_data/seq.json \
        --output classification_gpt4.json
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

# Lazy imports for GPU-dependent modules (torch, genechat)
# These are only loaded when method requires GeneChat model (genechat, genechat-generate)
torch = None
cudnn = None
CONV_VISION = None

def _import_genechat():
    """Import torch and genechat modules. Called only for genechat methods."""
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
    # imports modules for registration
    import importlib
    importlib.import_module("genechat.datasets.builders")
    importlib.import_module("genechat.models")
    importlib.import_module("genechat.runners")
    importlib.import_module("genechat.tasks")
    return Config, registry, init_distributed_mode, Chat, CONV_VISION


# ========================================
#         Classification Task Definitions
# ========================================

CLASSIFICATION_TASKS = {
    "organism": {
        "q_id": 0,
        # Use EXACT training prompt from seq_dataset.py
        "prompt": "Which organism does the gene belong to? Limit your answer to one or two words.",
        "categories": [
            "Homo sapiens", "Mus musculus", "Danio rerio",
            "Drosophila melanogaster", "Rattus norvegicus",
            "Saccharomyces cerevisiae", "Caenorhabditis elegans",
            "Pseudomonas aeruginosa", "Arabidopsis thaliana", "Equus caballus",
        ],
        # PPL candidates: use the DOMINANT form from training data
        # Training counts: sapiens(60K), Mus musculus(51K), Rattus norvegicus(45K),
        # Danio rerio(33K), Homo sapiens(2K), Equus caballus(1.6K),
        # aeruginosa(1.5K), elegans(1.2K), Drosophila melanogaster(1K),
        # thaliana(664), Saccharomyces cerevisiae S288C(451)
        "ppl_candidates": [
            "sapiens",                          # 60K in train (dominant for H. sapiens)
            "Mus musculus",                     # 51K in train
            "Rattus norvegicus",                # 45K in train
            "Danio rerio",                      # 33K in train
            "Equus caballus",                   # 1.6K in train
            "aeruginosa",                       # 1.5K in train
            "elegans",                          # 1.2K in train
            "Drosophila melanogaster",          # 1K in train
            "thaliana",                         # 664 in train
            "Saccharomyces cerevisiae S288C",   # 451 in train
        ],
        # Map PPL candidate -> canonical category
        "ppl_candidate_map": {
            "sapiens": "Homo sapiens",
            "Mus musculus": "Mus musculus",
            "Rattus norvegicus": "Rattus norvegicus",
            "Danio rerio": "Danio rerio",
            "Equus caballus": "Equus caballus",
            "aeruginosa": "Pseudomonas aeruginosa",
            "elegans": "Caenorhabditis elegans",
            "Drosophila melanogaster": "Drosophila melanogaster",
            "thaliana": "Arabidopsis thaliana",
            "Saccharomyces cerevisiae S288C": "Saccharomyces cerevisiae",
        },
        # For generation-based: map generated text -> canonical category
        "answer_aliases": {
            "sapiens": "Homo sapiens", "homo sapiens": "Homo sapiens",
            "musculus": "Mus musculus", "mus musculus": "Mus musculus",
            "rerio": "Danio rerio", "danio rerio": "Danio rerio",
            "melanogaster": "Drosophila melanogaster", "drosophila melanogaster": "Drosophila melanogaster",
            "norvegicus": "Rattus norvegicus", "rattus norvegicus": "Rattus norvegicus",
            "cerevisiae": "Saccharomyces cerevisiae", "saccharomyces cerevisiae": "Saccharomyces cerevisiae",
            "saccharomyces cerevisiae s288c": "Saccharomyces cerevisiae",
            "elegans": "Caenorhabditis elegans", "caenorhabditis elegans": "Caenorhabditis elegans",
            "aeruginosa": "Pseudomonas aeruginosa", "pseudomonas aeruginosa": "Pseudomonas aeruginosa",
            "thaliana": "Arabidopsis thaliana", "arabidopsis thaliana": "Arabidopsis thaliana",
            "caballus": "Equus caballus", "equus caballus": "Equus caballus",
            "human": "Homo sapiens", "mouse": "Mus musculus", "rat": "Rattus norvegicus",
            "zebrafish": "Danio rerio", "fruit fly": "Drosophila melanogaster",
            "yeast": "Saccharomyces cerevisiae", "worm": "Caenorhabditis elegans",
            "horse": "Equus caballus",
        },
    },
    "locus_type": {
        "q_id": 1,
        "prompt": "What is the locus type of the gene? Limit your answer to one or two words.",
        # Training counts: protein-coding(75K), biological-region(59K), ncRNA(28K),
        # pseudo(23K), tRNA(5K), snoRNA(2K), rRNA(2K), snRNA(2K), other(1K)
        "categories": [
            "protein-coding", "biological-region", "ncRNA",
            "pseudo", "tRNA", "snoRNA", "rRNA", "snRNA", "other",
        ],
        # PPL candidates: exact training answer forms (top 9 categories)
        "ppl_candidates": [
            "protein-coding", "biological-region", "ncRNA",
            "pseudo", "tRNA", "snoRNA", "rRNA", "snRNA", "other",
        ],
        "ppl_candidate_map": {
            "protein-coding": "protein-coding", "biological-region": "biological-region",
            "ncRNA": "ncRNA", "pseudo": "pseudo", "tRNA": "tRNA", "snoRNA": "snoRNA",
            "rRNA": "rRNA", "snRNA": "snRNA", "other": "other",
        },
        "answer_aliases": {
            "protein coding": "protein-coding", "protein-coding": "protein-coding",
            "biological region": "biological-region", "biological-region": "biological-region",
            "ncrna": "ncRNA", "non-coding rna": "ncRNA", "non coding rna": "ncRNA",
            "pseudo": "pseudo", "pseudogene": "pseudo",
            "trna": "tRNA", "transfer rna": "tRNA",
            "snorna": "snoRNA", "small nucleolar rna": "snoRNA",
            "rrna": "rRNA", "ribosomal rna": "rRNA",
            "snrna": "snRNA", "small nuclear rna": "snRNA",
            "other": "other",
        },
    },
    "chromosome": {
        "q_id": 2,
        "prompt": "On which chromosome is the gene located? Limit your answer to one or two words.",
        "categories": [
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "X", "Y", "MT",
        ],
        # PPL candidates: bare numbers are 40% of train data (dominant format)
        # Also include "Chromosome X" format (16% of train)
        # We use bare numbers since they're shorter and more common
        "ppl_candidates": [
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "X", "Y", "MT",
        ],
        "ppl_candidate_map": None,  # identity mapping
        "answer_aliases": {
            "chromosome 1": "1", "chromosome 2": "2", "chromosome 3": "3",
            "chromosome 4": "4", "chromosome 5": "5", "chromosome 6": "6",
            "chromosome 7": "7", "chromosome 8": "8", "chromosome 9": "9",
            "chromosome 10": "10", "chromosome 11": "11", "chromosome 12": "12",
            "chromosome 13": "13", "chromosome 14": "14", "chromosome 15": "15",
            "chromosome 16": "16", "chromosome 17": "17", "chromosome 18": "18",
            "chromosome 19": "19", "chromosome 20": "20", "chromosome 21": "21",
            "chromosome 22": "22", "chromosome x": "X", "chromosome y": "Y",
        },
    },
    "exon_count": {
        "q_id": 3,
        "prompt": "How many exons does the gene contain? Limit your answer to one or two words.",
        "categories": [
            "1", "2-5", "6-10", "11-20", "21-50", "more than 50",
        ],
        # PPL candidates: individual numbers the model was trained on
        # We score individual numbers and then bin them
        "ppl_candidates": [str(i) for i in range(1, 61)] + ["75", "100", "125"],
        "ppl_candidate_map": None,  # handled by binning
        "answer_aliases": {},  # handled by normalize_exon_count()
    },
}


# ========================================
#         Answer Normalization
# ========================================

# Map short organism names in qa_kw.json to full names
ORGANISM_NORMALIZE = {
    "sapiens": "Homo sapiens",
    "homo sapiens": "Homo sapiens",
    "elegans": "Caenorhabditis elegans",
    "caenorhabditis elegans": "Caenorhabditis elegans",
    "aeruginosa": "Pseudomonas aeruginosa",
    "pseudomonas aeruginosa": "Pseudomonas aeruginosa",
    "thaliana": "Arabidopsis thaliana",
    "arabidopsis thaliana": "Arabidopsis thaliana",
    "mus musculus": "Mus musculus",
    "mus musculus musculus": "Mus musculus",
    "mus musculus domesticus": "Mus musculus",
    "mus musculus castaneus": "Mus musculus",
    "mus musculus molossinus": "Mus musculus",
    "equus caballus": "Equus caballus",
    "rattus norvegicus": "Rattus norvegicus",
    "drosophila melanogaster": "Drosophila melanogaster",
    "danio rerio": "Danio rerio",
    "saccharomyces cerevisiae s288c": "Saccharomyces cerevisiae",
    "saccharomyces cerevisiae": "Saccharomyces cerevisiae",
}


def normalize_organism(answer):
    return ORGANISM_NORMALIZE.get(answer.lower().strip(), None)


def normalize_chromosome(answer):
    """Normalize chromosome answers to bare chromosome number/letter.

    Handles formats from training data:
    - Bare: "1", "2", "X", "Y", "MT"
    - Prefixed: "Chromosome 4", "chromosome 1"
    - Cytogenetic band: "14q11.2", "Xq28", "5q36"
    - cM format: "1 38.38 cM", "4 62.0 cM"
    """
    answer = answer.strip()
    # "chromosome 1" or "Chromosome 4"
    m = re.match(r"[Cc]hromosome\s+(\w+)", answer)
    if m:
        return m.group(1).upper() if m.group(1).upper() in ("X", "Y", "MT") else m.group(1)
    # "1 38.38 cM" -> "1"
    m = re.match(r"^(\d+|X|Y)\s+[\d.]+\s*cM", answer, re.IGNORECASE)
    if m:
        return m.group(1).upper() if m.group(1).upper() in ("X", "Y") else m.group(1)
    # "14q11.2" -> "14", "Xq28" -> "X"
    m = re.match(r"^([0-9XYxy]+)[pq]", answer)
    if m:
        return m.group(1).upper() if m.group(1).upper() in ("X", "Y") else m.group(1)
    # Plain number or letter
    m = re.match(r"^(\d+|X|Y|MT|x|y|mt)$", answer)
    if m:
        return m.group(1).upper()
    return None


def normalize_exon_count(answer):
    """Bin exon count into categories."""
    try:
        n = int(answer)
    except (ValueError, TypeError):
        return None
    if n == 1:
        return "1"
    elif 2 <= n <= 5:
        return "2-5"
    elif 6 <= n <= 10:
        return "6-10"
    elif 11 <= n <= 20:
        return "11-20"
    elif 21 <= n <= 50:
        return "21-50"
    elif n > 50:
        return "more than 50"
    return None


NORMALIZERS = {
    "organism": normalize_organism,
    "locus_type": lambda a: next((c for c in CLASSIFICATION_TASKS["locus_type"]["categories"] if c.lower() == a.strip().lower()), None),
    "chromosome": normalize_chromosome,
    "exon_count": normalize_exon_count,
}


# ========================================
#         PPL-based Classification (GeneChat)
# ========================================

def get_ppl_independent(chat, conv, img_list, candidate_list):
    """
    Fixed PPL scoring: score each candidate INDEPENDENTLY.

    The original chat.get_ppl() has a bug where it concatenates all previous
    candidates' embeddings when scoring candidate N, making later candidates
    get scored in wrong context. This version scores each candidate in isolation.
    """
    # Build context embeddings once
    conv.append_message(conv.roles[1], None)
    embs = chat.get_context_emb(conv, img_list)

    conf_list = []
    with torch.no_grad():
        for candidate in candidate_list:
            # Tokenize this candidate independently
            tokens = chat.model.llama_tokenizer(
                candidate, return_tensors="pt", add_special_tokens=False
            ).to(chat.device).input_ids

            # Skip if tokenization produces too few tokens
            if tokens.shape[1] <= 1:
                conf_list.append(float('inf'))
                continue

            # Embed candidate tokens
            candidate_emb = chat.model.llama_model.model.embed_tokens(tokens)

            # Concatenate: [context_embs, candidate_embs]
            input_emb = torch.cat([embs, candidate_emb], dim=1)

            # Create labels: -100 for context (ignored), actual tokens for candidate
            # Skip first token of candidate (like original code)
            label_tokens = tokens[:, 1:]
            prefix_length = input_emb.shape[1] - label_tokens.shape[1]
            prefix_labels = torch.full(
                (1, prefix_length), -100, dtype=torch.long
            ).to(chat.device)
            labels = torch.cat([prefix_labels, label_tokens], dim=1)

            # Forward pass
            results = chat.model.llama_model(inputs_embeds=input_emb, labels=labels)
            loss = results.loss.item()

            # Handle NaN: treat as infinite loss (worst possible)
            if np.isnan(loss) or np.isinf(loss):
                conf_list.append(float('inf'))
            else:
                conf_list.append(loss)

    # Remove the assistant message we appended
    conv.messages.pop()

    return conf_list


def classify_gene_ppl(chat, seq, question_prompt, task_name, task_def):
    """
    Classify a gene by scoring each candidate answer using perplexity.
    Uses training-style short answers as candidates, then maps to canonical categories.
    Lower loss = higher confidence = predicted class.
    """
    ppl_candidates = task_def["ppl_candidates"]
    ppl_candidate_map = task_def.get("ppl_candidate_map")
    categories = task_def["categories"]

    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_gene(seq, chat_state, img_list)
    chat.ask(question_prompt, chat_state)

    # Use fixed independent PPL scoring (original get_ppl has accumulation bug)
    losses = get_ppl_independent(chat, chat_state, img_list, ppl_candidates)

    predicted_idx = int(np.argmin(losses))
    raw_prediction = ppl_candidates[predicted_idx]

    # Map to canonical category
    if task_name == "exon_count":
        # Bin the predicted number
        predicted = normalize_exon_count(raw_prediction)
        if predicted is None:
            predicted = categories[0]  # fallback
    elif ppl_candidate_map is not None:
        predicted = ppl_candidate_map.get(raw_prediction, raw_prediction)
    else:
        predicted = raw_prediction

    return predicted, raw_prediction, losses


def _match_answer_to_category(raw_answer, task_name, task_def):
    """
    Try to match a generated answer string to a canonical category.
    Returns (predicted_category, matched) or (None, False).
    """
    categories = task_def["categories"]
    answer_aliases = task_def.get("answer_aliases", {})

    # Clean up the raw answer: remove trailing period, quotes, whitespace
    raw_clean = raw_answer.strip().rstrip(".").strip().strip("'\"").strip()
    raw_lower = raw_clean.lower()

    # 1. Exact match in answer_aliases
    if raw_lower in answer_aliases:
        return answer_aliases[raw_lower], True

    # 2. Direct match against categories (case-insensitive)
    categories_lower = {c.lower(): c for c in categories}
    if raw_lower in categories_lower:
        return categories_lower[raw_lower], True

    # 3. Task-specific normalization
    if task_name == "organism":
        result = normalize_organism(raw_clean)
        if result:
            return result, True
    elif task_name == "chromosome":
        norm = normalize_chromosome(raw_clean)
        if norm and norm in categories:
            return norm, True
    elif task_name == "exon_count":
        result = normalize_exon_count(raw_clean)
        if result:
            return result, True
    elif task_name == "locus_type":
        # Handle common generated patterns
        locus_map = {
            "intronless": "protein-coding",  # intronless genes are protein-coding
            "protein coding": "protein-coding",
            "pseudogene": "pseudo",
            "non-coding": "ncRNA", "noncoding": "ncRNA",
            "transfer rna": "tRNA", "ribosomal rna": "rRNA",
            "small nucleolar": "snoRNA", "small nuclear": "snRNA",
        }
        for key, val in locus_map.items():
            if key in raw_lower:
                return val, True

    # 4. Check if any category is contained in the answer
    for cat in categories:
        if cat.lower() in raw_lower:
            return cat, True

    # 5. Check if any alias key is contained in the answer
    for alias_key, alias_val in answer_aliases.items():
        if alias_key in raw_lower:
            return alias_val, True

    # 6. Handle quoted answers like "The locus type is 'pseudogene'"
    import re
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", raw_answer)
    for q in quoted:
        result, matched = _match_answer_to_category(q, task_name, task_def)
        if matched:
            return result, True

    return None, False


def classify_gene_generate(chat, seq, question_prompt, task_name, task_def):
    """
    Classify a gene by generating a short answer and matching to categories.
    This leverages the model's training on short-answer keyword QA format.
    """
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_gene(seq, chat_state, img_list)
    chat.ask(question_prompt, chat_state)

    # Generate short answer (low temperature for determinism)
    llm_message, _, loss = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=4,
        temperature=1e-3,
        max_new_tokens=32,  # short answer expected
        max_length=512,
    )

    raw_answer = llm_message.strip()
    predicted, _ = _match_answer_to_category(raw_answer, task_name, task_def)

    return predicted, raw_answer, loss


# ========================================
#         LLM API Baselines (Claude, GPT-4)
# ========================================

def _build_classification_prompt(task_name, task_def, seq_text=None, gene_id=None, use_genename=False):
    """Build a classification prompt for LLM API baselines."""
    categories = task_def["categories"]
    categories_str = ", ".join(categories)

    task_questions = {
        "organism": "Which organism does this gene belong to?",
        "locus_type": "What is the locus type of this gene?",
        "chromosome": "On which chromosome is this gene located?",
        "exon_count": "How many exons does this gene contain?",
    }
    question = task_questions[task_name]

    if use_genename:
        gene_info = f"Gene ID: {gene_id}"
    else:
        gene_info = f"DNA Sequence:\n{seq_text}"

    prompt = (
        f"You are a geneticist. Given the following gene information, answer the question by "
        f"selecting EXACTLY ONE option from the list below. Reply with ONLY the chosen option, "
        f"nothing else.\n\n"
        f"{gene_info}\n\n"
        f"Question: {question}\n"
        f"Options: {categories_str}\n\n"
        f"Answer:"
    )
    return prompt


def _parse_llm_response(response_text, categories):
    """Parse LLM response to match one of the valid categories."""
    response_clean = response_text.strip().strip(".").strip()
    categories_lower = {c.lower(): c for c in categories}

    # Exact match (case-insensitive)
    if response_clean.lower() in categories_lower:
        return categories_lower[response_clean.lower()]

    # Check if any category is contained in the response
    for cat_lower, cat in categories_lower.items():
        if cat_lower in response_clean.lower():
            return cat

    return None  # Could not parse


def classify_claude(api_key, model_name, task_name, task_def, task_data, seqs,
                    max_samples=None, use_genename=False, seq_length=150000):
    """Run classification using Claude API."""
    try:
        import anthropic
    except ImportError:
        print("Install: pip install anthropic")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    categories = task_def["categories"]

    if max_samples and len(task_data) > max_samples:
        random.shuffle(task_data)
        task_data = task_data[:max_samples]

    y_true, y_pred = [], []
    results_detail = []
    method_label = f"Claude ({model_name}, {'genename' if use_genename else 'seq'})"

    print(f"\nEvaluating: {method_label} on '{task_name}' ({len(task_data)} samples)")
    start_time = time.time()

    for idx, item in enumerate(task_data):
        gene_id = item["gene_id"]
        ground_truth = item["ground_truth"]

        seq_text = None
        if not use_genename:
            if gene_id not in seqs:
                continue
            seq = seqs[gene_id]
            if isinstance(seq, list):
                seq = seq[0]
            seq_text = seq[:seq_length] if len(seq) > seq_length else seq

        prompt = _build_classification_prompt(
            task_name, task_def, seq_text=seq_text,
            gene_id=gene_id, use_genename=use_genename
        )

        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text if message.content else ""
        except Exception as e:
            print(f"  Claude error for gene {gene_id}: {e}")
            response_text = ""

        predicted = _parse_llm_response(response_text, categories)

        y_true.append(ground_truth)
        y_pred.append(predicted if predicted else "UNPARSEABLE")

        results_detail.append({
            "gene_id": gene_id,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_response": response_text,
            "correct": ground_truth == predicted,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            valid = [yt for yt, yp in zip(y_true, y_pred) if yp != "UNPARSEABLE"]
            acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(len(y_true), 1)
            print(f"  [{idx+1}/{len(task_data)}] acc={acc:.3f} elapsed={elapsed:.1f}s")

        time.sleep(0.3)  # Rate limiting

    return _compute_results(task_name, y_true, y_pred, results_detail, start_time, method_label)


def classify_gpt4(api_key, model_name, task_name, task_def, task_data, seqs,
                   max_samples=None, use_genename=False, seq_length=6000):
    """Run classification using GPT-4 API."""
    try:
        import openai
    except ImportError:
        print("Install: pip install openai")
        return None

    client = openai.OpenAI(api_key=api_key)
    categories = task_def["categories"]

    if max_samples and len(task_data) > max_samples:
        random.shuffle(task_data)
        task_data = task_data[:max_samples]

    y_true, y_pred = [], []
    results_detail = []
    method_label = f"GPT-4 ({model_name}, {'genename' if use_genename else 'seq'})"

    print(f"\nEvaluating: {method_label} on '{task_name}' ({len(task_data)} samples)")
    start_time = time.time()

    for idx, item in enumerate(task_data):
        gene_id = item["gene_id"]
        ground_truth = item["ground_truth"]

        seq_text = None
        if not use_genename:
            if gene_id not in seqs:
                continue
            seq = seqs[gene_id]
            if isinstance(seq, list):
                seq = seq[0]
            seq_text = seq[:seq_length] if len(seq) > seq_length else seq

        prompt = _build_classification_prompt(
            task_name, task_def, seq_text=seq_text,
            gene_id=gene_id, use_genename=use_genename
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  GPT-4 error for gene {gene_id}: {e}")
            response_text = ""

        predicted = _parse_llm_response(response_text, categories)

        y_true.append(ground_truth)
        y_pred.append(predicted if predicted else "UNPARSEABLE")

        results_detail.append({
            "gene_id": gene_id,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_response": response_text,
            "correct": ground_truth == predicted,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(len(y_true), 1)
            print(f"  [{idx+1}/{len(task_data)}] acc={acc:.3f} elapsed={elapsed:.1f}s")

        time.sleep(0.3)

    return _compute_results(task_name, y_true, y_pred, results_detail, start_time, method_label)


def classify_gemini(api_key, model_name, task_name, task_def, task_data, seqs,
                     max_samples=None, use_genename=False, seq_length=30000):
    """Run classification using Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("Install: pip install google-generativeai")
        return None

    genai.configure(api_key=api_key)
    # Disable safety filters — gene names/sequences can trigger false positives
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
    categories = task_def["categories"]

    if max_samples and len(task_data) > max_samples:
        random.shuffle(task_data)
        task_data = task_data[:max_samples]

    y_true, y_pred = [], []
    results_detail = []
    method_label = f"Gemini ({model_name}, {'genename' if use_genename else 'seq'})"

    print(f"\nEvaluating: {method_label} on '{task_name}' ({len(task_data)} samples)")
    start_time = time.time()

    for idx, item in enumerate(task_data):
        gene_id = item["gene_id"]
        ground_truth = item["ground_truth"]

        seq_text = None
        if not use_genename:
            if gene_id not in seqs:
                continue
            seq = seqs[gene_id]
            if isinstance(seq, list):
                seq = seq[0]
            seq_text = seq[:seq_length] if len(seq) > seq_length else seq

        prompt = _build_classification_prompt(
            task_name, task_def, seq_text=seq_text,
            gene_id=gene_id, use_genename=use_genename
        )

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=64,
                    temperature=0.0,
                ),
            )
            # Try to extract text safely — handle safety blocks
            response_text = ""
            try:
                response_text = response.text.strip()
            except (ValueError, AttributeError):
                # Safety block or empty response — try to get partial text from candidates
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text = part.text.strip()
                            break
        except Exception as e:
            if idx < 5:
                print(f"  Gemini error for gene {gene_id}: {e}")
            response_text = ""

        # Debug: print first 5 raw responses to check what Gemini returns
        if idx < 5:
            print(f"    DEBUG gene={gene_id} raw='{response_text[:80]}' truth={ground_truth}")

        predicted = _parse_llm_response(response_text, categories)

        y_true.append(ground_truth)
        y_pred.append(predicted if predicted else "UNPARSEABLE")

        results_detail.append({
            "gene_id": gene_id,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_response": response_text,
            "correct": ground_truth == predicted,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(len(y_true), 1)
            print(f"  [{idx+1}/{len(task_data)}] acc={acc:.3f} elapsed={elapsed:.1f}s")

        time.sleep(0.2)  # Rate limiting

    return _compute_results(task_name, y_true, y_pred, results_detail, start_time, method_label)


def classify_random(task_name, task_def, task_data, max_samples=None):
    """Random baseline for classification."""
    categories = task_def["categories"]

    if max_samples and len(task_data) > max_samples:
        random.shuffle(task_data)
        task_data = task_data[:max_samples]

    y_true = [item["ground_truth"] for item in task_data]
    y_pred = [random.choice(categories) for _ in task_data]
    results_detail = [
        {"gene_id": item["gene_id"], "ground_truth": item["ground_truth"],
         "predicted": pred, "correct": item["ground_truth"] == pred}
        for item, pred in zip(task_data, y_pred)
    ]

    return _compute_results(task_name, y_true, y_pred, results_detail, time.time(), "Random")


def _compute_results(task_name, y_true, y_pred, results_detail, start_time, method_label):
    """Compute classification metrics from predictions."""
    elapsed = time.time() - start_time

    # Filter out unparseable predictions for metric computation
    valid_mask = [yp != "UNPARSEABLE" for yp in y_pred]
    y_true_valid = [yt for yt, m in zip(y_true, valid_mask) if m]
    y_pred_valid = [yp for yp, m in zip(y_pred, valid_mask) if m]
    n_unparseable = sum(1 for m in valid_mask if not m)

    if not y_true_valid:
        print(f"  No valid predictions for {method_label}")
        return None

    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)

    print(f"\n  Results for '{task_name}' ({method_label}):")
    print(f"    Accuracy:    {accuracy:.4f}")
    print(f"    Macro F1:    {macro_f1:.4f}")
    print(f"    Weighted F1: {weighted_f1:.4f}")
    if n_unparseable > 0:
        print(f"    Unparseable: {n_unparseable}/{len(y_pred)} ({100*n_unparseable/len(y_pred):.1f}%)")
    print(f"    Time: {elapsed:.1f}s")

    report = classification_report(y_true_valid, y_pred_valid, zero_division=0)
    print(f"\n  Classification Report:\n{report}")

    return {
        "task": task_name,
        "method": method_label,
        "num_samples": len(y_true),
        "num_valid": len(y_true_valid),
        "num_unparseable": n_unparseable,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "elapsed_seconds": elapsed,
        "per_sample": results_detail,
        "classification_report": classification_report(y_true_valid, y_pred_valid, output_dict=True, zero_division=0),
    }


# ========================================
#         Data Preparation
# ========================================

def prepare_task_data(qa_list, task_name, task_def):
    """Filter and normalize QA entries for a specific classification task."""
    q_id = task_def["q_id"]
    categories = task_def["categories"]
    categories_lower = [c.lower() for c in categories]
    normalizer = NORMALIZERS[task_name]

    filtered = []
    skipped = 0
    for item in qa_list:
        if item["Q_Id"] != q_id:
            continue
        raw_answer = item["A"]

        # Normalize the ground truth answer
        normalized = normalizer(raw_answer)
        if normalized is None:
            skipped += 1
            continue

        # Match to category list (case-insensitive)
        norm_lower = normalized.lower()
        if norm_lower not in categories_lower:
            skipped += 1
            continue

        # Find the canonical category name
        canonical = categories[categories_lower.index(norm_lower)]
        filtered.append({
            "gene_id": item["Gene Id"],
            "raw_answer": raw_answer,
            "ground_truth": canonical,
        })

    print(f"  Task '{task_name}': {len(filtered)} samples (skipped {skipped})")

    # Show class distribution
    dist = Counter(item["ground_truth"] for item in filtered)
    for cat in categories:
        print(f"    {cat}: {dist.get(cat, 0)}")

    return filtered


# ========================================
#         Evaluation
# ========================================

def evaluate_task_genechat(chat, seqs, task_name, task_def, task_data,
                           max_samples=None, classify_mode="ppl"):
    """
    Run GeneChat classification on a task and compute metrics.

    Args:
        classify_mode: "ppl" for perplexity-based scoring, "generate" for generation-based
    """
    prompt = task_def["prompt"]
    categories = task_def["categories"]

    if max_samples and len(task_data) > max_samples:
        random.shuffle(task_data)
        task_data = task_data[:max_samples]

    y_true = []
    y_pred = []
    results_detail = []

    mode_label = "PPL" if classify_mode == "ppl" else "Generate"
    print(f"\nEvaluating: GeneChat ({mode_label}) on '{task_name}' ({len(task_data)} samples)")
    start_time = time.time()

    for idx, item in enumerate(task_data):
        gene_id = item["gene_id"]
        ground_truth = item["ground_truth"]

        if gene_id not in seqs:
            print(f"  Warning: gene {gene_id} not found in sequences, skipping")
            continue

        seq = seqs[gene_id]
        if isinstance(seq, list):
            seq = seq[0]
        if len(seq) > 160000:
            seq = seq[:159999]

        try:
            if classify_mode == "ppl":
                predicted, raw_prediction, losses = classify_gene_ppl(
                    chat, seq, prompt, task_name, task_def
                )
                detail = {
                    "gene_id": gene_id,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "raw_prediction": raw_prediction,
                    "correct": ground_truth == predicted,
                    "losses": {cand: float(loss) for cand, loss in
                               zip(task_def["ppl_candidates"], losses)},
                }
            else:  # generate
                predicted, raw_answer, loss = classify_gene_generate(
                    chat, seq, prompt, task_name, task_def
                )
                if predicted is None:
                    predicted = "UNPARSEABLE"
                detail = {
                    "gene_id": gene_id,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "raw_generated": raw_answer,
                    "correct": ground_truth == predicted,
                }

            y_true.append(ground_truth)
            y_pred.append(predicted)
            results_detail.append(detail)

        except Exception as e:
            print(f"  Error processing gene {gene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (idx + 1) % 25 == 0 or idx == 0:
            elapsed = time.time() - start_time
            acc_so_far = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / max(len(y_true), 1)
            print(f"  [{idx+1}/{len(task_data)}] acc={acc_so_far:.3f} elapsed={elapsed:.1f}s")

    method_label = f"GeneChat ({mode_label})"
    return _compute_results(task_name, y_true, y_pred, results_detail, start_time, method_label)


# ========================================
#         Main
# ========================================

AVAILABLE_METHODS = [
    "genechat", "genechat-generate",
    "claude-seq", "claude-genename",
    "gpt4-seq", "gpt4-genename",
    "gemini-seq", "gemini-genename",
    "random",
]

CLAUDE_MODELS = [
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

GPT4_MODELS = [
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-3-pro",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Gene Classification Evaluation")
    parser.add_argument("--method", default="genechat", choices=AVAILABLE_METHODS,
                        help="Classification method to use (default: genechat)")
    parser.add_argument("--cfg-path", default="configs/genechat_eval.yaml",
                        help="Path to GeneChat evaluation config (only for method=genechat)")
    parser.add_argument("--test-qa", default="test_data/qa_kw.json",
                        help="Path to qa_kw.json")
    parser.add_argument("--test-seq", default="test_data/seq.json",
                        help="Path to seq.json")
    parser.add_argument("--output", default="gene_classification_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per task (for quick testing)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=list(CLASSIFICATION_TASKS.keys()),
                        help="Subset of tasks to run (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--options", nargs="+",
                        help="Override config settings (key=value)")
    # API keys
    parser.add_argument("--claude-api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--claude-model", default="claude-sonnet-4-5-20250929",
                        choices=CLAUDE_MODELS,
                        help="Claude model to use")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--gpt4-model", default="gpt-4-turbo",
                        choices=GPT4_MODELS,
                        help="GPT-4 model to use")
    parser.add_argument("--gemini-api-key", default=os.environ.get("GOOGLE_API_KEY"),
                        help="Google API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash",
                        choices=GEMINI_MODELS,
                        help="Gemini model to use")
    parser.add_argument("--seq-length", type=int, default=None,
                        help="Max DNA sequence length for API baselines (default: 150K for Claude, 30K for Gemini, 6K for GPT-4)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    method = args.method

    # Load GeneChat model only if needed
    chat = None
    if method in ("genechat", "genechat-generate"):
        Config, registry, init_distributed_mode, Chat, CONV_VISION = _import_genechat()
        torch.manual_seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        print("=" * 60)
        print("Loading GeneChat model...")
        print("=" * 60)
        cfg = Config(args)
        init_distributed_mode(cfg.run_cfg)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        device = f"cuda:{args.gpu_id}"
        model = model_cls.from_config(model_config).to(device)
        chat = Chat(model, device=device)
        print("Model loaded.\n")

    # Load data
    print("Loading data...")
    with open(args.test_qa, "r") as f:
        qa_list = json.load(f)

    seqs = {}
    needs_seq = method in ("genechat", "genechat-generate", "claude-seq", "gpt4-seq", "gemini-seq")
    if needs_seq:
        with open(args.test_seq, "r") as f:
            seqs = json.load(f)
        print(f"  Gene sequences: {len(seqs)}")
    print(f"  QA entries: {len(qa_list)}")

    # Determine tasks to run
    tasks_to_run = args.tasks or list(CLASSIFICATION_TASKS.keys())

    print(f"\n{'=' * 60}")
    print(f"Method: {method}")
    if method.startswith("claude"):
        print(f"Model:  {args.claude_model}")
    elif method.startswith("gpt4"):
        print(f"Model:  {args.gpt4_model}")
    elif method.startswith("gemini"):
        print(f"Model:  {args.gemini_model}")
    print(f"Tasks:  {', '.join(tasks_to_run)}")
    print(f"{'=' * 60}")

    # Prepare and evaluate each task
    all_results = {}
    summary = []

    for task_name in tasks_to_run:
        task_def = CLASSIFICATION_TASKS[task_name]
        print(f"\n{'=' * 60}")
        print(f"Preparing task: {task_name}")
        print(f"{'=' * 60}")

        task_data = prepare_task_data(qa_list, task_name, task_def)
        if not task_data:
            print(f"  No valid data for task '{task_name}', skipping.")
            continue

        result = None

        if method == "genechat":
            result = evaluate_task_genechat(chat, seqs, task_name, task_def, task_data,
                                            args.max_samples, classify_mode="ppl")

        elif method == "genechat-generate":
            result = evaluate_task_genechat(chat, seqs, task_name, task_def, task_data,
                                            args.max_samples, classify_mode="generate")

        elif method == "claude-seq":
            if not args.claude_api_key:
                print("  Error: --claude-api-key required for claude-seq")
                continue
            seq_len = args.seq_length or 150000
            result = classify_claude(
                args.claude_api_key, args.claude_model,
                task_name, task_def, task_data, seqs,
                args.max_samples, use_genename=False, seq_length=seq_len
            )

        elif method == "claude-genename":
            if not args.claude_api_key:
                print("  Error: --claude-api-key required for claude-genename")
                continue
            result = classify_claude(
                args.claude_api_key, args.claude_model,
                task_name, task_def, task_data, seqs,
                args.max_samples, use_genename=True
            )

        elif method == "gpt4-seq":
            if not args.openai_api_key:
                print("  Error: --openai-api-key required for gpt4-seq")
                continue
            seq_len = args.seq_length or 6000
            result = classify_gpt4(
                args.openai_api_key, args.gpt4_model,
                task_name, task_def, task_data, seqs,
                args.max_samples, use_genename=False, seq_length=seq_len
            )

        elif method == "gpt4-genename":
            if not args.openai_api_key:
                print("  Error: --openai-api-key required for gpt4-genename")
                continue
            result = classify_gpt4(
                args.openai_api_key, args.gpt4_model,
                task_name, task_def, task_data, seqs,
                args.max_samples, use_genename=True
            )

        elif method == "gemini-seq":
            if not args.gemini_api_key:
                print("  Error: --gemini-api-key required for gemini-seq")
                continue
            seq_len = args.seq_length or 30000
            result = classify_gemini(
                args.gemini_api_key, args.gemini_model,
                task_name, task_def, task_data, seqs,
                args.max_samples, use_genename=False, seq_length=seq_len
            )

        elif method == "gemini-genename":
            if not args.gemini_api_key:
                print("  Error: --gemini-api-key required for gemini-genename")
                continue
            result = classify_gemini(
                args.gemini_api_key, args.gemini_model,
                task_name, task_def, task_data, seqs,
                args.max_samples, use_genename=True
            )

        elif method == "random":
            result = classify_random(task_name, task_def, task_data, args.max_samples)

        if result:
            all_results[task_name] = result
            summary.append({
                "task": task_name,
                "method": result.get("method", method),
                "num_samples": result["num_samples"],
                "accuracy": round(result["accuracy"], 4),
                "macro_f1": round(result["macro_f1"], 4),
                "weighted_f1": round(result["weighted_f1"], 4),
            })

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY  (method: {method})")
    print(f"{'=' * 70}")
    print(f"{'Task':<20} {'N':>6} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 70)
    for s in summary:
        print(f"{s['task']:<20} {s['num_samples']:>6} {s['accuracy']:>10.4f} {s['macro_f1']:>10.4f} {s['weighted_f1']:>12.4f}")

    # Save results
    output = {
        "method": method,
        "model": args.claude_model if method.startswith("claude") else
                 args.gpt4_model if method.startswith("gpt4") else
                 args.gemini_model if method.startswith("gemini") else method,
        "summary": summary,
        "tasks": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                  for k, v in all_results.items()},
        "detailed_results": {k: v["per_sample"] for k, v in all_results.items()},
        "config": {
            "method": method,
            "cfg_path": args.cfg_path if method == "genechat" else None,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
