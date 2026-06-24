#!/usr/bin/env python
"""
Semantic Similarity Evaluation for Gene Description Generation.

This script evaluates gene descriptions using semantic similarity metrics:
1. BERTScore - Uses BERT embeddings for precision/recall/F1
2. SimCSE - Sentence embeddings cosine similarity
3. Also includes BLEU and METEOR for comparison

Supports evaluation of:
- GeneChat (with different encoders)
- Claude (Sonnet, Opus)
- ChatGPT (GPT-4, GPT-4o)
- LLaMA baselines
- Other baselines

Usage:
    # Evaluate a single results file
    python eval_semantic_similarity.py --results-file results/results_genechat.json --output-dir results/semantic

    # Compare multiple models
    python eval_semantic_similarity.py --compare-dir results/ --output-dir results/semantic_comparison

    # Evaluate with specific metrics only
    python eval_semantic_similarity.py --results-file results.json --metrics bertscore simcse
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# NLTK for BLEU/METEOR
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


# Model configurations with max sequence lengths
MODEL_CONFIGS = {
    'genechat': {
        'name': 'GeneChat (DNABERT-2)',
        'max_seq_length': 160000,
        'type': 'genechat'
    },
    'genechat_dnabert2': {
        'name': 'GeneChat + DNABERT-2',
        'max_seq_length': 160000,
        'type': 'genechat'
    },
    'genechat_hyenadna': {
        'name': 'GeneChat + HyenaDNA',
        'max_seq_length': 160000,
        'type': 'genechat'
    },
    'genechat_nucleotide_transformer': {
        'name': 'GeneChat + Nucleotide Transformer',
        'max_seq_length': 6000,
        'type': 'genechat'
    },
    'claude_sonnet': {
        'name': 'Claude Sonnet 4',
        'max_seq_length': 200000,  # ~200k tokens context
        'type': 'api'
    },
    'claude_opus': {
        'name': 'Claude Opus 4',
        'max_seq_length': 200000,
        'type': 'api'
    },
    'gpt4': {
        'name': 'GPT-4',
        'max_seq_length': 128000,  # 128k context
        'type': 'api'
    },
    'gpt4o': {
        'name': 'GPT-4o',
        'max_seq_length': 128000,
        'type': 'api'
    },
    'llama_zeroshot': {
        'name': 'LLaMA Zero-shot',
        'max_seq_length': 4096,
        'type': 'baseline'
    },
    'llama_fewshot': {
        'name': 'LLaMA Few-shot',
        'max_seq_length': 4096,
        'type': 'baseline'
    },
    'no_encoder': {
        'name': 'No Encoder (Raw Seq)',
        'max_seq_length': 2000,
        'type': 'baseline'
    },
    'hyenadna': {
        'name': 'GeneChat + HyenaDNA',
        'max_seq_length': 160000,
        'type': 'genechat'
    },
    'dnabert_s': {
        'name': 'GeneChat + DNABERT-S',
        'max_seq_length': 512,
        'type': 'genechat'
    },
    'dnabert': {
        'name': 'GeneChat + DNABERT (6-mer)',
        'max_seq_length': 512,
        'type': 'genechat'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic similarity evaluation for gene descriptions')
    parser.add_argument('--results-file', type=str, default=None,
                        help='Path to a single results JSON file')
    parser.add_argument('--compare-dir', type=str, default=None,
                        help='Directory containing multiple result files to compare')
    parser.add_argument('--output-dir', type=str, default='results/semantic_eval')
    parser.add_argument('--metrics', nargs='+', default=['bleu', 'meteor', 'bertscore', 'simcse'],
                        choices=['bleu', 'meteor', 'bertscore', 'simcse'],
                        help='Metrics to compute')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for BERTScore/SimCSE computation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--filter-empty', action='store_true', default=True,
                        help='Filter out samples with empty ground truth')
    parser.add_argument('--bertscore-model', type=str, default='microsoft/deberta-xlarge-mnli',
                        help='Model for BERTScore')
    parser.add_argument('--simcse-model', type=str, default='princeton-nlp/sup-simcse-roberta-large',
                        help='Model for SimCSE')
    return parser.parse_args()


# ============== BLEU & METEOR ==============

def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict:
    """Compute BLEU-1,2,3,4 scores."""
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    weights = {
        1: (1.0, 0, 0, 0),
        2: (0.5, 0.5, 0, 0),
        3: (1/3, 1/3, 1/3, 0),
        4: (0.25, 0.25, 0.25, 0.25)
    }
    smoothing = SmoothingFunction().method1

    for ref, hyp in zip(references, hypotheses):
        try:
            ref_tokens = word_tokenize(ref.lower())
            hyp_tokens = word_tokenize(hyp.lower())
        except:
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()

        if len(hyp_tokens) == 0:
            hyp_tokens = ['<empty>']

        for n in [1, 2, 3, 4]:
            try:
                score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights[n], smoothing_function=smoothing)
                bleu_scores[n].append(score)
            except:
                bleu_scores[n].append(0.0)

    return {
        'bleu_1': float(np.mean(bleu_scores[1])),
        'bleu_2': float(np.mean(bleu_scores[2])),
        'bleu_3': float(np.mean(bleu_scores[3])),
        'bleu_4': float(np.mean(bleu_scores[4])),
        'bleu_1_std': float(np.std(bleu_scores[1])),
        'bleu_2_std': float(np.std(bleu_scores[2])),
        'bleu_3_std': float(np.std(bleu_scores[3])),
        'bleu_4_std': float(np.std(bleu_scores[4])),
    }


def compute_meteor(references: List[str], hypotheses: List[str]) -> Dict:
    """Compute METEOR score."""
    meteor_scores = []

    for ref, hyp in zip(references, hypotheses):
        try:
            ref_tokens = word_tokenize(ref.lower())
            hyp_tokens = word_tokenize(hyp.lower())
        except:
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()

        if len(hyp_tokens) == 0:
            hyp_tokens = ['<empty>']

        try:
            score = meteor_score([ref_tokens], hyp_tokens)
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)

    return {
        'meteor': float(np.mean(meteor_scores)),
        'meteor_std': float(np.std(meteor_scores)),
    }


# ============== BERTScore ==============

def compute_bertscore(references: List[str], hypotheses: List[str],
                      model_type: str = 'microsoft/deberta-xlarge-mnli',
                      device: str = 'cuda:0', batch_size: int = 32) -> Dict:
    """Compute BERTScore (Precision, Recall, F1)."""
    try:
        from bert_score import score as bert_score
    except ImportError:
        print("Error: bert-score not installed. Run: pip install bert-score")
        return {'bertscore_p': 0, 'bertscore_r': 0, 'bertscore_f1': 0}

    print(f"Computing BERTScore with {model_type}...")

    # Filter empty strings
    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip() and h.strip()]
    if not valid_pairs:
        return {'bertscore_p': 0, 'bertscore_r': 0, 'bertscore_f1': 0}

    refs, hyps = zip(*valid_pairs)

    P, R, F1 = bert_score(
        cands=list(hyps),
        refs=list(refs),
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=True
    )

    return {
        'bertscore_p': float(P.mean()),
        'bertscore_r': float(R.mean()),
        'bertscore_f1': float(F1.mean()),
        'bertscore_p_std': float(P.std()),
        'bertscore_r_std': float(R.std()),
        'bertscore_f1_std': float(F1.std()),
    }


# ============== SimCSE ==============

def compute_simcse(references: List[str], hypotheses: List[str],
                   model_name: str = 'princeton-nlp/sup-simcse-roberta-large',
                   device: str = 'cuda:0', batch_size: int = 32) -> Dict:
    """Compute SimCSE cosine similarity."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("Error: transformers not installed")
        return {'simcse': 0}

    print(f"Computing SimCSE with {model_name}...")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def get_embeddings(texts: List[str]) -> torch.Tensor:
        """Get sentence embeddings using mean pooling."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    # Filter empty strings
    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip() and h.strip()]
    if not valid_pairs:
        return {'simcse': 0, 'simcse_std': 0}

    refs, hyps = zip(*valid_pairs)

    # Get embeddings
    ref_embeddings = get_embeddings(list(refs))
    hyp_embeddings = get_embeddings(list(hyps))

    # Compute cosine similarity
    ref_norm = ref_embeddings / ref_embeddings.norm(dim=1, keepdim=True)
    hyp_norm = hyp_embeddings / hyp_embeddings.norm(dim=1, keepdim=True)

    # Element-wise cosine similarity (each ref with its corresponding hyp)
    similarities = (ref_norm * hyp_norm).sum(dim=1)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        'simcse': float(similarities.mean()),
        'simcse_std': float(similarities.std()),
        'simcse_min': float(similarities.min()),
        'simcse_max': float(similarities.max()),
    }


# ============== Main Evaluation ==============

def load_results(file_path: str) -> Tuple[List[Dict], str]:
    """Load results from JSON file and detect model type."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        predictions = data
    elif isinstance(data, dict):
        if 'predictions' in data:
            predictions = data['predictions']
        elif 'results' in data:
            predictions = data['results']
        else:
            # Assume it's a flat list stored as dict values
            predictions = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []

    # Detect model type from filename or data
    filename = os.path.basename(file_path).lower()
    model_type = 'unknown'

    if 'claude' in filename:
        if 'opus' in filename:
            model_type = 'claude_opus'
        else:
            model_type = 'claude_sonnet'
    elif 'gpt' in filename:
        if '4o' in filename:
            model_type = 'gpt4o'
        else:
            model_type = 'gpt4'
    elif 'llama' in filename:
        if 'zeroshot' in filename or 'zero' in filename:
            model_type = 'llama_zeroshot'
        else:
            model_type = 'llama_fewshot'
    elif 'genechat' in filename or 'dnabert' in filename:
        if 'hyena' in filename:
            model_type = 'genechat_hyenadna'
        elif 'nucleotide' in filename:
            model_type = 'genechat_nucleotide_transformer'
        elif 'dnabert_s' in filename:
            model_type = 'dnabert_s'
        elif 'dnabert2' in filename or 'dnabert-2' in filename:
            model_type = 'genechat_dnabert2'
        elif 'dnabert' in filename and 'dnabert2' not in filename and 'dnabert_s' not in filename:
            # Original DNABERT (results_dnabert.json)
            model_type = 'dnabert'
        else:
            model_type = 'genechat_dnabert2'
    elif 'hyena' in filename:
        # HyenaDNA encoder (standalone)
        model_type = 'genechat_hyenadna'
    elif 'dnabert_s' in filename:
        # DNABERT-S encoder (standalone)
        model_type = 'dnabert_s'
    elif 'no_encoder' in filename:
        model_type = 'no_encoder'

    return predictions, model_type


def extract_texts(predictions: List[Dict], filter_empty: bool = True) -> Tuple[List[str], List[str], List[str]]:
    """Extract reference and hypothesis texts from predictions."""
    references = []
    hypotheses = []
    gene_ids = []

    for pred in predictions:
        # Handle different key names
        ref = pred.get('correct_func', pred.get('ground_truth', pred.get('reference', '')))
        hyp = pred.get('predict_func', pred.get('prediction', pred.get('hypothesis', '')))
        gene_id = pred.get('gene_id', pred.get('Gene Id', 'unknown'))

        # Filter empty
        if filter_empty and (not ref or not ref.strip()):
            continue

        references.append(ref)
        hypotheses.append(hyp if hyp else '<empty>')
        gene_ids.append(gene_id)

    return references, hypotheses, gene_ids


def evaluate_single_file(file_path: str, args) -> Dict:
    """Evaluate a single results file."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {file_path}")
    print(f"{'='*70}")

    predictions, model_type = load_results(file_path)
    references, hypotheses, gene_ids = extract_texts(predictions, args.filter_empty)

    if args.max_samples:
        references = references[:args.max_samples]
        hypotheses = hypotheses[:args.max_samples]
        gene_ids = gene_ids[:args.max_samples]

    print(f"Model type: {model_type}")
    print(f"Samples: {len(references)} (after filtering)")

    if len(references) == 0:
        print("WARNING: No valid samples found!")
        return {'error': 'No valid samples'}

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    results = {
        'file': file_path,
        'model_type': model_type,
        'model_name': MODEL_CONFIGS.get(model_type, {}).get('name', model_type),
        'num_samples': len(references),
    }

    # Compute metrics
    if 'bleu' in args.metrics:
        print("\nComputing BLEU...")
        bleu_results = compute_bleu(references, hypotheses)
        results.update(bleu_results)

    if 'meteor' in args.metrics:
        print("Computing METEOR...")
        meteor_results = compute_meteor(references, hypotheses)
        results.update(meteor_results)

    if 'bertscore' in args.metrics:
        bertscore_results = compute_bertscore(
            references, hypotheses,
            model_type=args.bertscore_model,
            device=device,
            batch_size=args.batch_size
        )
        results.update(bertscore_results)

    if 'simcse' in args.metrics:
        simcse_results = compute_simcse(
            references, hypotheses,
            model_name=args.simcse_model,
            device=device,
            batch_size=args.batch_size
        )
        results.update(simcse_results)

    return results


def compare_multiple_files(dir_path: str, args) -> Dict[str, Dict]:
    """Compare multiple result files in a directory."""
    all_results = {}

    # Find all JSON files
    json_files = []
    for f in os.listdir(dir_path):
        if f.endswith('.json') and 'summary' not in f.lower():
            json_files.append(os.path.join(dir_path, f))

    print(f"Found {len(json_files)} result files to compare")

    for file_path in sorted(json_files):
        try:
            results = evaluate_single_file(file_path, args)
            model_name = results.get('model_name', os.path.basename(file_path))
            all_results[model_name] = results
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    return all_results


def print_results_table(results: Dict):
    """Print formatted results table."""
    print("\n" + "="*120)
    print("SEMANTIC SIMILARITY EVALUATION RESULTS")
    print("="*120)

    # Header
    header = f"{'Model':<35} | {'BLEU-1':>8} | {'METEOR':>8} | {'BERTScore F1':>12} | {'SimCSE':>8} | {'Samples':>8}"
    print(header)
    print("-"*120)

    if isinstance(results, dict) and 'model_name' in results:
        # Single result
        results = {results['model_name']: results}

    for model_name, scores in results.items():
        # Skip non-dict entries (e.g., metadata)
        if not isinstance(scores, dict):
            continue
        bleu1 = scores.get('bleu_1', 0)
        meteor = scores.get('meteor', 0)
        bertscore = scores.get('bertscore_f1', 0)
        simcse = scores.get('simcse', 0)
        num_samples = scores.get('num_samples', 0)

        print(f"{model_name:<35} | {bleu1:>8.4f} | {meteor:>8.4f} | {bertscore:>12.4f} | {simcse:>8.4f} | {num_samples:>8}")

    print("="*120)


def plot_comparison(results: Dict[str, Dict], output_dir: str):
    """Plot comparison of all metrics across models."""
    import matplotlib.pyplot as plt

    models = list(results.keys())
    if len(models) < 2:
        print("Need at least 2 models to plot comparison")
        return

    # Metrics to plot
    metrics = [
        ('bleu_1', 'BLEU-1'),
        ('meteor', 'METEOR'),
        ('bertscore_f1', 'BERTScore F1'),
        ('simcse', 'SimCSE')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]
        values = [results[m].get(metric_key, 0) for m in models]
        stds = [results[m].get(f'{metric_key}_std', 0) for m in models]

        bars = ax.bar(range(len(models)), values, color=colors, yerr=stds, capsize=5)

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} by Model', fontsize=14)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m[:20] for m in models], rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, 'semantic_comparison.png')
    pdf_path = os.path.join(output_dir, 'semantic_comparison.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved plots to {png_path}")
    plt.close()

    # Create radar chart for overall comparison
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    metric_names = [m[1] for m in metrics]
    num_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for i, model in enumerate(models[:6]):  # Limit to 6 models for readability
        values = [results[model].get(m[0], 0) for m in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model[:25], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_title('Model Comparison (Radar)', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    radar_path = os.path.join(output_dir, 'semantic_radar.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to {radar_path}")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("SEMANTIC SIMILARITY EVALUATION")
    print("="*70)
    print(f"Metrics: {args.metrics}")
    print(f"Output dir: {args.output_dir}")
    print(f"BERTScore model: {args.bertscore_model}")
    print(f"SimCSE model: {args.simcse_model}")
    print("="*70)

    if args.results_file:
        # Single file evaluation
        results = evaluate_single_file(args.results_file, args)
        print_results_table(results)

        # Save results
        output_path = os.path.join(args.output_dir, 'semantic_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {output_path}")

    elif args.compare_dir:
        # Multiple file comparison
        all_results = compare_multiple_files(args.compare_dir, args)
        print_results_table(all_results)

        # Save results
        output_path = os.path.join(args.output_dir, 'semantic_comparison.json')
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved comparison to {output_path}")

        # Plot comparison
        if len(all_results) > 1:
            plot_comparison(all_results, args.output_dir)

    else:
        print("Error: Specify --results-file or --compare-dir")
        return

    print("\nSemantic similarity evaluation complete!")


if __name__ == '__main__':
    main()
