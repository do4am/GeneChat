#!/usr/bin/env python
"""
Robustness Testing for GeneChat.

This script evaluates GeneChat's robustness to sequence perturbations:
1. Random mutations (1%, 5%, 10% of nucleotides)
2. N-masking (replace portions with 'N')
3. Truncation (remove start, middle, or end)
4. Reversal (reverse the sequence)
5. Shuffling (shuffle chunks of the sequence)

Usage:
    python eval_robustness.py --perturbation mutation --mutation-rate 0.05 --max-samples 100
    python eval_robustness.py --run-all --max-samples 100
"""

import argparse
import os
import random
import json
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

from genechat.common.config import Config
from genechat.common.registry import registry
from genechat.common.conversation import Chat, CONV_VISION

from genechat.datasets.builders import *
from genechat.models import *
from genechat.runners import *
from genechat.tasks import *


# Perturbation configurations
PERTURBATION_CONFIGS = {
    'original': {
        'name': 'Original (No Perturbation)',
        'description': 'Baseline - no changes to sequence'
    },
    'mutation_1': {
        'name': 'Random Mutation (1%)',
        'description': '1% of nucleotides randomly mutated',
        'rate': 0.01
    },
    'mutation_5': {
        'name': 'Random Mutation (5%)',
        'description': '5% of nucleotides randomly mutated',
        'rate': 0.05
    },
    'mutation_10': {
        'name': 'Random Mutation (10%)',
        'description': '10% of nucleotides randomly mutated',
        'rate': 0.10
    },
    'n_mask_10': {
        'name': 'N-Masking (10%)',
        'description': '10% of nucleotides replaced with N',
        'rate': 0.10
    },
    'n_mask_25': {
        'name': 'N-Masking (25%)',
        'description': '25% of nucleotides replaced with N',
        'rate': 0.25
    },
    'truncate_start': {
        'name': 'Truncate Start (50%)',
        'description': 'Remove first 50% of sequence',
        'position': 'start',
        'rate': 0.50
    },
    'truncate_end': {
        'name': 'Truncate End (50%)',
        'description': 'Remove last 50% of sequence',
        'position': 'end',
        'rate': 0.50
    },
    'reverse': {
        'name': 'Reverse Sequence',
        'description': 'Reverse the entire sequence'
    },
    'shuffle_chunks': {
        'name': 'Shuffle Chunks',
        'description': 'Shuffle 1000bp chunks randomly'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Robustness testing for GeneChat')
    parser.add_argument('--perturbation', type=str, choices=list(PERTURBATION_CONFIGS.keys()),
                        help='Perturbation type to apply')
    parser.add_argument('--cfg-path', type=str, default='configs/genechat_eval.yaml')
    parser.add_argument('--test-seq-path', type=str,
                        default='/home/namdo/applications/GeneChat/exon_count/data/train_with_names/test_set/seq.json')
    parser.add_argument('--test-qa-path', type=str,
                        default='/home/namdo/applications/GeneChat/exon_count/data/train_with_names/test_set/qa_summary_rule.json')
    parser.add_argument('--output-dir', type=str, default='results/robustness')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--run-all', action='store_true', help='Run all perturbation types')
    parser.add_argument('--plot-only', type=str, default=None)
    return parser.parse_args()


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ============== Perturbation Functions ==============

def apply_mutation(seq, rate):
    """Randomly mutate nucleotides at given rate."""
    nucleotides = ['A', 'C', 'G', 'T']
    seq_list = list(seq)
    num_mutations = int(len(seq) * rate)

    positions = random.sample(range(len(seq)), num_mutations)
    for pos in positions:
        current = seq_list[pos]
        # Mutate to a different nucleotide
        choices = [n for n in nucleotides if n != current]
        if choices:
            seq_list[pos] = random.choice(choices)

    return ''.join(seq_list)


def apply_n_masking(seq, rate):
    """Replace random positions with 'N' (unknown nucleotide)."""
    seq_list = list(seq)
    num_masks = int(len(seq) * rate)

    positions = random.sample(range(len(seq)), num_masks)
    for pos in positions:
        seq_list[pos] = 'N'

    return ''.join(seq_list)


def apply_truncation(seq, position, rate):
    """Truncate sequence from start or end."""
    truncate_len = int(len(seq) * rate)

    if position == 'start':
        return seq[truncate_len:]
    elif position == 'end':
        return seq[:-truncate_len] if truncate_len > 0 else seq
    elif position == 'middle':
        # Remove middle portion
        mid = len(seq) // 2
        half_trunc = truncate_len // 2
        return seq[:mid - half_trunc] + seq[mid + half_trunc:]

    return seq


def apply_reverse(seq):
    """Reverse the sequence."""
    return seq[::-1]


def apply_shuffle_chunks(seq, chunk_size=1000):
    """Shuffle chunks of the sequence."""
    chunks = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]
    random.shuffle(chunks)
    return ''.join(chunks)


def apply_perturbation(seq, perturbation_type):
    """Apply the specified perturbation to a sequence."""
    if perturbation_type == 'original':
        return seq
    elif perturbation_type.startswith('mutation_'):
        rate = PERTURBATION_CONFIGS[perturbation_type]['rate']
        return apply_mutation(seq, rate)
    elif perturbation_type.startswith('n_mask_'):
        rate = PERTURBATION_CONFIGS[perturbation_type]['rate']
        return apply_n_masking(seq, rate)
    elif perturbation_type.startswith('truncate_'):
        config = PERTURBATION_CONFIGS[perturbation_type]
        return apply_truncation(seq, config['position'], config['rate'])
    elif perturbation_type == 'reverse':
        return apply_reverse(seq)
    elif perturbation_type == 'shuffle_chunks':
        return apply_shuffle_chunks(seq)

    return seq


# ============== Evaluation Functions ==============

def compute_bleu_meteor(predictions):
    """Compute BLEU and METEOR scores."""
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    meteor_scores_list = []

    weights = {
        1: (1.0, 0, 0, 0),
        2: (0.5, 0.5, 0, 0),
        3: (1/3, 1/3, 1/3, 0),
        4: (0.25, 0.25, 0.25, 0.25)
    }
    smoothing = SmoothingFunction().method1

    for pred in predictions:
        reference = pred['correct_func']
        hypothesis = pred['predict_func']

        try:
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
        except:
            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()

        if len(hyp_tokens) == 0:
            hyp_tokens = ['<empty>']

        for n in [1, 2, 3, 4]:
            try:
                score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights[n], smoothing_function=smoothing)
                bleu_scores[n].append(score)
            except:
                bleu_scores[n].append(0.0)

        try:
            meteor = meteor_score([ref_tokens], hyp_tokens)
            meteor_scores_list.append(meteor)
        except:
            meteor_scores_list.append(0.0)

    return {
        'bleu_1': float(np.mean(bleu_scores[1])),
        'bleu_2': float(np.mean(bleu_scores[2])),
        'bleu_3': float(np.mean(bleu_scores[3])),
        'bleu_4': float(np.mean(bleu_scores[4])),
        'meteor': float(np.mean(meteor_scores_list)),
        'bleu_1_std': float(np.std(bleu_scores[1])),
        'bleu_2_std': float(np.std(bleu_scores[2])),
        'bleu_3_std': float(np.std(bleu_scores[3])),
        'bleu_4_std': float(np.std(bleu_scores[4])),
        'meteor_std': float(np.std(meteor_scores_list)),
        'num_samples': len(predictions)
    }


def evaluate_perturbation(perturbation_type, seqs, qa_list, cfg_path, gpu_id, max_samples, num_beams):
    """Evaluate GeneChat with a specific perturbation."""
    config = PERTURBATION_CONFIGS[perturbation_type]
    print(f"\n{'='*70}")
    print(f"Evaluating: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}\n")

    device = f'cuda:{gpu_id}'

    # Load GeneChat model
    class Args:
        pass
    args = Args()
    args.cfg_path = cfg_path
    args.gpu_id = gpu_id
    args.options = []
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    chat = Chat(model, device=device)

    predictions = []
    samples = qa_list[:max_samples] if max_samples else qa_list

    questions = ["Tell me about this gene.", "Please provide a detailed description of the gene."]

    for item in tqdm(samples, desc=f"Evaluating {perturbation_type}"):
        gene_id = item['Gene Id']
        ground_truth = item['Summary']

        if gene_id not in seqs:
            continue

        seq = seqs[gene_id]
        if isinstance(seq, list):
            seq = seq[0]

        # Truncate to max length first
        if len(seq) > 160000:
            seq = seq[:159999]

        # Apply perturbation
        original_len = len(seq)
        perturbed_seq = apply_perturbation(seq, perturbation_type)
        perturbed_len = len(perturbed_seq)

        try:
            query = random.choice(questions)
            chat_state = CONV_VISION.copy()
            img_list = []

            gene_embeds, msg = chat.upload_gene(perturbed_seq, chat_state, img_list, gene_id)
            chat.ask(query, chat_state)

            llm_message, _, loss = chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=num_beams,
                temperature=1e-3,
                max_new_tokens=512,
                max_length=1500
            )

            predictions.append({
                'gene_id': gene_id,
                'perturbation': perturbation_type,
                'correct_func': ground_truth,
                'predict_func': llm_message,
                'original_seq_length': original_len,
                'perturbed_seq_length': perturbed_len
            })

        except Exception as e:
            print(f"Error processing gene {gene_id}: {e}")
            continue

    scores = compute_bleu_meteor(predictions)
    scores['perturbation'] = perturbation_type
    scores['perturbation_name'] = config['name']

    return predictions, scores


def plot_results(results, output_dir):
    """Plot robustness comparison results."""
    import matplotlib.pyplot as plt

    perturbations = list(results.keys())

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(perturbations)))

    # Plot BLEU-1 and METEOR as bar charts
    ax1 = axes[0]
    x = np.arange(len(perturbations))
    bleu1_values = [results[p]['bleu_1'] for p in perturbations]
    bars = ax1.bar(x, bleu1_values, color=colors)

    ax1.set_xlabel('Perturbation Type', fontsize=12)
    ax1.set_ylabel('BLEU-1 Score', fontsize=12)
    ax1.set_title('BLEU-1 Score by Perturbation', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PERTURBATION_CONFIGS[p]['name'][:15] for p in perturbations],
                        rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add baseline reference line
    if 'original' in results:
        ax1.axhline(y=results['original']['bleu_1'], color='red', linestyle='--',
                   label=f"Baseline: {results['original']['bleu_1']:.4f}")
        ax1.legend()

    # Plot METEOR
    ax2 = axes[1]
    meteor_values = [results[p]['meteor'] for p in perturbations]
    bars = ax2.bar(x, meteor_values, color=colors)

    ax2.set_xlabel('Perturbation Type', fontsize=12)
    ax2.set_ylabel('METEOR Score', fontsize=12)
    ax2.set_title('METEOR Score by Perturbation', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([PERTURBATION_CONFIGS[p]['name'][:15] for p in perturbations],
                        rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    if 'original' in results:
        ax2.axhline(y=results['original']['meteor'], color='red', linestyle='--',
                   label=f"Baseline: {results['original']['meteor']:.4f}")
        ax2.legend()

    plt.tight_layout()

    # Save plots
    png_path = os.path.join(output_dir, 'robustness_results.png')
    pdf_path = os.path.join(output_dir, 'robustness_results.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plots to {png_path} and {pdf_path}")
    plt.close()

    # Create degradation plot
    if 'original' in results:
        fig2, ax = plt.subplots(figsize=(12, 6))

        baseline_bleu = results['original']['bleu_1']
        baseline_meteor = results['original']['meteor']

        perturbations_no_orig = [p for p in perturbations if p != 'original']
        x = np.arange(len(perturbations_no_orig))
        width = 0.35

        bleu_degradation = [(results[p]['bleu_1'] - baseline_bleu) / baseline_bleu * 100
                           for p in perturbations_no_orig]
        meteor_degradation = [(results[p]['meteor'] - baseline_meteor) / baseline_meteor * 100
                             for p in perturbations_no_orig]

        ax.bar(x - width/2, bleu_degradation, width, label='BLEU-1 Change (%)', color='#3498db')
        ax.bar(x + width/2, meteor_degradation, width, label='METEOR Change (%)', color='#e74c3c')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Perturbation Type', fontsize=12)
        ax.set_ylabel('Score Change (%)', fontsize=12)
        ax.set_title('Performance Degradation from Baseline', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([PERTURBATION_CONFIGS[p]['name'][:15] for p in perturbations_no_orig],
                          rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        degradation_path = os.path.join(output_dir, 'robustness_degradation.png')
        plt.savefig(degradation_path, dpi=300, bbox_inches='tight')
        plt.savefig(degradation_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved degradation plot to {degradation_path}")
        plt.close()


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "="*110)
    print("ROBUSTNESS TESTING RESULTS")
    print("="*110)
    print(f"{'Perturbation':<25} | {'BLEU-1':>8} | {'BLEU-2':>8} | {'BLEU-3':>8} | {'BLEU-4':>8} | {'METEOR':>8} | {'Samples':>8}")
    print("-"*110)

    baseline_bleu = results.get('original', {}).get('bleu_1', None)
    baseline_meteor = results.get('original', {}).get('meteor', None)

    for perturbation, scores in results.items():
        name = PERTURBATION_CONFIGS[perturbation]['name'][:23]

        # Calculate degradation if baseline exists
        if baseline_bleu and perturbation != 'original':
            bleu_change = (scores['bleu_1'] - baseline_bleu) / baseline_bleu * 100
            meteor_change = (scores['meteor'] - baseline_meteor) / baseline_meteor * 100
            suffix = f" ({bleu_change:+.1f}%)"
        else:
            suffix = ""

        print(f"{name:<25} | {scores['bleu_1']:>8.4f} | {scores['bleu_2']:>8.4f} | "
              f"{scores['bleu_3']:>8.4f} | {scores['bleu_4']:>8.4f} | {scores['meteor']:>8.4f} | {scores['num_samples']:>8}{suffix}")

    print("="*110 + "\n")


def main():
    args = parse_args()
    setup_seeds()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("GENECHAT ROBUSTNESS TESTING")
    print("="*70)

    if args.plot_only:
        with open(args.plot_only, 'r') as f:
            results = json.load(f)
        print_results_table(results)
        plot_results(results, os.path.dirname(args.plot_only) or '.')
        return

    # Load data
    print(f"Loading data from {args.test_seq_path}...")
    with open(args.test_seq_path, 'r') as f:
        seqs = json.load(f)
    with open(args.test_qa_path, 'r') as f:
        qa_list = json.load(f)
    print(f"Loaded {len(seqs)} sequences and {len(qa_list)} QA items\n")

    # Determine which perturbations to run
    if args.run_all:
        perturbations_to_run = list(PERTURBATION_CONFIGS.keys())
    elif args.perturbation:
        perturbations_to_run = [args.perturbation]
    else:
        print("Error: Specify --perturbation or --run-all")
        return

    all_results = {}

    for perturbation in perturbations_to_run:
        predictions, scores = evaluate_perturbation(
            perturbation, seqs, qa_list, args.cfg_path, args.gpu_id, args.max_samples, args.num_beams
        )

        all_results[perturbation] = scores

        # Print results
        print(f"\nResults for {perturbation}:")
        print(f"  BLEU-1: {scores['bleu_1']:.4f}")
        print(f"  BLEU-2: {scores['bleu_2']:.4f}")
        print(f"  BLEU-3: {scores['bleu_3']:.4f}")
        print(f"  BLEU-4: {scores['bleu_4']:.4f}")
        print(f"  METEOR: {scores['meteor']:.4f}")

        # Save individual results
        result_path = os.path.join(args.output_dir, f'results_{perturbation}.json')
        with open(result_path, 'w') as f:
            json.dump({'scores': scores, 'predictions': predictions}, f, indent=2)
        print(f"Saved to {result_path}")

    # Save combined results
    summary_path = os.path.join(args.output_dir, 'robustness_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print table and plot
    print_results_table(all_results)
    if len(all_results) > 1:
        plot_results(all_results, args.output_dir)

    print("Robustness testing complete!")


if __name__ == '__main__':
    main()
