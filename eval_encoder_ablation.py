#!/usr/bin/env python
"""
Evaluate GeneChat with different gene encoders.

This script evaluates how different DNA encoders affect GeneChat's performance
on gene function prediction using BLEU and METEOR metrics.

Encoders evaluated:
1. no_encoder - Raw DNA sequence directly to LLaMA (max 10k bp)
2. dnabert - Original DNABERT (not DNABERT-2)
3. hyenadna - HyenaDNA model
4. nucleotide_transformer - Nucleotide Transformer

Usage:
    python eval_encoder_ablation.py --encoder no_encoder --max-samples 100 --gpu-id 0
    python eval_encoder_ablation.py --run-all --max-samples 100 --gpu-id 0
"""

import argparse
import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Import GeneChat modules
from genechat.common.config import Config
from genechat.common.registry import registry
from genechat.common.dist_utils import get_rank
from genechat.common.conversation import Chat, CONV_VISION

# imports modules for registration
from genechat.datasets.builders import *
from genechat.models import *
from genechat.runners import *
from genechat.tasks import *


# Encoder configurations
ENCODER_CONFIGS = {
    'no_encoder': {
        'name': 'No Encoder (Raw Sequence)',
        'max_seq_length': 2000,  # Reduced to fit in LLM context (2000 chars ~ 500-1000 tokens)
        'model_id': None,
        'description': 'Raw DNA sequence directly as text to LLM'
    },
    'dnabert': {
        'name': 'DNABERT (Original)',
        'max_seq_length': 512,
        'model_id': 'zhihan1996/DNABERT-S',
        'description': 'Original DNABERT with k-mer tokenization'
    },
    'hyenadna': {
        'name': 'HyenaDNA',
        'max_seq_length': 160000,
        'model_id': 'LongSafari/hyenadna-medium-160k-seqlen-hf',
        'description': 'HyenaDNA with character-level tokenization'
    },
    'nucleotide_transformer': {
        'name': 'Nucleotide Transformer',
        'max_seq_length': 6000,
        'model_id': 'InstaDeepAI/nucleotide-transformer-500m-human-ref',
        'description': 'Nucleotide Transformer for genomic sequences'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GeneChat with different encoders')
    parser.add_argument('--encoder', type=str, choices=list(ENCODER_CONFIGS.keys()),
                        help='Encoder to use')
    parser.add_argument('--cfg-path', type=str, default='configs/genechat_eval.yaml')
    parser.add_argument('--test-seq-path', type=str,
                        default='/home/namdo/applications/GeneChat/exon_count/data/train_with_names/test_set/seq.json')
    parser.add_argument('--test-qa-path', type=str,
                        default='/home/namdo/applications/GeneChat/exon_count/data/train_with_names/test_set/qa_summary_rule.json')
    parser.add_argument('--output-dir', type=str, default='results/encoder_ablation')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--run-all', action='store_true', help='Run all encoders')
    parser.add_argument('--plot-only', type=str, default=None)
    return parser.parse_args()


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


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


def evaluate_no_encoder(seqs, qa_list, cfg_path, gpu_id, max_samples, num_beams):
    """Evaluate without encoder - raw DNA sequence as text prompt."""
    encoder_config = ENCODER_CONFIGS['no_encoder']
    print(f"\n{'='*70}")
    print(f"Evaluating: {encoder_config['name']}")
    print(f"Description: {encoder_config['description']}")
    print(f"Max sequence length: {encoder_config['max_seq_length']:,} bp")
    print(f"{'='*70}\n")

    device = f'cuda:{gpu_id}'

    # Load LLaMA/Vicuna model directly
    class Args:
        pass
    args = Args()
    args.cfg_path = cfg_path
    args.gpu_id = gpu_id
    args.options = []

    print("Loading config...")
    cfg = Config(args)
    llama_path = cfg.model_cfg.llama_model
    print(f"LLaMA path: {llama_path}")

    print(f"Loading tokenizer from {llama_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
        print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return [], {'error': str(e)}

    print(f"Loading model from {llama_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.float16,
            device_map={'': gpu_id}
        )
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return [], {'error': str(e)}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    predictions = []
    samples = qa_list[:max_samples] if max_samples else qa_list
    max_seq_len = encoder_config['max_seq_length']

    print(f"\nStarting evaluation loop with {len(samples)} samples...")
    print(f"Max sequence length: {max_seq_len}")

    error_count = 0
    success_count = 0
    skip_count = 0

    for idx, item in enumerate(tqdm(samples, desc="Evaluating no_encoder")):
        gene_id = item['Gene Id']
        ground_truth = item['Summary']

        if gene_id not in seqs:
            skip_count += 1
            if skip_count <= 3:
                print(f"Warning: Gene {gene_id} not found in sequences")
            continue

        # Debug first iteration
        if idx == 0:
            print(f"Processing first gene: {gene_id}")

        seq = seqs[gene_id]
        if isinstance(seq, list):
            seq = seq[0]

        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]

        try:
            # Create a concise prompt that fits in context
            prompt = f"Human: Given this DNA sequence: {seq}\n\nWhat is the function of this gene?\n\nAssistant:"

            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=3000).to(device)

            # Remove token_type_ids if present (LLaMA doesn't use it)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            # Debug: print token count for first sample
            if success_count == 0:
                print(f"Debug: Input token count = {inputs['input_ids'].shape[1]}")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=256,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.split('###')[0].split('Human:')[0].strip()

            if not response:
                response = "<empty response>"

            predictions.append({
                'gene_id': gene_id,
                'encoder': 'no_encoder',
                'correct_func': ground_truth,
                'predict_func': response,
                'seq_length_used': len(seq)
            })
            success_count += 1

            # Print first prediction for debugging
            if success_count == 1:
                print(f"\nFirst prediction sample:")
                print(f"  Ground truth: {ground_truth[:100]}...")
                print(f"  Prediction: {response[:100]}...")

        except Exception as e:
            error_count += 1
            print(f"Error processing gene {gene_id}: {e}")
            import traceback
            if error_count <= 3:  # Only print full traceback for first 3 errors
                traceback.print_exc()
            continue

    print(f"\nCompleted: {success_count} successful, {error_count} errors, {skip_count} skipped")

    scores = compute_bleu_meteor(predictions)
    scores['encoder'] = 'no_encoder'
    scores['encoder_name'] = encoder_config['name']
    scores['max_seq_length'] = max_seq_len

    return predictions, scores


def evaluate_with_genechat_encoder(encoder_name, seqs, qa_list, cfg_path, gpu_id, max_samples, num_beams):
    """Evaluate using GeneChat's built-in encoder (DNABERT-2) with sequence length cap."""
    encoder_config = ENCODER_CONFIGS[encoder_name]
    print(f"\n{'='*70}")
    print(f"Evaluating: {encoder_config['name']}")
    print(f"Description: {encoder_config['description']}")
    print(f"Max sequence length: {encoder_config['max_seq_length']:,} bp")
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
    max_seq_len = encoder_config['max_seq_length']

    questions = ["Tell me about this gene.", "Please provide a detailed description of the gene."]

    for item in tqdm(samples, desc=f"Evaluating {encoder_name}"):
        gene_id = item['Gene Id']
        ground_truth = item['Summary']

        if gene_id not in seqs:
            continue

        seq = seqs[gene_id]
        if isinstance(seq, list):
            seq = seq[0]

        # Truncate to encoder's max length
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]

        try:
            query = random.choice(questions)
            chat_state = CONV_VISION.copy()
            img_list = []

            gene_embeds, msg = chat.upload_gene(seq, chat_state, img_list, gene_id)
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
                'encoder': encoder_name,
                'correct_func': ground_truth,
                'predict_func': llm_message,
                'seq_length_used': len(seq)
            })

        except Exception as e:
            print(f"Error processing gene {gene_id}: {e}")
            continue

    scores = compute_bleu_meteor(predictions)
    scores['encoder'] = encoder_name
    scores['encoder_name'] = encoder_config['name']
    scores['max_seq_length'] = max_seq_len

    return predictions, scores


def plot_results(results, output_dir):
    """Plot encoder comparison results."""
    import matplotlib.pyplot as plt

    encoders = list(results.keys())
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor']

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colors for each encoder
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # Plot BLEU scores
    ax1 = axes[0]
    x = np.arange(4)
    width = 0.2

    for i, encoder in enumerate(encoders):
        bleu_values = [results[encoder]['bleu_1'], results[encoder]['bleu_2'],
                       results[encoder]['bleu_3'], results[encoder]['bleu_4']]
        ax1.bar(x + i * width, bleu_values, width, label=ENCODER_CONFIGS[encoder]['name'],
                color=colors[i % len(colors)])

    ax1.set_xlabel('BLEU N-gram', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('BLEU Scores by Encoder', fontsize=14)
    ax1.set_xticks(x + width * (len(encoders) - 1) / 2)
    ax1.set_xticklabels(['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot METEOR scores
    ax2 = axes[1]
    meteor_values = [results[enc]['meteor'] for enc in encoders]
    meteor_stds = [results[enc].get('meteor_std', 0) for enc in encoders]
    bars = ax2.bar(range(len(encoders)), meteor_values, color=colors[:len(encoders)])
    ax2.errorbar(range(len(encoders)), meteor_values, yerr=meteor_stds, fmt='none', color='black', capsize=5)

    ax2.set_xlabel('Encoder', fontsize=12)
    ax2.set_ylabel('METEOR Score', fontsize=12)
    ax2.set_title('METEOR Scores by Encoder', fontsize=14)
    ax2.set_xticks(range(len(encoders)))
    ax2.set_xticklabels([ENCODER_CONFIGS[enc]['name'] for enc in encoders], rotation=15, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots
    png_path = os.path.join(output_dir, 'encoder_ablation.png')
    pdf_path = os.path.join(output_dir, 'encoder_ablation.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plots to {png_path} and {pdf_path}")
    plt.close()


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "="*100)
    print("ENCODER ABLATION RESULTS")
    print("="*100)
    print(f"{'Encoder':<30} | {'BLEU-1':>8} | {'BLEU-2':>8} | {'BLEU-3':>8} | {'BLEU-4':>8} | {'METEOR':>8} | {'Samples':>8}")
    print("-"*100)

    for encoder, scores in results.items():
        name = ENCODER_CONFIGS[encoder]['name'][:28]
        print(f"{name:<30} | {scores['bleu_1']:>8.4f} | {scores['bleu_2']:>8.4f} | "
              f"{scores['bleu_3']:>8.4f} | {scores['bleu_4']:>8.4f} | {scores['meteor']:>8.4f} | {scores['num_samples']:>8}")

    print("="*100 + "\n")


def main():
    args = parse_args()
    setup_seeds()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("GENECHAT ENCODER ABLATION STUDY")
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

    # Determine which encoders to run
    if args.run_all:
        encoders_to_run = list(ENCODER_CONFIGS.keys())
    elif args.encoder:
        encoders_to_run = [args.encoder]
    else:
        print("Error: Specify --encoder or --run-all")
        return

    all_results = {}

    for encoder_name in encoders_to_run:
        if encoder_name == 'no_encoder':
            predictions, scores = evaluate_no_encoder(
                seqs, qa_list, args.cfg_path, args.gpu_id, args.max_samples, args.num_beams
            )
        else:
            predictions, scores = evaluate_with_genechat_encoder(
                encoder_name, seqs, qa_list, args.cfg_path, args.gpu_id, args.max_samples, args.num_beams
            )

        all_results[encoder_name] = scores

        # Print results
        print(f"\nResults for {encoder_name}:")
        print(f"  BLEU-1: {scores['bleu_1']:.4f}")
        print(f"  BLEU-2: {scores['bleu_2']:.4f}")
        print(f"  BLEU-3: {scores['bleu_3']:.4f}")
        print(f"  BLEU-4: {scores['bleu_4']:.4f}")
        print(f"  METEOR: {scores['meteor']:.4f}")

        # Save individual results
        result_path = os.path.join(args.output_dir, f'results_{encoder_name}.json')
        with open(result_path, 'w') as f:
            json.dump({'scores': scores, 'predictions': predictions}, f, indent=2)
        print(f"Saved to {result_path}")

    # Save combined results
    summary_path = os.path.join(args.output_dir, 'encoder_ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print table and plot
    print_results_table(all_results)
    if len(all_results) > 1:
        plot_results(all_results, args.output_dir)

    print("Encoder ablation complete!")


if __name__ == '__main__':
    main()
