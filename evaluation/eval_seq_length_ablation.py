#!/usr/bin/env python
"""
Generate GeneChat predictions for different sequence lengths.

This script generates predictions for sequence length ablation study.
Run eval_semantic_similarity.py on the output JSON files to compute metrics.

Usage:
    # Step 1: Generate predictions
    python eval_seq_length_ablation.py \
        --seq-lengths 1000 5000 10000 50000 100000 160000 \
        --output-dir results/seq_length_ablation \
        --max-samples 100 \
        --gpu-id 0

    # Step 2: Compute metrics on each predictions file
    python eval_semantic_similarity.py \
        --results-file results/seq_length_ablation/predictions_seqlen_1000.json \
        --output-dir results/seq_length_ablation/metrics_1000
"""

import argparse
import os
import random
import json
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# Import GeneChat modules
from genechat.common.config import Config
from genechat.common.registry import registry
from genechat.common.dist_utils import get_rank, init_distributed_mode
from genechat.common.conversation import Chat, CONV_VISION

# imports modules for registration
from genechat.datasets.builders import *
from genechat.models import *
from genechat.runners import *
from genechat.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate GeneChat predictions for different sequence lengths'
    )
    parser.add_argument(
        '--cfg-path',
        type=str,
        default='configs/genechat_eval.yaml',
        help='Path to GeneChat config file'
    )
    parser.add_argument(
        '--test-seq-path',
        type=str,
        default=None,
        help='Path to sequences JSON file (defaults to config path)'
    )
    parser.add_argument(
        '--test-qa-path',
        type=str,
        default=None,
        help='Path to QA annotations JSON file (defaults to config path)'
    )
    parser.add_argument(
        '--seq-lengths',
        type=int,
        nargs='+',
        default=[1000, 5000, 10000, 25000, 50000, 100000, 160000],
        help='List of sequence lengths to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/seq_length_ablation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID to use'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per evaluation (None for all)'
    )
    parser.add_argument(
        '--num-beams',
        type=int,
        default=4,
        help='Number of beams for generation'
    )
    parser.add_argument(
        '--min-seq-length',
        type=int,
        default=None,
        help='Minimum sequence length filter - only evaluate samples with sequences >= this length'
    )
    parser.add_argument(
        '--options',
        nargs='+',
        default=[],
        help='Override config options'
    )
    return parser.parse_args()


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_model(cfg_path, gpu_id=0):
    """Load GeneChat model from config."""
    print('Initializing GeneChat model...')

    # Create a simple args object for Config
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
    model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
    model.eval()

    chat = Chat(model, device=f'cuda:{gpu_id}')
    print('Model initialization finished.\n')
    return chat, model, cfg


def truncate_sequence(seq, max_length):
    """Truncate sequence to specified maximum length."""
    if len(seq) > max_length:
        return seq[:max_length]
    return seq


def upload_gene(chat, gene_seq, gene_id=0):
    """Upload gene sequence to GeneChat."""
    chat_state = CONV_VISION.copy()
    img_list = []
    gene_embeds, msg = chat.upload_gene(gene_seq, chat_state, img_list, gene_id)
    return chat_state, img_list, gene_embeds


def gradio_ask(chat, user_message, chat_state):
    """Ask a question."""
    chat.ask(user_message, chat_state)
    return chat_state


def gradio_answer(chat, chat_state, img_list, num_beams=4):
    """Generate answer from GeneChat."""
    llm_message, _, loss = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=1e-3,
        max_new_tokens=512,
        max_length=1500
    )
    return llm_message, chat_state, img_list, loss


def generate_predictions(chat, seqs, qa_list, seq_length, max_samples=None, num_beams=4):
    """
    Generate GeneChat predictions at a specific sequence length.

    Args:
        chat: GeneChat Chat object
        seqs: Dictionary mapping gene IDs to sequences
        qa_list: List of QA items with Gene Id and Summary
        seq_length: Maximum sequence length to use
        max_samples: Maximum number of samples to evaluate (None for all)
        num_beams: Number of beams for generation

    Returns:
        List of prediction dictionaries
    """
    predictions = []

    samples = qa_list[:max_samples] if max_samples else qa_list

    questions = [
        "Tell me about this gene.",
        "Please provide a detailed description of the gene."
    ]

    for item in tqdm(samples, desc=f"Generating predictions (seq_length={seq_length:,})"):
        gene_id = item['Gene Id']
        ground_truth = item['Summary']

        if gene_id not in seqs:
            print(f"Warning: Gene {gene_id} not found in sequences, skipping")
            continue

        seq = seqs[gene_id]
        if isinstance(seq, list):
            seq = seq[0]

        # Truncate sequence to specified length
        truncated_seq = truncate_sequence(seq, seq_length)

        try:
            # Select random question
            query = random.choice(questions)

            # Upload gene
            chat_state, img_list, gene_embeds = upload_gene(chat, truncated_seq, gene_id)

            # Ask question
            chat_state = gradio_ask(chat, query, chat_state)

            # Generate answer
            llm_message, chat_state, img_list, loss = gradio_answer(
                chat, chat_state, img_list, num_beams=num_beams
            )

            predictions.append({
                'gene_id': gene_id,
                'query': query,
                'correct_func': ground_truth,
                'predict_func': llm_message,
                'seq_length_used': len(truncated_seq),
                'original_seq_length': len(seq),
                'target_seq_length': seq_length
            })

        except Exception as e:
            print(f"Error processing gene {gene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return predictions


def run_sequence_length_ablation(args):
    """
    Generate predictions for different sequence lengths.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup seeds
    setup_seeds(42)

    # Load model
    chat, model, cfg = load_model(args.cfg_path, args.gpu_id)

    # Get data paths from config if not provided
    if args.test_seq_path is None or args.test_qa_path is None:
        try:
            data_dir = cfg.datasets_cfg.seq.build_info.train.storage
            if args.test_seq_path is None:
                args.test_seq_path = os.path.join(data_dir, 'seq.json')
            if args.test_qa_path is None:
                args.test_qa_path = os.path.join(data_dir, 'qa_summary_rule.json')
            print(f"Using data paths from config: {data_dir}")
        except Exception as e:
            print(f"Error: Could not get data paths from config: {e}")
            print("Please provide --test-seq-path and --test-qa-path")
            return None

    # Load data
    print("Loading sequences and QA data...")
    print(f"  Sequences: {args.test_seq_path}")
    print(f"  QA: {args.test_qa_path}")
    with open(args.test_seq_path, 'r') as f:
        seqs = json.load(f)
    with open(args.test_qa_path, 'r') as f:
        qa_list = json.load(f)

    print(f"Loaded {len(seqs)} sequences and {len(qa_list)} QA items")

    # Filter by minimum sequence length if specified
    if args.min_seq_length is not None:
        filtered_qa_list = []
        for item in qa_list:
            gene_id = item['Gene Id']
            if gene_id in seqs:
                seq = seqs[gene_id]
                if isinstance(seq, list):
                    seq = seq[0]
                if len(seq) >= args.min_seq_length:
                    filtered_qa_list.append(item)
        print(f"Filtered to {len(filtered_qa_list)} samples with seq length >= {args.min_seq_length:,} bp")
        qa_list = filtered_qa_list

    print()

    # Sort sequence lengths
    sequence_lengths = sorted(args.seq_lengths)

    # Generate predictions for each sequence length
    prediction_files = []

    for seq_length in sequence_lengths:
        print(f"\n{'='*70}")
        print(f"Generating predictions with max sequence length: {seq_length:,} bp")
        print(f"{'='*70}")

        predictions = generate_predictions(
            chat, seqs, qa_list, seq_length,
            max_samples=args.max_samples,
            num_beams=args.num_beams
        )

        # Save predictions
        output_file = os.path.join(args.output_dir, f'predictions_seqlen_{seq_length}.json')
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {output_file}")
        prediction_files.append(output_file)

    return prediction_files


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("GENECHAT SEQUENCE LENGTH ABLATION - PREDICTION GENERATION")
    print("="*70)
    print(f"Config: {args.cfg_path}")
    print(f"Test sequences: {args.test_seq_path or '(from config)'}")
    print(f"Test QA: {args.test_qa_path or '(from config)'}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Max samples: {args.max_samples}")
    print(f"Min seq length filter: {args.min_seq_length}")
    print("="*70 + "\n")

    # Generate predictions
    prediction_files = run_sequence_length_ablation(args)

    if prediction_files:
        print("\n" + "="*70)
        print("PREDICTION GENERATION COMPLETE")
        print("="*70)
        print("\nGenerated prediction files:")
        for f in prediction_files:
            print(f"  {f}")

        print("\nNext step: Run eval_semantic_similarity.py on each file to compute metrics:")
        print(f"\n  for f in {args.output_dir}/predictions_seqlen_*.json; do")
        print(f"    python eval_semantic_similarity.py --results-file $f --output-dir ${{f%.json}}_metrics")
        print("  done")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
