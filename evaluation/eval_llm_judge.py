#!/usr/bin/env python
"""
LLM-as-Judge Evaluation for GeneChat.

This script uses Claude (or GPT-4) to evaluate the quality of GeneChat's
gene descriptions by scoring them on multiple criteria.

Evaluation Criteria:
1. Accuracy - Is the biological information correct?
2. Completeness - Does it cover key gene functions?
3. Relevance - Is the information relevant to the gene?
4. Fluency - Is the text well-written and coherent?

Usage:
    python eval_llm_judge.py --results-file results/evaluation_results.json --judge claude
    python eval_llm_judge.py --results-file results/evaluation_results.json --judge gpt4
"""

import argparse
import os
import json
import time
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='LLM-as-Judge evaluation for GeneChat')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to GeneChat evaluation results JSON file')
    parser.add_argument('--output-dir', type=str, default='results/llm_judge')
    parser.add_argument('--judge', type=str, choices=['claude', 'gpt4'], default='claude',
                        help='LLM to use as judge')
    parser.add_argument('--claude-api-key', type=str, default=None,
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to use (e.g., claude-sonnet-4-20250514, gpt-4o)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls (seconds)')
    return parser.parse_args()


def create_judge_prompt(ground_truth, prediction, gene_id=None):
    """Create the evaluation prompt for LLM judge."""
    prompt = f"""You are an expert in molecular biology and genetics. Your task is to evaluate the quality of an AI-generated gene description by comparing it to the ground truth.

**Gene ID:** {gene_id if gene_id else 'Unknown'}

**Ground Truth Description:**
{ground_truth}

**AI-Generated Description:**
{prediction}

Please evaluate the AI-generated description on the following criteria using a scale of 1-10:

1. **Accuracy (1-10)**: Is the biological information factually correct? Does it avoid hallucinations or false claims?
   - 1-3: Major factual errors or hallucinations
   - 4-6: Some inaccuracies but mostly correct
   - 7-10: Highly accurate, no significant errors

2. **Completeness (1-10)**: Does it cover the key functions and characteristics of the gene?
   - 1-3: Missing most important information
   - 4-6: Covers some key points but misses others
   - 7-10: Comprehensive coverage of gene function

3. **Relevance (1-10)**: Is the information specifically relevant to this gene (not generic)?
   - 1-3: Generic or off-topic information
   - 4-6: Somewhat relevant but vague
   - 7-10: Highly specific and relevant to the gene

4. **Fluency (1-10)**: Is the text well-written, coherent, and easy to understand?
   - 1-3: Poorly written, hard to understand
   - 4-6: Readable but could be improved
   - 7-10: Well-written and clear

**Response Format:**
Please respond with ONLY a JSON object in the following format (no additional text):
{{
    "accuracy": <score>,
    "completeness": <score>,
    "relevance": <score>,
    "fluency": <score>,
    "overall": <average of all scores>,
    "reasoning": "<brief explanation of your scores>"
}}
"""
    return prompt


def evaluate_with_claude(predictions, api_key, model, delay):
    """Evaluate predictions using Claude as judge."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        return None

    if not api_key:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Anthropic API key not provided. Set --claude-api-key or ANTHROPIC_API_KEY env var")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    model = model or "claude-sonnet-4-20250514"

    results = []

    for pred in tqdm(predictions, desc=f"Evaluating with Claude ({model})"):
        gene_id = pred.get('gene_id', None)
        ground_truth = pred.get('correct_func', '')
        prediction = pred.get('predict_func', '')

        if not ground_truth or not prediction:
            continue

        prompt = create_judge_prompt(ground_truth, prediction, gene_id)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON response
            try:
                # Handle potential markdown code blocks
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]

                scores = json.loads(response_text)

                result = {
                    'gene_id': gene_id,
                    'accuracy': scores.get('accuracy', 0),
                    'completeness': scores.get('completeness', 0),
                    'relevance': scores.get('relevance', 0),
                    'fluency': scores.get('fluency', 0),
                    'overall': scores.get('overall', 0),
                    'reasoning': scores.get('reasoning', ''),
                    'ground_truth': ground_truth,
                    'prediction': prediction
                }
                results.append(result)

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse response for gene {gene_id}: {e}")
                print(f"Response: {response_text[:200]}...")
                continue

        except Exception as e:
            print(f"Error evaluating gene {gene_id}: {e}")
            continue

        time.sleep(delay)

    return results


def evaluate_with_gpt4(predictions, api_key, model, delay):
    """Evaluate predictions using GPT-4 as judge."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        return None

    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not provided. Set --openai-api-key or OPENAI_API_KEY env var")
        return None

    client = OpenAI(api_key=api_key)
    model = model or "gpt-4o"

    results = []

    for pred in tqdm(predictions, desc=f"Evaluating with GPT-4 ({model})"):
        gene_id = pred.get('gene_id', None)
        ground_truth = pred.get('correct_func', '')
        prediction = pred.get('predict_func', '')

        if not ground_truth or not prediction:
            continue

        prompt = create_judge_prompt(ground_truth, prediction, gene_id)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0
            )

            response_text = response.choices[0].message.content.strip()

            # Parse JSON response
            try:
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]

                scores = json.loads(response_text)

                result = {
                    'gene_id': gene_id,
                    'accuracy': scores.get('accuracy', 0),
                    'completeness': scores.get('completeness', 0),
                    'relevance': scores.get('relevance', 0),
                    'fluency': scores.get('fluency', 0),
                    'overall': scores.get('overall', 0),
                    'reasoning': scores.get('reasoning', ''),
                    'ground_truth': ground_truth,
                    'prediction': prediction
                }
                results.append(result)

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse response for gene {gene_id}: {e}")
                continue

        except Exception as e:
            print(f"Error evaluating gene {gene_id}: {e}")
            continue

        time.sleep(delay)

    return results


def compute_statistics(results):
    """Compute aggregate statistics from LLM judge results."""
    if not results:
        return {}

    accuracy_scores = [r['accuracy'] for r in results]
    completeness_scores = [r['completeness'] for r in results]
    relevance_scores = [r['relevance'] for r in results]
    fluency_scores = [r['fluency'] for r in results]
    overall_scores = [r['overall'] for r in results]

    stats = {
        'accuracy': {
            'mean': float(np.mean(accuracy_scores)),
            'std': float(np.std(accuracy_scores)),
            'min': float(np.min(accuracy_scores)),
            'max': float(np.max(accuracy_scores))
        },
        'completeness': {
            'mean': float(np.mean(completeness_scores)),
            'std': float(np.std(completeness_scores)),
            'min': float(np.min(completeness_scores)),
            'max': float(np.max(completeness_scores))
        },
        'relevance': {
            'mean': float(np.mean(relevance_scores)),
            'std': float(np.std(relevance_scores)),
            'min': float(np.min(relevance_scores)),
            'max': float(np.max(relevance_scores))
        },
        'fluency': {
            'mean': float(np.mean(fluency_scores)),
            'std': float(np.std(fluency_scores)),
            'min': float(np.min(fluency_scores)),
            'max': float(np.max(fluency_scores))
        },
        'overall': {
            'mean': float(np.mean(overall_scores)),
            'std': float(np.std(overall_scores)),
            'min': float(np.min(overall_scores)),
            'max': float(np.max(overall_scores))
        },
        'num_samples': len(results)
    }

    return stats


def plot_results(stats, results, output_dir, judge_name):
    """Plot LLM judge results."""
    import matplotlib.pyplot as plt

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean scores by criteria
    ax1 = axes[0]
    criteria = ['Accuracy', 'Completeness', 'Relevance', 'Fluency', 'Overall']
    means = [stats['accuracy']['mean'], stats['completeness']['mean'],
             stats['relevance']['mean'], stats['fluency']['mean'], stats['overall']['mean']]
    stds = [stats['accuracy']['std'], stats['completeness']['std'],
            stats['relevance']['std'], stats['fluency']['std'], stats['overall']['std']]

    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c']
    bars = ax1.bar(criteria, means, yerr=stds, capsize=5, color=colors)

    ax1.set_ylabel('Score (1-10)', fontsize=12)
    ax1.set_title(f'LLM Judge Scores ({judge_name})', fontsize=14)
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{mean:.2f}', ha='center', fontsize=10)

    # Plot 2: Distribution of overall scores
    ax2 = axes[1]
    overall_scores = [r['overall'] for r in results]
    ax2.hist(overall_scores, bins=10, range=(1, 10), color='#3498db', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(overall_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(overall_scores):.2f}')
    ax2.set_xlabel('Overall Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Overall Scores', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plots
    png_path = os.path.join(output_dir, f'llm_judge_{judge_name}.png')
    pdf_path = os.path.join(output_dir, f'llm_judge_{judge_name}.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plots to {png_path}")
    plt.close()


def print_results_table(stats, judge_name):
    """Print formatted results table."""
    print("\n" + "="*80)
    print(f"LLM-AS-JUDGE RESULTS ({judge_name.upper()})")
    print("="*80)
    print(f"{'Criteria':<15} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-"*80)

    for criteria in ['accuracy', 'completeness', 'relevance', 'fluency', 'overall']:
        s = stats[criteria]
        print(f"{criteria.capitalize():<15} | {s['mean']:>8.2f} | {s['std']:>8.2f} | {s['min']:>8.2f} | {s['max']:>8.2f}")

    print("-"*80)
    print(f"Total samples evaluated: {stats['num_samples']}")
    print("="*80 + "\n")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("LLM-AS-JUDGE EVALUATION")
    print("="*70)
    print(f"Results file: {args.results_file}")
    print(f"Judge: {args.judge}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print("="*70 + "\n")

    # Load predictions
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file, 'r') as f:
        data = json.load(f)

    # Handle different result file formats
    if isinstance(data, list):
        predictions = data
    elif isinstance(data, dict):
        if 'predictions' in data:
            predictions = data['predictions']
        else:
            # Assume it's the output from eval_seq_length_ablation or similar
            predictions = data
    else:
        print("Error: Unknown results file format")
        return

    # Filter to items with correct_func and predict_func
    predictions = [p for p in predictions if 'correct_func' in p and 'predict_func' in p]

    if args.max_samples:
        predictions = predictions[:args.max_samples]

    print(f"Loaded {len(predictions)} predictions to evaluate\n")

    # Run evaluation
    if args.judge == 'claude':
        results = evaluate_with_claude(predictions, args.claude_api_key, args.model, args.delay)
    else:
        results = evaluate_with_gpt4(predictions, args.openai_api_key, args.model, args.delay)

    if not results:
        print("No results obtained. Check your API key and try again.")
        return

    # Compute statistics
    stats = compute_statistics(results)

    # Print results
    print_results_table(stats, args.judge)

    # Save results
    output_file = os.path.join(args.output_dir, f'llm_judge_{args.judge}_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'statistics': stats,
            'individual_results': results,
            'judge': args.judge,
            'model': args.model
        }, f, indent=2)
    print(f"Saved detailed results to {output_file}")

    # Save summary
    summary_file = os.path.join(args.output_dir, f'llm_judge_{args.judge}_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved summary to {summary_file}")

    # Plot results
    plot_results(stats, results, args.output_dir, args.judge)

    print("\nLLM-as-Judge evaluation complete!")


if __name__ == '__main__':
    main()
