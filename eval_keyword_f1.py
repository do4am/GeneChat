#!/usr/bin/env python3
"""
Keyword Extraction F1 Evaluation for GeneChat

This script evaluates gene function predictions using keyword extraction F1 scores.
It extracts biological keywords from both predictions and ground truth using KeyBERT,
then computes precision, recall, and F1 based on keyword overlap.

Usage:
    python eval_keyword_f1.py --results-file results.json
    python eval_keyword_f1.py --compare-dir results_remote/results/encoder_ablation_v1/
    python eval_keyword_f1.py --files file1.json file2.json --output-plot comparison.png
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Plotting disabled.")

# Try to import KeyBERT, fall back to simple extraction if not available
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("Warning: KeyBERT not installed. Using simple noun extraction.")
    print("Install with: pip install keybert")

# Try to import spaCy for fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# Biological stopwords to filter out generic terms
BIO_STOPWORDS = {
    'gene', 'genes', 'protein', 'proteins', 'sequence', 'sequences',
    'region', 'regions', 'element', 'elements', 'function', 'functions',
    'activity', 'activities', 'cell', 'cells', 'based', 'using',
    'found', 'shown', 'identified', 'validated', 'predicted',
    'genomic', 'molecular', 'biological', 'functional',
    'regulatory', 'transcriptional', 'expression',
    'assay', 'assays', 'analysis', 'study', 'studies',
    'human', 'mouse', 'data', 'result', 'results',
    'type', 'types', 'role', 'roles', 'involved',
    'associated', 'related', 'including', 'also',
    'may', 'can', 'could', 'would', 'this', 'that',
    'which', 'where', 'when', 'what', 'how', 'why',
}


class KeywordExtractor:
    """Extract biological keywords from text using KeyBERT or fallback methods."""

    def __init__(self, method: str = 'keybert', top_n: int = 10):
        self.method = method
        self.top_n = top_n
        self.model = None

        if method == 'keybert' and KEYBERT_AVAILABLE:
            # Use a smaller, faster model
            self.model = KeyBERT(model='all-MiniLM-L6-v2')
        elif method == 'spacy' and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Downloading spaCy model...")
                os.system('python -m spacy download en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
        else:
            self.method = 'simple'

    def extract(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        if not text or text.strip() == '':
            return set()

        if self.method == 'keybert' and self.model:
            return self._extract_keybert(text)
        elif self.method == 'spacy' and SPACY_AVAILABLE:
            return self._extract_spacy(text)
        else:
            return self._extract_simple(text)

    def _extract_keybert(self, text: str) -> Set[str]:
        """Extract keywords using KeyBERT."""
        try:
            # Extract keywords with KeyBERT
            keywords = self.model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=self.top_n,
                use_maxsum=True,
                nr_candidates=20
            )
            # Filter and normalize keywords
            extracted = set()
            for kw, score in keywords:
                kw_lower = kw.lower().strip()
                # Skip if in stopwords or too short
                if kw_lower not in BIO_STOPWORDS and len(kw_lower) > 2:
                    extracted.add(kw_lower)
            return extracted
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return self._extract_simple(text)

    def _extract_spacy(self, text: str) -> Set[str]:
        """Extract keywords using spaCy noun chunks."""
        doc = self.nlp(text)
        keywords = set()

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            kw = chunk.text.lower().strip()
            if kw not in BIO_STOPWORDS and len(kw) > 2:
                keywords.add(kw)

        # Also extract named entities
        for ent in doc.ents:
            kw = ent.text.lower().strip()
            if kw not in BIO_STOPWORDS and len(kw) > 2:
                keywords.add(kw)

        return keywords

    def _extract_simple(self, text: str) -> Set[str]:
        """Simple keyword extraction using regex and filtering."""
        # Convert to lowercase and extract words
        text_lower = text.lower()

        # Extract potential biological terms (multi-word and single-word)
        # Pattern for biological terms: words with hyphens, numbers, or Greek letters
        patterns = [
            r'\b[a-z]+[-][a-z0-9]+\b',  # hyphenated terms like "H3K4me1"
            r'\b[a-z]{2,}[0-9]+[a-z]*\b',  # alphanumeric like "BRD2", "CDK7"
            r'\b[a-z]+-[a-z]+-[a-z]+\b',  # multi-hyphenated
            r'\b(?:enhancer|promoter|silencer|repressor|activator)\b',
            r'\b(?:kinase|phosphatase|enzyme|receptor|ligand)\b',
            r'\b(?:transcription|translation|replication|methylation)\b',
            r'\b(?:chromatin|histone|nucleosome|chromosome)\b',
            r'\b(?:pathway|signaling|cascade|network)\b',
            r'\b(?:binding|interaction|complex|domain)\b',
            r'\b(?:embryonic|stem|carcinoma|tumor|cancer)\b',
            r'\b(?:dna|rna|mrna|trna|rrna)\b',
            r'\b(?:starr-seq|chip-seq|atac-seq|faire-seq|dnase)\b',
        ]

        keywords = set()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match not in BIO_STOPWORDS and len(match) > 2:
                    keywords.add(match)

        # Also extract capitalized terms (likely gene/protein names)
        cap_pattern = r'\b[A-Z][A-Z0-9]{1,}[a-z]*\b'
        cap_matches = re.findall(cap_pattern, text)
        for match in cap_matches:
            match_lower = match.lower()
            if match_lower not in BIO_STOPWORDS and len(match) > 1:
                keywords.add(match_lower)

        return keywords


def compute_keyword_f1(
    ref_keywords: Set[str],
    hyp_keywords: Set[str]
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 for keyword sets."""
    if len(hyp_keywords) == 0:
        precision = 0.0
    else:
        precision = len(ref_keywords & hyp_keywords) / len(hyp_keywords)

    if len(ref_keywords) == 0:
        recall = 0.0
    else:
        recall = len(ref_keywords & hyp_keywords) / len(ref_keywords)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def load_predictions(filepath: str) -> Tuple[List[Dict], str]:
    """Load predictions from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    predictions = []

    # Handle nested structure like {"claude-opus-4": {"scores": ..., "predictions": [...]}}
    if 'predictions' in data:
        predictions = data['predictions']
    elif isinstance(data, dict):
        # Check if it's a nested structure with model name as key
        for key, value in data.items():
            if isinstance(value, dict) and 'predictions' in value:
                predictions = value['predictions']
                break
        # If still empty, check if data itself is a list
        if not predictions and isinstance(data, list):
            predictions = data
    elif isinstance(data, list):
        predictions = data

    # Infer model type from filename
    filename = os.path.basename(filepath).lower()
    model_type = 'unknown'
    if 'dnabert2' in filename:
        model_type = 'GeneChat + DNABERT-2'
    elif 'dnabert_s' in filename:
        model_type = 'GeneChat + DNABERT-S'
    elif 'dnabert' in filename:
        model_type = 'GeneChat + DNABERT (6-mer)'
    elif 'hyenadna' in filename or 'hyena' in filename:
        model_type = 'GeneChat + HyenaDNA'
    elif 'no_encoder' in filename:
        model_type = 'No Encoder (Raw Seq)'
    elif 'opus' in filename:
        model_type = 'Claude Opus 4.5'
    elif 'sonnet' in filename:
        model_type = 'Claude Sonnet 4.5'
    elif 'claude' in filename:
        model_type = 'Claude'
    elif 'llama' in filename:
        model_type = 'LLaMA'
    elif 'gemini' in filename:
        model_type = 'Gemini'
    elif 'gpt' in filename:
        model_type = 'GPT-4o'
    elif 'genechat' in filename:
        model_type = 'GeneChat'

    return predictions, model_type


def evaluate_keyword_f1(
    predictions: List[Dict],
    extractor: KeywordExtractor,
    max_samples: int = 200,
    verbose: bool = False
) -> Dict:
    """Evaluate keyword F1 scores for predictions."""

    # Filter for non-empty ground truth
    valid_predictions = [
        p for p in predictions
        if p.get('correct_func', '').strip() != ''
    ]

    # Limit samples
    if len(valid_predictions) > max_samples:
        valid_predictions = valid_predictions[:max_samples]

    print(f"  Evaluating {len(valid_predictions)} samples with non-empty ground truth")

    precisions = []
    recalls = []
    f1s = []

    all_ref_keywords = []
    all_hyp_keywords = []

    for i, pred in enumerate(valid_predictions):
        ref_text = pred.get('correct_func', '')
        hyp_text = pred.get('predict_func', '')

        # Extract keywords
        ref_kw = extractor.extract(ref_text)
        hyp_kw = extractor.extract(hyp_text)

        all_ref_keywords.append(ref_kw)
        all_hyp_keywords.append(hyp_kw)

        # Compute metrics
        p, r, f1 = compute_keyword_f1(ref_kw, hyp_kw)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        if verbose and i < 3:
            print(f"\n  Sample {i+1}:")
            print(f"    Reference keywords: {ref_kw}")
            print(f"    Hypothesis keywords: {hyp_kw}")
            print(f"    Overlap: {ref_kw & hyp_kw}")
            print(f"    P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    # Compute aggregate statistics
    results = {
        'num_samples': len(valid_predictions),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        # Micro-averaged (pooled keywords)
        'precision_micro': None,
        'recall_micro': None,
        'f1_micro': None,
    }

    # Compute micro-averaged metrics
    all_ref = set().union(*all_ref_keywords) if all_ref_keywords else set()
    all_hyp = set().union(*all_hyp_keywords) if all_hyp_keywords else set()

    total_overlap = sum(len(ref & hyp) for ref, hyp in zip(all_ref_keywords, all_hyp_keywords))
    total_hyp = sum(len(hyp) for hyp in all_hyp_keywords)
    total_ref = sum(len(ref) for ref in all_ref_keywords)

    if total_hyp > 0:
        results['precision_micro'] = total_overlap / total_hyp
    else:
        results['precision_micro'] = 0.0

    if total_ref > 0:
        results['recall_micro'] = total_overlap / total_ref
    else:
        results['recall_micro'] = 0.0

    if results['precision_micro'] + results['recall_micro'] > 0:
        results['f1_micro'] = (
            2 * results['precision_micro'] * results['recall_micro'] /
            (results['precision_micro'] + results['recall_micro'])
        )
    else:
        results['f1_micro'] = 0.0

    # Average keywords per sample
    results['avg_ref_keywords'] = np.mean([len(kw) for kw in all_ref_keywords])
    results['avg_hyp_keywords'] = np.mean([len(kw) for kw in all_hyp_keywords])

    return results


def print_results_table(all_results: Dict[str, Dict]):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print("KEYWORD EXTRACTION F1 RESULTS")
    print("="*100)

    # Header
    print(f"{'Model':<35} {'Samples':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} "
          f"{'Prec-μ':>8} {'Rec-μ':>8} {'F1-μ':>8}")
    print("-"*100)

    # Sort by F1 descending
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)

    for model_name, results in sorted_models:
        print(f"{model_name:<35} "
              f"{results['num_samples']:>8} "
              f"{results['precision_mean']:>8.4f} "
              f"{results['recall_mean']:>8.4f} "
              f"{results['f1_mean']:>8.4f} "
              f"{results['precision_micro']:>8.4f} "
              f"{results['recall_micro']:>8.4f} "
              f"{results['f1_micro']:>8.4f}")

    print("="*100)
    print("Note: Prec/Recall/F1 are macro-averaged (per-sample). μ variants are micro-averaged (pooled).")


def save_results(all_results: Dict[str, Dict], output_path: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types
    serializable = {}
    for model, results in all_results.items():
        serializable[model] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in results.items()
        }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def plot_comparison(all_results: Dict[str, Dict], output_path: str, use_micro: bool = False):
    """Create a bar chart comparing models on Precision, Recall, and F1."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Cannot create plot.")
        return

    # Sort models by F1 descending
    suffix = '_micro' if use_micro else '_mean'
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get(f'f1{suffix}', x[1].get('f1_mean', 0)),
        reverse=True
    )
    model_names = [m[0] for m in sorted_models]

    # Extract metrics
    if use_micro:
        precisions = [all_results[m]['precision_micro'] for m in model_names]
        recalls = [all_results[m]['recall_micro'] for m in model_names]
        f1s = [all_results[m]['f1_micro'] for m in model_names]
        metric_type = "Micro-averaged"
    else:
        precisions = [all_results[m]['precision_mean'] for m in model_names]
        recalls = [all_results[m]['recall_mean'] for m in model_names]
        f1s = [all_results[m]['f1_mean'] for m in model_names]
        metric_type = "Macro-averaged"

    # Setup the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar positions
    x = np.arange(len(model_names))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, f1s, width, label='F1', color='#e74c3c', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Keyword Extraction F1 Comparison ({metric_type})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save PDF version
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate gene function predictions using keyword extraction F1'
    )
    parser.add_argument(
        '--results-file', type=str, default=None,
        help='Path to a single results JSON file'
    )
    parser.add_argument(
        '--files', type=str, nargs='+', default=None,
        help='Multiple result files to compare (e.g., --files file1.json file2.json)'
    )
    parser.add_argument(
        '--compare-dir', type=str, default=None,
        help='Directory containing multiple results files to compare'
    )
    parser.add_argument(
        '--output', type=str, default='keyword_f1_results.json',
        help='Output file path for results'
    )
    parser.add_argument(
        '--output-plot', type=str, default=None,
        help='Output file path for comparison plot (e.g., comparison.png)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=200,
        help='Maximum number of samples to evaluate (default: 200)'
    )
    parser.add_argument(
        '--method', type=str, default='keybert',
        choices=['keybert', 'spacy', 'simple'],
        help='Keyword extraction method (default: keybert)'
    )
    parser.add_argument(
        '--top-n', type=int, default=15,
        help='Number of top keywords to extract per text (default: 15)'
    )
    parser.add_argument(
        '--use-micro', action='store_true',
        help='Use micro-averaged metrics for plotting (default: macro-averaged)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed per-sample results'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("KEYWORD EXTRACTION F1 EVALUATION")
    print("="*70)

    # Initialize keyword extractor
    print(f"\nInitializing keyword extractor (method: {args.method})...")
    extractor = KeywordExtractor(method=args.method, top_n=args.top_n)
    print(f"Using extraction method: {extractor.method}")

    all_results = {}

    if args.results_file:
        # Single file evaluation
        print(f"\nLoading predictions from: {args.results_file}")
        predictions, model_type = load_predictions(args.results_file)
        print(f"Model type: {model_type}")
        print(f"Total predictions: {len(predictions)}")

        results = evaluate_keyword_f1(
            predictions, extractor,
            max_samples=args.max_samples,
            verbose=args.verbose
        )
        all_results[model_type] = results

    elif args.files:
        # Compare multiple specified files
        print(f"\nComparing {len(args.files)} result files...")

        for filepath in args.files:
            if not os.path.exists(filepath):
                print(f"\nWarning: File not found: {filepath}")
                continue

            filename = os.path.basename(filepath)
            print(f"\n{'='*50}")
            print(f"Processing: {filename}")

            predictions, model_type = load_predictions(filepath)
            if not predictions:
                print(f"  Skipping - no predictions found")
                continue

            print(f"  Model type: {model_type}")
            print(f"  Total predictions: {len(predictions)}")

            results = evaluate_keyword_f1(
                predictions, extractor,
                max_samples=args.max_samples,
                verbose=args.verbose
            )
            all_results[model_type] = results

    elif args.compare_dir:
        # Compare multiple files in directory
        print(f"\nComparing results in: {args.compare_dir}")

        for filename in sorted(os.listdir(args.compare_dir)):
            if filename.endswith('.json') and 'results_' in filename:
                filepath = os.path.join(args.compare_dir, filename)
                print(f"\n{'='*50}")
                print(f"Processing: {filename}")

                predictions, model_type = load_predictions(filepath)
                if not predictions:
                    print(f"  Skipping - no predictions found")
                    continue

                print(f"  Model type: {model_type}")
                print(f"  Total predictions: {len(predictions)}")

                results = evaluate_keyword_f1(
                    predictions, extractor,
                    max_samples=args.max_samples,
                    verbose=args.verbose
                )
                all_results[model_type] = results
    else:
        print("Error: Specify --results-file, --files, or --compare-dir")
        return

    # Print results table
    print_results_table(all_results)

    # Save results
    save_results(all_results, args.output)

    # Generate plot if requested
    if args.output_plot:
        plot_comparison(all_results, args.output_plot, use_micro=args.use_micro)

    print("\nKeyword F1 evaluation complete!")


if __name__ == '__main__':
    main()
