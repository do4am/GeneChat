#!/bin/bash
# Run Semantic Similarity Evaluation for Gene Description Generation
# Supports: BERTScore, SimCSE, BLEU, METEOR

set -e

GPU_ID=${GPU_ID:-0}
RESULTS_FILE=${RESULTS_FILE:-""}
COMPARE_DIR=${COMPARE_DIR:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"results/semantic_eval"}
MAX_SAMPLES=${MAX_SAMPLES:-""}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "=============================================="
echo "Semantic Similarity Evaluation"
echo "=============================================="
echo "GPU ID: $GPU_ID"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "Metrics: BLEU, METEOR, BERTScore, SimCSE"
echo "=============================================="

mkdir -p $OUTPUT_DIR

# Build command
CMD="python eval_semantic_similarity.py --gpu-id $GPU_ID --output-dir $OUTPUT_DIR --batch-size $BATCH_SIZE"

if [ -n "$RESULTS_FILE" ]; then
    echo "Evaluating single file: $RESULTS_FILE"
    CMD="$CMD --results-file $RESULTS_FILE"
elif [ -n "$COMPARE_DIR" ]; then
    echo "Comparing all files in: $COMPARE_DIR"
    CMD="$CMD --compare-dir $COMPARE_DIR"
else
    echo ""
    echo "Usage:"
    echo "  Single file:   RESULTS_FILE=results/genechat.json ./run_semantic_eval.sh"
    echo "  Compare all:   COMPARE_DIR=results/ ./run_semantic_eval.sh"
    echo ""
    echo "Optional:"
    echo "  GPU_ID=1"
    echo "  MAX_SAMPLES=100"
    echo "  BATCH_SIZE=16"
    exit 1
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=============================================="
echo "Semantic evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
