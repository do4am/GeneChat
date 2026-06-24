#!/bin/bash
# Run GeneChat robustness testing
# Tests: original, mutations (1%, 5%, 10%), N-masking, truncation, reverse, shuffle

set -e

GPU_ID=${GPU_ID:-0}
MAX_SAMPLES=${MAX_SAMPLES:-100}
OUTPUT_DIR=${OUTPUT_DIR:-"results/robustness"}

echo "=============================================="
echo "GeneChat Robustness Testing"
echo "=============================================="
echo "GPU ID: $GPU_ID"
echo "Max samples: $MAX_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

mkdir -p $OUTPUT_DIR

# Run all perturbation tests
python eval_robustness.py \
    --run-all \
    --output-dir $OUTPUT_DIR \
    --gpu-id $GPU_ID \
    --max-samples $MAX_SAMPLES

echo ""
echo "=============================================="
echo "Robustness testing complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
