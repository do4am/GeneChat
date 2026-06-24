#!/bin/bash
# Run GeneChat sequence length ablation study
# This script evaluates BLEU and METEOR scores at different sequence lengths

set -e

# Default parameters
GPU_ID=${GPU_ID:-0}
MAX_SAMPLES=${MAX_SAMPLES:-100}
OUTPUT_DIR=${OUTPUT_DIR:-"results/seq_length_ablation"}

# Sequence lengths to evaluate (in base pairs)
# Default: 1K, 5K, 10K, 25K, 50K, 100K, 160K
SEQ_LENGTHS=${SEQ_LENGTHS:-"1000 5000 10000 25000 50000 100000 160000"}

# Data paths (remote server paths)
CFG_PATH=${CFG_PATH:-"configs/genechat_eval.yaml"}
TEST_SEQ_PATH=${TEST_SEQ_PATH:-"/home/namdo/applications/GeneChat/exon_count/data/train_with_names/test_set/seq.json"}
TEST_QA_PATH=${TEST_QA_PATH:-"/home/namdo/applications/GeneChat/exon_count/data/train_with_names/test_set/qa_summary_rule.json"}

echo "=============================================="
echo "GeneChat Sequence Length Ablation Study"
echo "=============================================="
echo "GPU ID: $GPU_ID"
echo "Max samples: $MAX_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
echo "Sequence lengths: $SEQ_LENGTHS"
echo "Config: $CFG_PATH"
echo "Test sequences: $TEST_SEQ_PATH"
echo "Test QA: $TEST_QA_PATH"
echo "=============================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
python eval_seq_length_ablation.py \
    --cfg-path $CFG_PATH \
    --test-seq-path $TEST_SEQ_PATH \
    --test-qa-path $TEST_QA_PATH \
    --seq-lengths $SEQ_LENGTHS \
    --output-dir $OUTPUT_DIR \
    --gpu-id $GPU_ID \
    --max-samples $MAX_SAMPLES \
    --num-beams 4

echo ""
echo "=============================================="
echo "Ablation study complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
