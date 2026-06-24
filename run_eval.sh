#!/bin/bash

# Evaluation script for GeneChat
# Edit the paths below to match your data locations

# GPU to use
GPU_ID=0

# Path to your checkpoint (already configured in genechat_eval.yaml)
CHECKPOINT="/home/namdo/applications/GeneChat/exon_count/data/checkpoints/20251216095/checkpoint_35000.pth"

# TODO: Update these paths to your actual test data locations
TEST_SEQ_PATH="/path/to/your/test_set/seq.json"
TEST_QA_PATH="/path/to/your/test_set/qa_summary_rule.json"

# Output path
OUTPUT_PATH="results/evaluation_checkpoint_35000.json"

# Configuration file
CONFIG_PATH="configs/genechat_eval.yaml"

echo "========================================"
echo "GeneChat Evaluation Script"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Test sequences: $TEST_SEQ_PATH"
echo "Test QA data: $TEST_QA_PATH"
echo "Output: $OUTPUT_PATH"
echo "========================================"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU_ID python run_evaluation.py \
    --cfg-path $CONFIG_PATH \
    --gpu-id 0 \
    --test-seq-path $TEST_SEQ_PATH \
    --test-qa-path $TEST_QA_PATH \
    --output-path $OUTPUT_PATH \
    --batch-size 100

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_PATH"
echo "========================================"
