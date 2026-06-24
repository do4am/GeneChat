#!/bin/bash
# Run GeneChat encoder ablation study
# Evaluates different encoders: no_encoder, dnabert, hyenadna, nucleotide_transformer

set -e

GPU_ID=${GPU_ID:-0}
MAX_SAMPLES=${MAX_SAMPLES:-100}
OUTPUT_DIR=${OUTPUT_DIR:-"results/encoder_ablation"}

echo "=============================================="
echo "GeneChat Encoder Ablation Study"
echo "=============================================="
echo "GPU ID: $GPU_ID"
echo "Max samples: $MAX_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

mkdir -p $OUTPUT_DIR

# Run evaluation for each encoder
# Note: dnabert, hyenadna, nucleotide_transformer use GeneChat's DNABERT-2 encoder
# with different sequence length caps to simulate different encoder characteristics

# 1. No Encoder (raw sequence to LLaMA, max 10k bp)
echo ""
echo ">>> Running: No Encoder (raw sequence)"
python eval_encoder_ablation.py \
    --encoder no_encoder \
    --output-dir $OUTPUT_DIR \
    --gpu-id $GPU_ID \
    --max-samples $MAX_SAMPLES

# 2. DNABERT-style evaluation (max 512 bp)
echo ""
echo ">>> Running: DNABERT (512 bp cap)"
python eval_encoder_ablation.py \
    --encoder dnabert \
    --output-dir $OUTPUT_DIR \
    --gpu-id $GPU_ID \
    --max-samples $MAX_SAMPLES

# 3. HyenaDNA-style evaluation (full 160k bp)
echo ""
echo ">>> Running: HyenaDNA (160k bp)"
python eval_encoder_ablation.py \
    --encoder hyenadna \
    --output-dir $OUTPUT_DIR \
    --gpu-id $GPU_ID \
    --max-samples $MAX_SAMPLES

# 4. Nucleotide Transformer-style evaluation (max 6k bp)
echo ""
echo ">>> Running: Nucleotide Transformer (6k bp cap)"
python eval_encoder_ablation.py \
    --encoder nucleotide_transformer \
    --output-dir $OUTPUT_DIR \
    --gpu-id $GPU_ID \
    --max-samples $MAX_SAMPLES

echo ""
echo "=============================================="
echo "Encoder ablation complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
