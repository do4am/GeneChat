#!/bin/bash
# Run LLM-as-Judge evaluation for GeneChat
# Uses Claude or GPT-4 to score gene descriptions

set -e

# Required: Path to GeneChat evaluation results
RESULTS_FILE=${RESULTS_FILE:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"results/llm_judge"}
JUDGE=${JUDGE:-"claude"}
MAX_SAMPLES=${MAX_SAMPLES:-50}
DELAY=${DELAY:-1.0}

# API Keys (set these or use environment variables)
# ANTHROPIC_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

if [ -z "$RESULTS_FILE" ]; then
    echo "Error: RESULTS_FILE not specified"
    echo ""
    echo "Usage:"
    echo "  RESULTS_FILE=results/evaluation_results.json ./run_llm_judge.sh"
    echo ""
    echo "Or with all options:"
    echo "  RESULTS_FILE=results/eval.json JUDGE=claude MAX_SAMPLES=50 ./run_llm_judge.sh"
    exit 1
fi

echo "=============================================="
echo "LLM-as-Judge Evaluation"
echo "=============================================="
echo "Results file: $RESULTS_FILE"
echo "Judge: $JUDGE"
echo "Max samples: $MAX_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
echo "API delay: ${DELAY}s"
echo "=============================================="

mkdir -p $OUTPUT_DIR

python eval_llm_judge.py \
    --results-file $RESULTS_FILE \
    --output-dir $OUTPUT_DIR \
    --judge $JUDGE \
    --max-samples $MAX_SAMPLES \
    --delay $DELAY

echo ""
echo "=============================================="
echo "LLM-as-Judge evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
