# GeneChat: Multi-Modal Large Language Model Enables Gene Function Prediction

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.06.05.658031-b31b1b)](https://www.biorxiv.org/content/10.1101/2025.06.05.658031v1)

GeneChat is a multi-modal large language model that predicts gene function descriptions from genomic sequences. It combines a DNA/mRNA sequence encoder with a large language model (Vicuna-13B), bridged by a linear adaptor trained end-to-end on (sequence, prompt, answer) triplets.

![overview](fig/GeneChat.png)

## Architecture

| Component | Original GeneChat | GeneChat-mRNA |
|-----------|-------------------|---------------|
| Encoder | DNABERT-2 | DNABERT-2 or NT-v2 |
| Adaptor | Linear projection | Linear projection |
| LLM | Vicuna-13B-v1.5 | Vicuna-13B-v1.5 |
| LoRA rank/alpha | 8 / 16 (q, v) | 16 / 32 (q, k, v, o; layers 5–10, 32–39) |
| Training data | 50,248-gene NCBI | Human–Mouse (HM) mRNA |
| SimCSE (test) | 0.860 | 0.749 (DNABERT-2), 0.642 (NT-v2) |

## Getting Started

### Installation

```bash
git clone https://github.com/do4am/GeneChat.git
cd GeneChat
conda env create -f environment.yml
conda activate genechat
```

Verify: `python -c "import torch; print(torch.__version__)"`.
You need at least **70 GB GPU memory** for training, 40 GB for inference.

### Dataset

**Original GeneChat dataset (~51K NCBI genes)**
Download from [Google Drive](https://drive.google.com/drive/folders/1g0Pe0HxfzdhXWbG54rkd-Iya7c6wYZdO?usp=sharing) and place as `train_set/` and `test_set/`.

**GeneChat-mRNA dataset (Human–Mouse mRNA)**
Build using the preparation scripts:
```bash
python prepare_human_mouse_data.py   # download and align HM sequences
python prepare_mrna_data.py          # filter, tokenize, train/valid split
```

> **Cluster storage (A00 GPU cluster):** Pre-processed datasets, trained checkpoints, and
> evaluation results for both the original GeneChat and GeneChat-mRNA experiments are stored
> on the A00 cluster at:
> ```
> /data2/genechat/
> ├── train_set/              # Original NCBI training data
> ├── valid_set/              # Validation split
> ├── train_set_mrna/         # HM mRNA training data (Stage-1 and Stage-2)
> ├── checkpoints/            # Saved model checkpoints
> └── result_mrna/            # GeneChat-mRNA evaluation outputs
> ```
> If you have access to the A00 cluster, copy directly instead of re-running data preparation:
> ```bash
> scp -r <user>@a00-cluster:/data2/genechat/train_set ./train_set
> ```

### Pretrained Weights

Download Vicuna-13B-v1.5 from [Hugging Face](https://huggingface.co/lmsys/vicuna-13b-v1.5) and set the path in your config:
```yaml
# configs/genechat_stage1.yaml
model:
  llama_model: "/path/to/vicuna-13b-v1.5"
```

Stage-1 GeneChat checkpoint (NCBI, 47k genes): [Google Drive](https://drive.google.com/drive/folders/1AaSzc9nlh_kfOJDuhLBfDHo3pGKrcKAE?usp=sharing)

## Training

### Original GeneChat (NCBI dataset)

```bash
# Stage 1: train adaptor only
bash finetune.sh --cfg-path configs/genechat_stage1.yaml

# Stage 2: LoRA fine-tune encoder + adaptor
bash finetune.sh --cfg-path configs/genechat_stage2.yaml

# Stage 3: instruction fine-tuning (optional)
bash finetune.sh --cfg-path configs/genechat_stage3.yaml
```

### GeneChat-mRNA (HM mRNA dataset)

```bash
# DNABERT-2 encoder
bash finetune.sh --cfg-path configs/genechat_stage1_mrna.yaml
bash finetune.sh --cfg-path configs/genechat_stage2_dnabert2.yaml

# NT-v2 encoder
bash finetune.sh --cfg-path configs/genechat_stage1_ntv2.yaml
bash finetune.sh --cfg-path configs/genechat_stage2_ntv2.yaml
```

Selective-layer LoRA (layers 5–10 and 32–39 only) is configured inside each Stage-2 YAML. The manual gradient-freezing logic lives in [genechat/models/genechat.py](genechat/models/genechat.py).

## Evaluation

```bash
# Interactive demo
bash demo.sh

# Batch inference (single-question)
python inference_all.py --cfg-path configs/genechat_eval.yaml

# Aspect-based evaluation (10 targeted questions per gene)
python inference_aspect.py --cfg-path configs/genechat_eval_dnabert2_stage2.yaml

# SimCSE semantic similarity scoring
python eval_semantic_similarity.py --results_file <path/to/results.json>

# GO functional classification
python eval_go_classification.py --results_file <path/to/results.json>
```

## Reproduce Figures

All plotting scripts write to `Report/fig/` (not committed). Run from the repo root:

```bash
python plot_aspect_evaluation.py      # Fig: single-question vs. aspect-based SimCSE
python plot_stage2_convergence.py     # Fig: Stage-2 training convergence
python plot_mrna_metrics.py           # Fig: BLEU + SimCSE multi-metric bar chart
python plot_go_classification.py      # Fig: GO classification accuracy
python plot_ablation_studies.py       # Fig: encoder and adaptor ablations
```

## Project Structure

```
GeneChat/
├── genechat/               # Core model library
│   ├── models/             # GeneChat model, gene encoder, adaptor
│   ├── datasets/           # Dataset builders and loaders
│   ├── runners/            # Training loop and checkpoint management
│   └── tasks/              # Task-specific forward/eval logic
├── configs/                # Training and evaluation YAML configs
├── prepare_*.py            # Data preparation scripts
├── inference_*.py          # Inference scripts
├── eval_*.py               # Evaluation scripts
├── plot_*.py               # Figure generation scripts
├── finetune.sh             # Training entry point
└── environment.yml         # Conda environment
```

## Acknowledgements

- [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2)
- [Nucleotide Transformer (NT-v2)](https://github.com/instadeepai/nucleotide-transformer)
- [MiniGPT-4](https://minigpt-4.github.io/)
- [Lavis](https://github.com/salesforce/LAVIS)
- [Vicuna](https://github.com/lm-sys/FastChat)

## License

[BSD 3-Clause License](LICENSE.md)
