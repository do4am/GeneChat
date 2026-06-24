# GeneChat: Multi-Modal Large Language Model Enables Gene Function Prediction

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.06.05.658031-b31b1b)](https://www.biorxiv.org/content/10.1101/2025.06.05.658031v1)

GeneChat is a multi-modal large language model that predicts gene function descriptions from genomic sequences. It combines a DNA/mRNA sequence encoder with a large language model (Vicuna-13B), bridged by a linear adaptor trained end-to-end on (sequence, prompt, answer) triplets.

> **Repository history:** This repository builds on the original GeneChat implementation by
> [Shashi-Sekar](https://github.com/Shashi-Sekar/GeneChat). The `main` branch preserves that
> original codebase; the `mrna-dev` branch extends it with the GeneChat-mRNA experiments
> described below.

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
Build using the data preparation scripts:
```bash
python data_prep/prepare_human_mouse_data.py   # download and align HM sequences
python data_prep/prepare_mrna_data.py          # filter, tokenize, train/valid split
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
python inference/inference_all.py --cfg-path configs/genechat_eval.yaml

# Aspect-based evaluation (10 targeted questions per gene)
python inference/inference_aspect.py --cfg-path configs/genechat_eval_dnabert2_stage2.yaml

# SimCSE semantic similarity scoring
python evaluation/eval_semantic_similarity.py --results_file <path/to/results.json>

# GO functional classification
python evaluation/eval_go_classification.py --results_file <path/to/results.json>
```

## Reproduce Figures

All plotting scripts write to `Report/fig/` (not committed). Run from the repo root:

```bash
python visualization/plot_aspect_evaluation.py    # Fig: single-question vs. aspect-based SimCSE
python visualization/plot_stage2_convergence.py   # Fig: Stage-2 training convergence
python visualization/plot_mrna_metrics.py         # Fig: BLEU + SimCSE multi-metric bar chart
python visualization/plot_go_classification.py    # Fig: GO classification accuracy
python visualization/plot_ablation_studies.py     # Fig: encoder and adaptor ablations
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
├── data_prep/              # Data download, cleaning, and formatting scripts
├── evaluation/             # Evaluation scripts and shell launchers
├── visualization/          # Figure generation scripts (outputs to Report/fig/)
├── inference/              # Batch inference scripts
├── finetune.sh             # Training entry point
├── demo.sh                 # Interactive demo
├── train_esm.py            # ESM-2 encoder training
└── environment.yml         # Conda environment
```

## Report

The full technical report covering methodology, experiments, and analysis is available on Overleaf:
**[https://www.overleaf.com/project/696898fa315570270f8cdb18](https://www.overleaf.com/project/696898fa315570270f8cdb18)**

The preprint is also available on bioRxiv:
**[https://www.biorxiv.org/content/10.1101/2025.06.05.658031v1](https://www.biorxiv.org/content/10.1101/2025.06.05.658031v1)**

## Conclusions and Future Directions

**What we found:**
GeneChat demonstrates that genomic sequence encoders can be coupled with large language models to produce meaningful natural-language descriptions of gene function. The original model (DNABERT-2 encoder, NCBI 50k-gene corpus) achieves a SimCSE score of 0.860 on the test set. The GeneChat-mRNA extension, trained on a human-mouse mRNA dataset with a two-stage curriculum and selective-layer LoRA, reaches SimCSE 0.749 (DNABERT-2) and 0.642 (NT-v2). Aspect-based evaluation — decomposing gene function into 10 targeted questions — reveals that the performance ceiling is primarily set by the encoder's ability to distinguish similar sequences, rather than the LLM's language generation capacity.

**Limitations of the current work:**
- The mRNA encoder (DNABERT-2 / NT-v2) was trained on a relatively small HM dataset, limiting generalization to less-studied genes.
- The linear adaptor is a weak bridge; richer cross-attention or MLP projections may better transfer sequence representations into LLM embedding space.
- Evaluation relies on SimCSE and BLEU, which capture surface-level semantic similarity but do not directly measure biological accuracy.

**Suggested directions for future work:**
- **Stronger encoders:** Replace DNABERT-2 with larger or more recent DNA/RNA foundation models (e.g., HyenaDNA, Evo, RNA-FM) and evaluate whether the encoder bottleneck shifts.
- **Richer adaptors:** Replace the linear projection with a multi-layer MLP or cross-attention bridge to improve encoder-to-LLM alignment.
- **Larger and more diverse training data:** Expand beyond the HM mRNA dataset to include multi-species transcriptomes and non-coding RNA families.
- **Biological evaluation metrics:** Complement SimCSE with GO-term prediction accuracy, protein interaction recall, or expert human evaluation to validate functional correctness.
- **Preference-based fine-tuning:** The DPO and REINFORCE task variants included in this codebase (`genechat/tasks/protein_text_dpo.py`) are ready to use once preference data (e.g., expert-ranked gene descriptions) becomes available.

## Acknowledgements

- [Shashi-Sekar/GeneChat](https://github.com/Shashi-Sekar/GeneChat) — original GeneChat implementation
- [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2)
- [Nucleotide Transformer (NT-v2)](https://github.com/instadeepai/nucleotide-transformer)
- [MiniGPT-4](https://minigpt-4.github.io/)
- [Lavis](https://github.com/salesforce/LAVIS)
- [Vicuna](https://github.com/lm-sys/FastChat)

## License

[BSD 3-Clause License](LICENSE.md)
