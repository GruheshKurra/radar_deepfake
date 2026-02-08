# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RADAR (Reasoning and Artifact Detection for Deepfake Recognition) is a deep learning model for deepfake detection. It uses dual artifact detection modules (boundary and frequency) with iterative evidence refinement through a reasoning module. This is a research paper implementation.

## Commands

### Training
```bash
conda activate ml
python 2_train_model.py --data_dir ./data/combined --experiment_name radar_training --num_epochs 30 --batch_size 128
python 2_train_model.py --data_dir ./data/combined --batch_size 64 --learning_rate 0.0005 --freeze_backbone_epochs 3
```

### Ablation Studies
```bash
python 3_train_ablation.py --data_dir ./data/combined --experiment_name ablation_badm_only --use_badm true --use_aadm false
python 3_train_ablation.py --data_dir ./data/combined --experiment_name ablation_aadm_only --use_badm false --use_aadm true
python 3_train_ablation.py --data_dir ./data/combined --experiment_name ablation_no_reasoning --reasoning_iterations 1
./run_ablations.sh
```

### Compare and Visualize
```bash
python 4_compare_ablations.py --output_dir ./outputs
python 5_visualize_attention.py --checkpoint ./outputs/radar_training/best.pth --data_dir ./data/combined
```

### Dataset Download
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
./0_download_combined_dataset.sh
```

## Architecture

### Three-Component Design
1. **BADM** (`src/method/boundary.py`): Computes Sobel gradients, encodes via deeper conv pyramid (1→32→64→128→256) with residual blocks and spatial attention, fuses with ViT patch features (mean+max aggregation). Outputs 128-dim evidence.
2. **AADM** (`src/method/frequency.py`): Multi-scale FFT at 3 frequency bands (cutoffs H/16, H/8, H/4) producing 3-channel input. Encodes via conv pyramid (3→48→96→192→256) with residual blocks, fuses with ViT CLS token. Outputs 128-dim evidence.
3. **ERM** (`src/method/reasoning.py`): Cross-attention + GRU + LayerNorm + FFN per iteration. Iteratively refines evidence over 3 iterations with feed-forward refinement at each step.

### Forward Pass Flow (`src/method/radar.py`)
```
Image → ViT-Small (vit_small_patch16_224, pretrained)
  ├─ CLS token (384-dim) → AADM → 128-dim evidence
  └─ Patch tokens (196×384) → BADM → 128-dim evidence
      ↓
  Evidence stack (B, 2, 128)
      ↓
  ├─ Reasoning path: ERM iterative refinement → logit_reasoning
  └─ External path: concat evidence → MLP classifier → logit_external
      ↓
  Gating: final = α·reasoning + (1-α)·external
  α = sigmoid(learnable_param), initialized at 0.5
```

### Backbone Freezing
The ViT backbone is frozen for the first N epochs (default 3) to let BADM/AADM/ERM warm up without destroying pretrained features. After unfreezing, backbone trains at 10x lower learning rate than heads.

### Loss Function (`src/method/loss.py`)
Four components combined:
- `lambda_main=1.0`: BCE on final prediction (label smoothing 0.05)
- `lambda_branch=0.3`: BCE on individual BADM/AADM branch predictions
- `lambda_orthogonal=0.01`: Hinge loss on cosine similarity between BADM/AADM evidence (margin=0.1)
- `lambda_deep_supervision=0.05`: Weighted BCE on each ERM iteration's intermediate predictions

### Key Interfaces
- Model: `forward(x, freq_cached, sobel_cached, use_badm, use_aadm) → Dict` with keys: logit, prob, reasoning_logit, reasoning_prob, external_logit, gating_alpha, badm_evidence, aadm_evidence, attention_history, iteration_logits, iteration_probs, convergence_delta
- Loss: `forward(outputs, labels, use_badm, use_aadm) → Dict` with keys: total, main, branch, orthogonal, deep_supervision
- Baselines: `forward(x, freq_cached, sobel_cached) → Dict` with keys: logit, prob
- RADAR model methods: `freeze_backbone()`, `unfreeze_backbone()`, `get_backbone_params()`, `get_head_params()`

## Data Pipeline

### Dataset Structure
Supports two layouts detected by `is_presplit_dataset()` in `src/data/splits.py`:
- **Pre-split**: `data_dir/{train,val,test}/{real,fake}/*.{jpg,png,jpeg}`
- **Single-domain**: `data_dir/{domain_name}/{real,fake}/*.{jpg,png,jpeg}` (auto-split 80/10/10)

### Transforms (`src/data/dataset.py`)
- Train: Resize(224) → HFlip(0.5) → ShiftScaleRotate(0.3) → ColorJitter(0.3) → CoarseDropout(0.2) → ImageNet normalize
- Val/Test: Resize(224) → ImageNet normalize
- GaussianBlur and ImageCompression are intentionally excluded to preserve boundary/frequency artifacts

## Training Details

### Optimizer/Scheduler
AdamW with differential learning rates: heads at base LR, backbone at LR*0.1. OneCycleLR with cosine annealing and 10% warmup. AMP via GradScaler on CUDA. Gradient clipping at norm=1.0.

### Monitoring Metrics
- `convergence_delta`: |prob_iter3 - prob_iter2|, should be 0.001-0.005
- `gating_alpha`: evolves from 0.5, indicates reasoning vs external preference
- `skipped_batches`: should be 0, indicates NaN/Inf instability
- `grad_norm`: typical 0.5-2.0, >10 indicates instability

### Output Structure
```
outputs/<experiment_name>/
├── best.pth       (model + optimizer state dict)
├── config.yaml    (training config snapshot)
└── metrics.json   (history + test results)
```

## Code Style
- No comments in code
- Minimal, functional implementations
- All modules return dictionaries for flexible monitoring

## File Naming Convention
- `0_*.sh`: Data download/setup
- `2_*.py`: Training scripts
- `3_*.py`: Ablation experiments
- `4_*.py`: Analysis/comparison
- `5_*.py`: Visualization

## Baselines (`src/baselines/`)
ViT-Small, ResNet50, Xception, EfficientNet-B0. All pretrained, single logit output. Same forward interface: `forward(x, freq_cached, sobel_cached) → {logit, prob}`.

## Analysis Utilities (`src/analysis/`)
- `stats.py`: Bootstrap AUC confidence intervals (2000 iterations), DeLong test for AUC comparison
- `aggregate.py`: Multi-seed result aggregation (mean/std/min/max)
- `plots.py`: Training curves, ROC curves, ablation bar charts
