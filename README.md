# Dual-Stream Artifact Detection with Iterative Evidence Refinement for Frame-Level Deepfake Recognition

Code for the paper accepted at **IEEE NEPCON 2026** (Kurra, Pagadala, Batumalay, Boddapati). Internal code name: RADAR.

A ViT-Small backbone with two artifact streams, a Boundary Artifact Detection Module (Sobel edges) and a frequency Artifact Detection Module (three-band FFT), kept apart by an orthogonality penalty, followed by an Evidence Refinement Module that runs three cross-attention iterations with GRU state updates.

**Headline result (video-disjoint evaluation): 96.3% AUC overall, 87.1% AUC on Celeb-DF.** See [Results](#results) for why this, and not the full-split 95.6%, is the figure to cite.

**Links**: [Dataset (Kaggle)](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames) | [Pre-trained checkpoint, config and metrics (release v1.0)](https://github.com/GruheshKurra/radar_deepfake/releases/tag/v1.0)

## Setup

### Quick Setup (Recommended)

Run the setup script to automatically create and configure your conda environment:

```bash
./setup_env.sh
```

This will:
- Create a conda environment named `ml` with Python 3.11
- Install PyTorch with appropriate acceleration (MPS for M1/M2 Macs, CUDA for others)
- Install all dependencies from `requirements.txt`

### Manual Setup

If you prefer to set up manually:

```bash
# Create and activate conda environment
conda create -n ml python=3.11 -y
conda activate ml

# Install dependencies
pip install -r requirements.txt
```

### Activating the Environment

Before running any scripts:

```bash
conda activate ml
```

## Dataset

Download RADAR Combined Dataset (227k frames, public):

**Kaggle Dataset**: [radar-deepfake-frames](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames)

```bash
./0_download_combined_dataset.sh
```

The script automatically:
- Downloads the public dataset (about 14.4 GB). If the direct link returns 404, sign in to Kaggle and download from the dataset page, or use the Kaggle CLI with an API token
- No extraction by default (train from ZIP directly)
- For fast extraction: `python 1_fast_extract.py --num_workers 96`
- Creates `./data/{train,val,test}` structure when extracted

Or download directly:
```bash
curl -L -o ~/Downloads/radar-deepfake-frames.zip \
  https://www.kaggle.com/api/v1/datasets/download/gruheshkurra/radar-deepfake-frames
```

## Training

```bash
# From extracted directory
python 2_train_model.py \
  --data_dir ./data \
  --experiment_name radar_training \
  --num_epochs 30 \
  --batch_size 128

# Or directly from ZIP (no extraction)
python 2_train_model.py \
  --data_dir ./data/radar-deepfake-frames.zip \
  --experiment_name radar_training \
  --num_epochs 30 \
  --batch_size 128
```

## Ablation Studies

```bash
./run_ablations.sh

python 4_compare_ablations.py
```

## Visualization

```bash
python 5_visualize_attention.py \
  --checkpoint ./outputs/radar_training/best.pth \
  --data_dir ./data
```

## Architecture

- **BADM**: Boundary Artifact Detection Module
- **AADM**: Frequency Artifact Detection Module
- **ERM**: Evidence Refinement Module with iterative reasoning

## Results

| Evaluation | Test frames | AUC | Accuracy |
|---|---|---|---|
| **Video-disjoint subset** (FaceForensics++ and Celeb-DF test frames whose source videos never appear in training) | 5,680 | **96.3%** | |
| Celeb-DF portion of the video-disjoint subset | | **87.1%** | |
| Full held-out frame split (random frame-level split, seed 42) | | 95.6% | 90.1% |

### Leakage audit: why the video-disjoint figure is the one to cite

The frame-level split in `src/data/splits.py` shuffles individual frames, so frames from the same source video can land in both training and test. An audit of the test set found that **52.9% of all test frames share a source video with training frames**, rising to **97.3% for WildDeepfake (6,588 of 6,768 frames)**. Frames from a video the model has already seen make the full-split number optimistic, so the paper reports the video-disjoint subset as its main result. The subset is drawn from the existing held-out test set, and the model was not retrained.

This is a single-run study. Baselines and ablations are future work; the ablation scripts in this repository are provided but their results are not reported in the paper.

### Released files

- `best.pth`, `config.yaml` and `metrics.json` in [release v1.0](https://github.com/GruheshKurra/radar_deepfake/releases/tag/v1.0). `metrics.json` holds the full-split test metrics (AUC 0.956, accuracy 0.901) and the per-epoch training history; best validation AUC 94.5% at epoch 14 of 15. Training took about 2 hours on one A40 GPU.
- The video-disjoint audit script will be added to `src/analysis/` alongside the source-video mapping it needs.
