# RADAR: Reasoning and Artifact Detection for Deepfake Recognition

Deep learning architecture for deepfake detection using dual artifact detection modules with iterative evidence refinement.

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

```bash
./0_download_combined_dataset.sh
```

The script automatically:
- Downloads the public dataset (no Kaggle authentication needed)
- Installs 7zip for fast multi-threaded extraction
- Shows progress bars during download and extraction
- Creates `./data/combined` with train/val/test splits

## Training

```bash
python 2_train_model.py \
  --data_dir ./data/combined \
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
  --data_dir ./data/combined
```

## Architecture

- **BADM**: Boundary Artifact Detection Module
- **AADM**: Frequency Artifact Detection Module
- **ERM**: Evidence Refinement Module with iterative reasoning

## Results

Training on RADAR Combined Dataset (227k frames) achieves high validation AUC with convergence in approximately 2 hours on A40 GPU.
