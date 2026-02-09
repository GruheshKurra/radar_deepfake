# RADAR: Reasoning and Artifact Detection for Deepfake Recognition

Deep learning architecture for deepfake detection using dual artifact detection modules with iterative evidence refinement.

**📊 Performance**: Test AUC **95.6%** | Test Accuracy **90.1%**

**🔗 Links**: [GitHub](https://github.com/GruheshKurra/radar_deepfake) | [Dataset](https://www.kaggle.com/datasets/gruheshkurra/radar-deepfake-frames) | [Pre-trained Model](https://github.com/GruheshKurra/radar_deepfake/releases/tag/v1.0)

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
- Downloads the public dataset (no Kaggle authentication needed)
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

Training on RADAR Combined Dataset (227k frames):
- **Test AUC**: 95.6%
- **Test Accuracy**: 90.1%
- **Best Validation AUC**: 94.5% (epoch 14)
- **Training Time**: ~2 hours on A40 GPU

Pre-trained model available in [releases](https://github.com/GruheshKurra/radar_deepfake/releases/tag/v1.0).
