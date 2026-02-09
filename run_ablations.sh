#!/bin/bash

echo "======================================"
echo "RADAR Ablation Study"
echo "Baseline vs Baseline + RADAR Modules"
echo "======================================"
echo ""

DATA_DIR="${1:-./data}"
NUM_EPOCHS="${2:-15}"

echo "Data directory: $DATA_DIR"
echo "Epochs per run: $NUM_EPOCHS"
echo "Backbones: ViT-Small, ResNet-50, Xception, EfficientNet-B0"
echo "Total experiments: 8 (4 backbones × 2 modes)"
echo ""

python 3_train_ablation.py \
  --data_dir "$DATA_DIR" \
  --num_epochs "$NUM_EPOCHS"

echo ""
echo "======================================"
echo "Generating comparison tables..."
echo "======================================"
echo ""

python 4_compare_ablations.py

echo ""
echo "======================================"
echo "All ablations complete!"
echo "======================================"
