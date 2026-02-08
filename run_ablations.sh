#!/bin/bash

echo "Starting RADAR Ablation Studies"
echo "================================"

DATA_DIR="./data/combined"
OUTPUT_DIR="./outputs"

echo ""
echo "Ablation 1: BADM Only (Boundary artifacts only)"
echo "------------------------------------------------"
python 3_train_ablation.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --experiment_name ablation_badm_only \
  --use_badm true \
  --use_aadm false \
  --num_epochs 30

echo ""
echo "Ablation 2: AADM Only (Frequency artifacts only)"
echo "-------------------------------------------------"
python 3_train_ablation.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --experiment_name ablation_aadm_only \
  --use_badm false \
  --use_aadm true \
  --num_epochs 30

echo ""
echo "Ablation 3: Both modules, no reasoning (Direct fusion)"
echo "-------------------------------------------------------"
python 3_train_ablation.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --experiment_name ablation_no_reasoning \
  --use_badm true \
  --use_aadm true \
  --reasoning_iterations 1 \
  --num_epochs 30

echo ""
echo "Ablation 4: Different reasoning iterations"
echo "-------------------------------------------"
for iters in 1 2 3 4 5; do
  echo "Testing with $iters iterations..."
  python 3_train_ablation.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --experiment_name ablation_iters_${iters} \
    --use_badm true \
    --use_aadm true \
    --reasoning_iterations $iters \
    --num_epochs 30
done

echo ""
echo "================================"
echo "All ablations complete!"
echo "Results saved in: $OUTPUT_DIR"
