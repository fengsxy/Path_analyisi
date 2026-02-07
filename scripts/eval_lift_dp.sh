#!/bin/bash
# Step 3: Generate images with DP paths and compute FID
#
# Usage:
#   ./scripts/eval_lift_dp.sh 800 1000 2000    # Specific epochs
#   ./scripts/eval_lift_dp.sh                   # All epochs
#
# Output:
#   - results/grid_lift_dp_64_XXXep.png (9x9 grid)
#   - results/grid_lift_dp_total_XXXep.png (9x9 grid)
#   - results/fid_lift_dp_results.csv (FID scores)

set -e

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
DEVICE=2
NUM_IMAGES=15803
NUM_STEPS=18
BATCH_SIZE=64
RESULTS_DIR="results"

ALL_EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)

if [ $# -gt 0 ]; then
    EPOCHS=("$@")
else
    EPOCHS=("${ALL_EPOCHS[@]}")
fi

echo "=========================================="
echo "Step 3: Generate Images & Compute FID"
echo "=========================================="
echo "Epochs: ${EPOCHS[*]}"
echo ""

$PYTHON eval_lift_dp.py \
    --epochs ${EPOCHS[@]} \
    --num_images $NUM_IMAGES \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --output_dir $RESULTS_DIR
