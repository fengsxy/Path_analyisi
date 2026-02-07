#!/bin/bash
# Evaluate LIFT models (batch mode, load real features once)
#
# Usage:
#   ./scripts/eval_lift.sh                    # Run for all epochs
#   ./scripts/eval_lift.sh 200 400 600        # Run for specific epochs
#
# Output:
#   - results/grid_lift_XXXep.png (9x9 grid)
#   - results/fid_lift_results.csv (summary)
#   - results/real_features.npz (cached, reused)

set -e

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
DEVICE=0
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
echo "LIFT Batch Evaluation"
echo "=========================================="

$PYTHON eval_fid_batch.py \
    --model_type lift \
    --epochs ${EPOCHS[@]} \
    --num_images $NUM_IMAGES \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --output_dir $RESULTS_DIR
