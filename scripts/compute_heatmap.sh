#!/bin/bash
# Step 1: Compute 30x30 error heatmap for LIFT models
#
# Usage:
#   ./scripts/compute_heatmap.sh 800 1000 2000    # Specific epochs
#   ./scripts/compute_heatmap.sh                   # All epochs
#
# Output:
#   - results/heatmap_30_XXXep.pth (error map data)

set -e

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
DEVICE=0
NUM_STEPS=18
RESULTS_DIR="results"

ALL_EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)

if [ $# -gt 0 ]; then
    EPOCHS=("$@")
else
    EPOCHS=("${ALL_EPOCHS[@]}")
fi

echo "=========================================="
echo "Step 1: Compute Heatmaps"
echo "=========================================="
echo "Epochs: ${EPOCHS[*]}"
echo ""

mkdir -p $RESULTS_DIR

for EPOCH in "${EPOCHS[@]}"; do
    echo ""
    echo "=== Epoch $EPOCH ==="

    CKPT="checkpoints/lift_dual_timestep_${EPOCH}ep.pth"
    HEATMAP="$RESULTS_DIR/heatmap_30_${EPOCH}ep.pth"

    if [ ! -f "$CKPT" ]; then
        echo "[Skip] Checkpoint not found: $CKPT"
        continue
    fi

    if [ -f "$HEATMAP" ]; then
        echo "[Skip] Heatmap exists: $HEATMAP"
        continue
    fi

    echo "Computing heatmap..."
    $PYTHON compute_heatmap_30.py \
        --checkpoint $CKPT \
        --output "$RESULTS_DIR/heatmap_30_${EPOCH}ep.png" \
        --num_steps $NUM_STEPS \
        --device $DEVICE
done

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
