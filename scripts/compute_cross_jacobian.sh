#!/bin/bash
# Compute cross-scale Jacobian analysis for LIFT models
#
# Usage:
#   ./scripts/compute_cross_jacobian.sh 400 1000 2000   # Specific epochs
#   ./scripts/compute_cross_jacobian.sh                  # All epochs
#   ./scripts/compute_cross_jacobian.sh --ema 400 1000   # EMA weights
#
# Output:
#   - results/cross_jacobian_XXXep.pth (4 error matrices)
#   - results/cross_jacobian_XXXep.png (2×2 heatmap panel)

set -e

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
DEVICE=0
RESULTS_DIR="results"

ALL_EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)

# Parse --ema flag
EMA_FLAG=""
EMA_SUFFIX=""
EPOCHS=()
for arg in "$@"; do
    if [ "$arg" = "--ema" ]; then
        EMA_FLAG="--ema"
        EMA_SUFFIX="_ema"
    else
        EPOCHS+=("$arg")
    fi
done

if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=("${ALL_EPOCHS[@]}")
fi

echo "=========================================="
echo "Cross-Scale Jacobian Analysis"
echo "=========================================="
echo "Epochs: ${EPOCHS[*]}"
echo "EMA: ${EMA_FLAG:-no}"
echo ""

mkdir -p $RESULTS_DIR

for EPOCH in "${EPOCHS[@]}"; do
    echo ""
    echo "=== Epoch $EPOCH ==="

    if [ -n "$EMA_FLAG" ]; then
        CKPT="checkpoints/lift_ema_${EPOCH}ep.pth"
    else
        CKPT="checkpoints/lift_dual_timestep_${EPOCH}ep.pth"
    fi

    OUTPUT="$RESULTS_DIR/cross_jacobian${EMA_SUFFIX}_${EPOCH}ep.png"

    if [ ! -f "$CKPT" ]; then
        echo "[Skip] Checkpoint not found: $CKPT"
        continue
    fi

    if [ -f "${OUTPUT%.png}.pth" ]; then
        echo "[Skip] Results exist: ${OUTPUT%.png}.pth"
        continue
    fi

    echo "Computing cross-Jacobian heatmap..."
    $PYTHON compute_cross_jacobian.py \
        --checkpoint $CKPT \
        --output "$OUTPUT" \
        --device $DEVICE \
        $EMA_FLAG
done

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
