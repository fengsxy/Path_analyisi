#!/bin/bash
# Evaluate with 100×100 timestep-space heatmap and DDIM-aware DP paths
#
# Usage:
#   ./scripts/eval_heatmap_100.sh              # Run for all epochs
#   ./scripts/eval_heatmap_100.sh 1000         # Run for specific epoch
#   ./scripts/eval_heatmap_100.sh 1000 1200    # Run for multiple epochs
#
# This script:
#   1. Computes 100×100 error heatmap in timestep space
#   2. Computes DP paths with DDIM step constraints
#   3. Generates images using the new DP paths
#   4. Computes FID scores

set -e

# Configuration
PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
DEVICE=0
NUM_IMAGES=1000
NUM_STEPS=50
BATCH_SIZE=32
NUM_POINTS=100  # 100×100 heatmap
RESULTS_DIR="results"
FID_REAL_DIR="results/fid_real"

# All epochs to evaluate
ALL_EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)

# Parse command line arguments
if [ $# -gt 0 ]; then
    EPOCHS=("$@")
else
    EPOCHS=("${ALL_EPOCHS[@]}")
fi

echo "=========================================="
echo "100×100 Heatmap Evaluation (Timestep Space)"
echo "=========================================="
echo "Epochs to evaluate: ${EPOCHS[*]}"
echo "Grid size: ${NUM_POINTS}×${NUM_POINTS}"
echo "Generation steps: ${NUM_STEPS}"
echo ""

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p figures

# Prepare real images
prepare_real_images() {
    if [ -d "$FID_REAL_DIR" ] && [ "$(ls -A $FID_REAL_DIR 2>/dev/null | wc -l)" -ge "$NUM_IMAGES" ]; then
        echo "[Skip] Real images already prepared"
    else
        echo "[Step 0] Preparing real images for FID..."
        $PYTHON prepare_fid_real.py \
            --output_dir $FID_REAL_DIR \
            --num_images $NUM_IMAGES
    fi
}

# Initialize CSV
init_csv() {
    CSV_FILE="$RESULTS_DIR/fid_results_heatmap100.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo "Epoch,Baseline,LIFT_Diagonal,LIFT_DP64_100,LIFT_DP_Total_100" > $CSV_FILE
        echo "Created: $CSV_FILE"
    fi
}

# Evaluate single epoch
evaluate_epoch() {
    local EPOCH=$1
    echo ""
    echo "=========================================="
    echo "Evaluating Epoch $EPOCH (100×100 heatmap)"
    echo "=========================================="

    CHECKPOINT_BASELINE="checkpoints/baseline_${EPOCH}ep.pth"
    CHECKPOINT_LIFT="checkpoints/lift_dual_timestep_${EPOCH}ep.pth"
    HEATMAP_FILE="$RESULTS_DIR/heatmap_timestep_${EPOCH}ep.pth"

    # Check checkpoints
    if [ ! -f "$CHECKPOINT_BASELINE" ]; then
        echo "[Warning] Baseline checkpoint not found: $CHECKPOINT_BASELINE"
        return 1
    fi
    if [ ! -f "$CHECKPOINT_LIFT" ]; then
        echo "[Warning] LIFT checkpoint not found: $CHECKPOINT_LIFT"
        return 1
    fi

    # Step 1: Compute 100×100 heatmap (if not exists)
    if [ ! -f "$HEATMAP_FILE" ]; then
        echo "[Step 1] Computing 100×100 error heatmap..."
        $PYTHON compute_heatmap_timestep.py \
            --checkpoint $CHECKPOINT_LIFT \
            --output "$RESULTS_DIR/heatmap_timestep_${EPOCH}ep.png" \
            --num_points $NUM_POINTS \
            --num_steps $NUM_STEPS \
            --device $DEVICE
    else
        echo "[Skip] Heatmap already exists: $HEATMAP_FILE"
    fi

    # Step 2: Generate images
    echo "[Step 2] Generating images..."

    # Baseline (reuse if exists from eval_all_epochs.sh)
    BASELINE_DIR="$RESULTS_DIR/fid_baseline_${EPOCH}ep"
    if [ ! -d "$BASELINE_DIR" ] || [ "$(ls -A $BASELINE_DIR 2>/dev/null | wc -l)" -lt "$NUM_IMAGES" ]; then
        echo "  - Generating Baseline images..."
        $PYTHON generate_baseline_for_fid.py \
            --checkpoint $CHECKPOINT_BASELINE \
            --output_dir $BASELINE_DIR \
            --num_images $NUM_IMAGES \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --device $DEVICE
    else
        echo "  - [Skip] Baseline images exist"
    fi

    # LIFT Diagonal (reuse if exists)
    DIAGONAL_DIR="$RESULTS_DIR/fid_lift_diagonal_${EPOCH}ep"
    if [ ! -d "$DIAGONAL_DIR" ] || [ "$(ls -A $DIAGONAL_DIR 2>/dev/null | wc -l)" -lt "$NUM_IMAGES" ]; then
        echo "  - Generating LIFT Diagonal images..."
        $PYTHON generate_for_fid.py \
            --checkpoint $CHECKPOINT_LIFT \
            --output_dir $DIAGONAL_DIR \
            --num_images $NUM_IMAGES \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --mode diagonal \
            --device $DEVICE
    else
        echo "  - [Skip] LIFT Diagonal images exist"
    fi

    # LIFT DP-64 with 100×100 heatmap
    DP64_100_DIR="$RESULTS_DIR/fid_lift_dp64_100_${EPOCH}ep"
    if [ ! -d "$DP64_100_DIR" ] || [ "$(ls -A $DP64_100_DIR 2>/dev/null | wc -l)" -lt "$NUM_IMAGES" ]; then
        echo "  - Generating LIFT DP-64 (100×100) images..."
        $PYTHON generate_with_dp_path.py \
            --checkpoint $CHECKPOINT_LIFT \
            --output_dir $DP64_100_DIR \
            --num_images $NUM_IMAGES \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --heatmap $HEATMAP_FILE \
            --path_type dp_64 \
            --device $DEVICE
    else
        echo "  - [Skip] LIFT DP-64 (100×100) images exist"
    fi

    # LIFT DP-Total with 100×100 heatmap
    DP_TOTAL_100_DIR="$RESULTS_DIR/fid_lift_dp_total_100_${EPOCH}ep"
    if [ ! -d "$DP_TOTAL_100_DIR" ] || [ "$(ls -A $DP_TOTAL_100_DIR 2>/dev/null | wc -l)" -lt "$NUM_IMAGES" ]; then
        echo "  - Generating LIFT DP-Total (100×100) images..."
        $PYTHON generate_with_dp_path.py \
            --checkpoint $CHECKPOINT_LIFT \
            --output_dir $DP_TOTAL_100_DIR \
            --num_images $NUM_IMAGES \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --heatmap $HEATMAP_FILE \
            --path_type dp_total \
            --device $DEVICE
    else
        echo "  - [Skip] LIFT DP-Total (100×100) images exist"
    fi

    # Step 3: Compute FID scores
    echo "[Step 3] Computing FID scores..."

    FID_BASELINE=$($PYTHON -m pytorch_fid $FID_REAL_DIR $BASELINE_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")
    FID_DIAGONAL=$($PYTHON -m pytorch_fid $FID_REAL_DIR $DIAGONAL_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")
    FID_DP64_100=$($PYTHON -m pytorch_fid $FID_REAL_DIR $DP64_100_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")
    FID_DP_TOTAL_100=$($PYTHON -m pytorch_fid $FID_REAL_DIR $DP_TOTAL_100_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")

    echo ""
    echo "Results for Epoch $EPOCH (100×100 heatmap):"
    echo "  Baseline:           $FID_BASELINE"
    echo "  LIFT Diagonal:      $FID_DIAGONAL"
    echo "  LIFT DP-64 (100):   $FID_DP64_100"
    echo "  LIFT DP-Total (100): $FID_DP_TOTAL_100"

    # Append to CSV
    echo "$EPOCH,$FID_BASELINE,$FID_DIAGONAL,$FID_DP64_100,$FID_DP_TOTAL_100" >> "$RESULTS_DIR/fid_results_heatmap100.csv"
}

# Main execution
prepare_real_images
init_csv

for EPOCH in "${EPOCHS[@]}"; do
    evaluate_epoch $EPOCH || echo "[Warning] Failed to evaluate epoch $EPOCH"
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR/fid_results_heatmap100.csv"
echo ""
cat "$RESULTS_DIR/fid_results_heatmap100.csv"
