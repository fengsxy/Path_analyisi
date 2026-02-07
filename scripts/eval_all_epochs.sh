#!/bin/bash
# Evaluate all checkpoints from 200ep to 2000ep
#
# Usage:
#   ./scripts/eval_all_epochs.sh              # Run full evaluation for all epochs
#   ./scripts/eval_all_epochs.sh 200          # Run evaluation for specific epoch
#   ./scripts/eval_all_epochs.sh 200 400 600  # Run evaluation for multiple epochs
#
# Output:
#   - results/fid_results_all.csv  (FID scores for all epochs)
#   - results/error_heatmap_XXXep.pth (heatmaps for each epoch)

set -e

# Configuration
PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
DEVICE=0
NUM_IMAGES=1000
NUM_STEPS=50
BATCH_SIZE=32
RESULTS_DIR="results"
FID_REAL_DIR="results/fid_real"

# All epochs to evaluate (200 to 2000, step 200)
ALL_EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)

# Parse command line arguments
if [ $# -gt 0 ]; then
    EPOCHS=("$@")
else
    EPOCHS=("${ALL_EPOCHS[@]}")
fi

echo "=========================================="
echo "LIFT vs Baseline Full Evaluation"
echo "=========================================="
echo "Epochs to evaluate: ${EPOCHS[*]}"
echo ""

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p figures

# Step 0: Prepare real images for FID (if not exists)
prepare_real_images() {
    if [ -d "$FID_REAL_DIR" ] && [ "$(ls -A $FID_REAL_DIR 2>/dev/null | wc -l)" -ge "$NUM_IMAGES" ]; then
        echo "[Skip] Real images already prepared in $FID_REAL_DIR"
    else
        echo "[Step 0] Preparing real images for FID..."
        $PYTHON prepare_fid_real.py \
            --output_dir $FID_REAL_DIR \
            --num_images $NUM_IMAGES
        echo "Real images saved to: $FID_REAL_DIR"
    fi
}

# Initialize CSV file
init_csv() {
    CSV_FILE="$RESULTS_DIR/fid_results_all.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo "Epoch,Baseline,LIFT_Diagonal,LIFT_DP64,LIFT_DP_Total" > $CSV_FILE
        echo "Created: $CSV_FILE"
    fi
}

# Evaluate single epoch
evaluate_epoch() {
    local EPOCH=$1
    echo ""
    echo "=========================================="
    echo "Evaluating Epoch $EPOCH"
    echo "=========================================="

    CHECKPOINT_BASELINE="checkpoints/baseline_${EPOCH}ep.pth"
    CHECKPOINT_LIFT="checkpoints/lift_dual_timestep_${EPOCH}ep.pth"
    HEATMAP_FILE="$RESULTS_DIR/error_heatmap_${EPOCH}ep.pth"

    # Check if checkpoints exist
    if [ ! -f "$CHECKPOINT_BASELINE" ]; then
        echo "[Warning] Baseline checkpoint not found: $CHECKPOINT_BASELINE"
        return 1
    fi
    if [ ! -f "$CHECKPOINT_LIFT" ]; then
        echo "[Warning] LIFT checkpoint not found: $CHECKPOINT_LIFT"
        return 1
    fi

    # Step 1: Compute error heatmap (if not exists)
    if [ ! -f "$HEATMAP_FILE" ]; then
        echo "[Step 1] Computing error heatmap..."
        $PYTHON compute_error_heatmap.py \
            --checkpoint $CHECKPOINT_LIFT \
            --output "${HEATMAP_FILE%.pth}.png" \
            --num_points 15 \
            --device $DEVICE
    else
        echo "[Skip] Heatmap already exists: $HEATMAP_FILE"
    fi

    # Step 2: Generate images
    echo "[Step 2] Generating images..."

    # Baseline
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

    # LIFT Diagonal
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

    # LIFT DP-64
    DP64_DIR="$RESULTS_DIR/fid_lift_dp64_${EPOCH}ep"
    if [ ! -d "$DP64_DIR" ] || [ "$(ls -A $DP64_DIR 2>/dev/null | wc -l)" -lt "$NUM_IMAGES" ]; then
        echo "  - Generating LIFT DP-64 images..."
        $PYTHON generate_for_fid.py \
            --checkpoint $CHECKPOINT_LIFT \
            --output_dir $DP64_DIR \
            --num_images $NUM_IMAGES \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --mode dp_64 \
            --heatmap $HEATMAP_FILE \
            --device $DEVICE
    else
        echo "  - [Skip] LIFT DP-64 images exist"
    fi

    # LIFT DP-Total
    DP_TOTAL_DIR="$RESULTS_DIR/fid_lift_dp_total_${EPOCH}ep"
    if [ ! -d "$DP_TOTAL_DIR" ] || [ "$(ls -A $DP_TOTAL_DIR 2>/dev/null | wc -l)" -lt "$NUM_IMAGES" ]; then
        echo "  - Generating LIFT DP-Total images..."
        $PYTHON generate_for_fid.py \
            --checkpoint $CHECKPOINT_LIFT \
            --output_dir $DP_TOTAL_DIR \
            --num_images $NUM_IMAGES \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --mode dp \
            --heatmap $HEATMAP_FILE \
            --device $DEVICE
    else
        echo "  - [Skip] LIFT DP-Total images exist"
    fi

    # Step 3: Compute FID scores
    echo "[Step 3] Computing FID scores..."

    FID_BASELINE=$($PYTHON -m pytorch_fid $FID_REAL_DIR $BASELINE_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")
    FID_DIAGONAL=$($PYTHON -m pytorch_fid $FID_REAL_DIR $DIAGONAL_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")
    FID_DP64=$($PYTHON -m pytorch_fid $FID_REAL_DIR $DP64_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")
    FID_DP_TOTAL=$($PYTHON -m pytorch_fid $FID_REAL_DIR $DP_TOTAL_DIR 2>&1 | grep -oP 'FID:\s*\K[\d.]+' || echo "N/A")

    echo ""
    echo "Results for Epoch $EPOCH:"
    echo "  Baseline:      $FID_BASELINE"
    echo "  LIFT Diagonal: $FID_DIAGONAL"
    echo "  LIFT DP-64:    $FID_DP64"
    echo "  LIFT DP-Total: $FID_DP_TOTAL"

    # Append to CSV
    echo "$EPOCH,$FID_BASELINE,$FID_DIAGONAL,$FID_DP64,$FID_DP_TOTAL" >> "$RESULTS_DIR/fid_results_all.csv"
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
echo "Results saved to: $RESULTS_DIR/fid_results_all.csv"
echo ""
cat "$RESULTS_DIR/fid_results_all.csv"
