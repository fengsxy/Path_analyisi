#!/bin/bash
# Evaluate 100-epoch models: generate images and calculate FID
#
# Usage:
#   ./eval100.sh              # Run full evaluation
#   ./eval100.sh generate     # Only generate images
#   ./eval100.sh fid          # Only compute FID

set -e

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"

# Configuration
DEVICE=0
NUM_IMAGES=1000
NUM_STEPS=50
BATCH_SIZE=32

# Checkpoints
CHECKPOINT_BASELINE="checkpoints/baseline_100ep.pth"
CHECKPOINT_LIFT="checkpoints/lift_100ep.pth"
HEATMAP_FILE="results/error_heatmap_100ep.pth"

# Output directories
FID_REAL_DIR="results/fid_real"
FID_BASELINE_DIR="results/fid_baseline_100ep"
FID_DIAGONAL_DIR="results/fid_lift_diagonal_100ep"
FID_DP_TOTAL_DIR="results/fid_lift_dp_total_100ep"
FID_DP_64_DIR="results/fid_lift_dp_64_100ep"

echo "=========================================="
echo "100 Epoch Model Evaluation"
echo "=========================================="

# Check if checkpoints exist
check_checkpoints() {
    if [ ! -f "$CHECKPOINT_BASELINE" ]; then
        echo "Error: $CHECKPOINT_BASELINE not found. Run train100.sh first."
        exit 1
    fi
    if [ ! -f "$CHECKPOINT_LIFT" ]; then
        echo "Error: $CHECKPOINT_LIFT not found. Run train100.sh first."
        exit 1
    fi
    echo "Checkpoints found."
}

# Compute error heatmap for 100ep LIFT model
compute_heatmap() {
    echo ""
    echo "[Step 1] Computing error heatmap for 100ep LIFT model..."
    $PYTHON compute_error_heatmap.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output $HEATMAP_FILE \
        --num_points 15 \
        --device $DEVICE
    echo "Heatmap saved to: $HEATMAP_FILE"
}

# Generate images
generate_images() {
    echo ""
    echo "[Step 2] Generating images for FID evaluation..."

    # Baseline 100ep
    echo "  - Generating Baseline 100ep images..."
    $PYTHON generate_baseline_for_fid.py \
        --checkpoint $CHECKPOINT_BASELINE \
        --output_dir $FID_BASELINE_DIR \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --device $DEVICE

    # LIFT Diagonal 100ep
    echo "  - Generating LIFT Diagonal 100ep images..."
    $PYTHON generate_for_fid.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output_dir $FID_DIAGONAL_DIR \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --mode diagonal \
        --device $DEVICE

    # LIFT DP Total 100ep
    echo "  - Generating LIFT DP Total 100ep images..."
    $PYTHON generate_for_fid.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output_dir $FID_DP_TOTAL_DIR \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --mode dp \
        --heatmap $HEATMAP_FILE \
        --device $DEVICE

    # LIFT DP 64×64 100ep
    echo "  - Generating LIFT DP 64×64 100ep images..."
    $PYTHON generate_for_fid.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output_dir $FID_DP_64_DIR \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --mode dp_64 \
        --heatmap $HEATMAP_FILE \
        --device $DEVICE

    echo "Generated images saved to results/fid_*_100ep/"
}

# Compute FID
compute_fid() {
    echo ""
    echo "[Step 3] Computing FID scores..."

    echo ""
    echo "Baseline 100ep:"
    $PYTHON -m pytorch_fid $FID_REAL_DIR $FID_BASELINE_DIR

    echo ""
    echo "LIFT Diagonal 100ep:"
    $PYTHON -m pytorch_fid $FID_REAL_DIR $FID_DIAGONAL_DIR

    echo ""
    echo "LIFT DP Total 100ep:"
    $PYTHON -m pytorch_fid $FID_REAL_DIR $FID_DP_TOTAL_DIR

    echo ""
    echo "LIFT DP 64×64 100ep:"
    $PYTHON -m pytorch_fid $FID_REAL_DIR $FID_DP_64_DIR

    echo ""
    echo "=========================================="
    echo "Comparison: 30 epochs vs 100 epochs"
    echo "=========================================="
    echo ""
    echo "30 Epoch Results:"
    echo "  Baseline:      78.83"
    echo "  LIFT DP 64×64: 101.53"
    echo "  LIFT DP Total: 109.20"
    echo "  LIFT Diagonal: 116.14"
    echo ""
    echo "100 Epoch Results: (see above)"
    echo "=========================================="
}

# Parse arguments
case "${1:-all}" in
    heatmap)
        check_checkpoints
        compute_heatmap
        ;;
    generate)
        check_checkpoints
        generate_images
        ;;
    fid)
        compute_fid
        ;;
    all)
        check_checkpoints
        compute_heatmap
        generate_images
        compute_fid
        ;;
    *)
        echo "Usage: $0 [heatmap|generate|fid|all]"
        exit 1
        ;;
esac
