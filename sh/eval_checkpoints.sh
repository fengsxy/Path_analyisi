#!/bin/bash
# Evaluate FID for all checkpoints (20, 40, 60, 80, 100 epochs)
#
# Usage:
#   ./eval_checkpoints.sh              # Evaluate all checkpoints
#   ./eval_checkpoints.sh baseline     # Evaluate only Baseline checkpoints
#   ./eval_checkpoints.sh lift         # Evaluate only LIFT checkpoints
#
# Purpose:
#   Track FID progression throughout training to see when LIFT surpasses Baseline

set -e

# Change to project root directory
cd /home/ylong030/slot/simple_diffusion_clean

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"

# Configuration
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"
NUM_IMAGES=1000
BATCH_SIZE=32
NUM_STEPS=50
DEVICE=0

# Epochs to evaluate
EPOCHS=(20 40 60 80 100)

# Create results directory
mkdir -p $RESULTS_DIR

echo "=========================================="
echo "FID Evaluation for Multiple Checkpoints"
echo "=========================================="
echo "Checkpoints: ${EPOCHS[@]} epochs"
echo "Images per checkpoint: $NUM_IMAGES"
echo "Device: GPU $DEVICE"
echo ""

# Function to compute FID using pytorch-fid
compute_fid() {
    local fake_dir=$1
    local real_dir=$2

    echo "Computing FID..."
    $PYTHON -m pytorch_fid "$real_dir" "$fake_dir" --device cuda:$DEVICE
}

# Prepare real images if not exists
if [ ! -d "$RESULTS_DIR/fid_real" ]; then
    echo "Preparing real images for FID..."
    $PYTHON prepare_fid_real.py \
        --output_dir $RESULTS_DIR/fid_real \
        --num_images $NUM_IMAGES
    echo ""
fi

eval_baseline() {
    echo "=========================================="
    echo "Evaluating Baseline Checkpoints"
    echo "=========================================="

    for epoch in "${EPOCHS[@]}"; do
        checkpoint="$CHECKPOINT_DIR/baseline_${epoch}ep.pth"

        if [ ! -f "$checkpoint" ]; then
            echo "⚠️  Checkpoint not found: $checkpoint"
            continue
        fi

        output_dir="$RESULTS_DIR/fid_baseline_${epoch}ep"

        echo ""
        echo "--- Baseline ${epoch} epochs ---"
        echo "Checkpoint: $checkpoint"
        echo "Output: $output_dir"

        # Generate images
        $PYTHON tmp/generate_baseline_for_fid.py \
            --checkpoint "$checkpoint" \
            --output_dir "$output_dir" \
            --num_images $NUM_IMAGES \
            --batch_size $BATCH_SIZE \
            --num_steps $NUM_STEPS \
            --device $DEVICE

        # Compute FID
        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ Baseline ${epoch}ep FID: $fid_score"
    done

    echo ""
    echo "=========================================="
    echo "Baseline Evaluation Complete"
    echo "=========================================="
}

eval_lift() {
    echo "=========================================="
    echo "Evaluating LIFT Checkpoints"
    echo "=========================================="

    # First, generate error heatmaps for each checkpoint
    echo "Generating error heatmaps..."
    for epoch in "${EPOCHS[@]}"; do
        checkpoint="$CHECKPOINT_DIR/lift_dual_timestep_${epoch}ep.pth"

        if [ ! -f "$checkpoint" ]; then
            echo "⚠️  Checkpoint not found: $checkpoint"
            continue
        fi

        heatmap_file="$RESULTS_DIR/error_heatmap_${epoch}ep.pth"

        if [ ! -f "$heatmap_file" ]; then
            echo "Generating heatmap for ${epoch}ep..."
            $PYTHON compute_error_heatmap.py \
                --checkpoint "$checkpoint" \
                --output "$RESULTS_DIR/error_heatmap_${epoch}ep.png" \
                --device $DEVICE
        else
            echo "Heatmap exists: $heatmap_file"
        fi
    done
    echo ""

    # Evaluate each checkpoint with different generation paths
    for epoch in "${EPOCHS[@]}"; do
        checkpoint="$CHECKPOINT_DIR/lift_dual_timestep_${epoch}ep.pth"

        if [ ! -f "$checkpoint" ]; then
            continue
        fi

        heatmap_file="$RESULTS_DIR/error_heatmap_${epoch}ep.pth"

        echo ""
        echo "--- LIFT ${epoch} epochs ---"
        echo "Checkpoint: $checkpoint"

        # 1. Diagonal path
        output_dir="$RESULTS_DIR/fid_lift_diagonal_${epoch}ep"
        echo ""
        echo "Path: Diagonal (γ₁ = γ₀)"
        $PYTHON generate_for_fid.py \
            --checkpoint "$checkpoint" \
            --output_dir "$output_dir" \
            --num_images $NUM_IMAGES \
            --batch_size $BATCH_SIZE \
            --num_steps $NUM_STEPS \
            --mode diagonal \
            --device $DEVICE

        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ LIFT ${epoch}ep Diagonal FID: $fid_score"

        # 2. DP 64×64 path (optimize 64×64 error only)
        output_dir="$RESULTS_DIR/fid_lift_dp64_${epoch}ep"
        echo ""
        echo "Path: DP 64×64 (optimize 64×64 error)"
        $PYTHON generate_for_fid.py \
            --checkpoint "$checkpoint" \
            --output_dir "$output_dir" \
            --num_images $NUM_IMAGES \
            --batch_size $BATCH_SIZE \
            --num_steps $NUM_STEPS \
            --mode dp_64 \
            --heatmap "$heatmap_file" \
            --device $DEVICE

        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ LIFT ${epoch}ep DP-64 FID: $fid_score"
    done

    echo ""
    echo "=========================================="
    echo "LIFT Evaluation Complete"
    echo "=========================================="
}

# Parse command line arguments
case "${1:-both}" in
    baseline)
        eval_baseline
        ;;
    lift)
        eval_lift
        ;;
    both)
        eval_baseline
        eval_lift
        ;;
    *)
        echo "Usage: $0 [baseline|lift|both]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Next steps:"
echo "  1. Analyze FID progression curves"
echo "  2. Identify when LIFT surpasses Baseline"
echo "  3. Create visualization plots"
echo "=========================================="
