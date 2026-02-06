#!/bin/bash
# Re-evaluate FID with 10000 images for more stable results
set -e

cd /home/ylong030/slot/simple_diffusion_clean

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"

# Configuration
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results_10k"
NUM_IMAGES=10000
BATCH_SIZE=32
NUM_STEPS=50
DEVICE=3  # Use GPU 3

# Epochs to evaluate
EPOCHS=(20 40 60 80 100)

mkdir -p $RESULTS_DIR

echo "=========================================="
echo "FID Evaluation with 10000 Images"
echo "=========================================="
echo "Checkpoints: ${EPOCHS[@]} epochs"
echo "Images per checkpoint: $NUM_IMAGES"
echo "Device: GPU $DEVICE"
echo ""

# Function to compute FID
compute_fid() {
    local fake_dir=$1
    local real_dir=$2
    echo "Computing FID..."
    $PYTHON -m pytorch_fid "$real_dir" "$fake_dir" --device cuda:$DEVICE
}

# Prepare real images if not exists
if [ ! -d "$RESULTS_DIR/fid_real" ]; then
    echo "Preparing 10000 real images for FID..."
    $PYTHON prepare_fid_real.py \
        --output_dir $RESULTS_DIR/fid_real \
        --num_images $NUM_IMAGES
    echo ""
fi

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

    # Skip if already generated
    if [ -d "$output_dir" ] && [ $(ls -1 "$output_dir"/*.png 2>/dev/null | wc -l) -ge $NUM_IMAGES ]; then
        echo ""
        echo "--- Baseline ${epoch} epochs ---"
        echo "✓ Already generated, skipping..."

        # Compute FID if not already done
        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ Baseline ${epoch}ep FID (10k): $fid_score"
        continue
    fi

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
    echo "✓ Baseline ${epoch}ep FID (10k): $fid_score"
done

echo ""
echo "=========================================="
echo "Evaluating LIFT Checkpoints"
echo "=========================================="

for epoch in "${EPOCHS[@]}"; do
    checkpoint="$CHECKPOINT_DIR/lift_dual_timestep_${epoch}ep.pth"

    if [ ! -f "$checkpoint" ]; then
        echo "⚠️  Checkpoint not found: $checkpoint"
        continue
    fi

    heatmap_file="$RESULTS_DIR/error_heatmap_${epoch}ep.pth"

    # Use existing heatmap from 1k evaluation
    if [ ! -f "$heatmap_file" ]; then
        if [ -f "results/error_heatmap_${epoch}ep.pth" ]; then
            echo "Copying heatmap from 1k evaluation..."
            cp "results/error_heatmap_${epoch}ep.pth" "$heatmap_file"
        else
            echo "Generating heatmap for ${epoch}ep..."
            $PYTHON compute_error_heatmap.py \
                --checkpoint "$checkpoint" \
                --output "$RESULTS_DIR/error_heatmap_${epoch}ep.png" \
                --device $DEVICE
        fi
    fi

    echo ""
    echo "--- LIFT ${epoch} epochs ---"
    echo "Checkpoint: $checkpoint"

    # 1. Diagonal path
    output_dir="$RESULTS_DIR/fid_lift_diagonal_${epoch}ep"

    if [ -d "$output_dir" ] && [ $(ls -1 "$output_dir"/*.png 2>/dev/null | wc -l) -ge $NUM_IMAGES ]; then
        echo ""
        echo "Path: Diagonal (γ₁ = γ₀) - Already generated, skipping..."
        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ LIFT ${epoch}ep Diagonal FID (10k): $fid_score"
    else
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
        echo "✓ LIFT ${epoch}ep Diagonal FID (10k): $fid_score"
    fi

    # 2. DP 64×64 path
    output_dir="$RESULTS_DIR/fid_lift_dp64_${epoch}ep"

    if [ -d "$output_dir" ] && [ $(ls -1 "$output_dir"/*.png 2>/dev/null | wc -l) -ge $NUM_IMAGES ]; then
        echo ""
        echo "Path: DP 64×64 - Already generated, skipping..."
        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ LIFT ${epoch}ep DP-64 FID (10k): $fid_score"
    else
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
        echo "✓ LIFT ${epoch}ep DP-64 FID (10k): $fid_score"
    fi

    # 3. DP Total path
    output_dir="$RESULTS_DIR/fid_lift_dp_total_${epoch}ep"

    if [ -d "$output_dir" ] && [ $(ls -1 "$output_dir"/*.png 2>/dev/null | wc -l) -ge $NUM_IMAGES ]; then
        echo ""
        echo "Path: DP Total - Already generated, skipping..."
        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ LIFT ${epoch}ep DP-Total FID (10k): $fid_score"
    else
        echo ""
        echo "Path: DP Total (optimize total error)"
        $PYTHON generate_for_fid.py \
            --checkpoint "$checkpoint" \
            --output_dir "$output_dir" \
            --num_images $NUM_IMAGES \
            --batch_size $BATCH_SIZE \
            --num_steps $NUM_STEPS \
            --mode dp \
            --heatmap "$heatmap_file" \
            --device $DEVICE

        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ LIFT ${epoch}ep DP-Total FID (10k): $fid_score"
    fi
done

echo ""
echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR/"
