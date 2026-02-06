#!/bin/bash
# Evaluate DP-Total path for 1k images (to complete the comparison)
set -e

cd /home/ylong030/slot/simple_diffusion_clean

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"

CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"
NUM_IMAGES=1000
BATCH_SIZE=32
NUM_STEPS=50
DEVICE=3  # Use GPU 3

EPOCHS=(20 40 60 80 100)

echo "=========================================="
echo "Evaluating LIFT DP-Total Path (1k images)"
echo "=========================================="
echo "Device: GPU $DEVICE"
echo ""

# Function to compute FID
compute_fid() {
    local fake_dir=$1
    local real_dir=$2
    echo "Computing FID..."
    $PYTHON -m pytorch_fid "$real_dir" "$fake_dir" --device cuda:$DEVICE
}

for epoch in "${EPOCHS[@]}"; do
    checkpoint="$CHECKPOINT_DIR/lift_dual_timestep_${epoch}ep.pth"

    if [ ! -f "$checkpoint" ]; then
        echo "⚠️  Checkpoint not found: $checkpoint"
        continue
    fi

    heatmap_file="$RESULTS_DIR/error_heatmap_${epoch}ep.pth"

    if [ ! -f "$heatmap_file" ]; then
        echo "⚠️  Heatmap not found: $heatmap_file"
        continue
    fi

    output_dir="$RESULTS_DIR/fid_lift_dp_total_${epoch}ep"

    # Skip if already generated
    if [ -d "$output_dir" ] && [ $(ls -1 "$output_dir"/*.png 2>/dev/null | wc -l) -ge $NUM_IMAGES ]; then
        echo ""
        echo "--- LIFT ${epoch} epochs ---"
        echo "✓ Already generated, computing FID..."
        fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
        echo "✓ LIFT ${epoch}ep DP-Total FID: $fid_score"
        continue
    fi

    echo ""
    echo "--- LIFT ${epoch} epochs ---"
    echo "Checkpoint: $checkpoint"
    echo "Output: $output_dir"

    # Generate images
    $PYTHON generate_for_fid.py \
        --checkpoint "$checkpoint" \
        --output_dir "$output_dir" \
        --num_images $NUM_IMAGES \
        --batch_size $BATCH_SIZE \
        --num_steps $NUM_STEPS \
        --mode dp \
        --heatmap "$heatmap_file" \
        --device $DEVICE

    # Compute FID
    fid_score=$(compute_fid "$output_dir" "$RESULTS_DIR/fid_real")
    echo "✓ LIFT ${epoch}ep DP-Total FID: $fid_score"
done

echo ""
echo "=========================================="
echo "DP-Total Evaluation Complete!"
echo "=========================================="
