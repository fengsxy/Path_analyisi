#!/bin/bash
# Evaluate models: compute heatmap, generate images, and calculate FID
#
# Usage:
#   ./eval.sh              # Run full evaluation pipeline
#   ./eval.sh heatmap      # Only compute error heatmap
#   ./eval.sh generate     # Only generate images for FID
#   ./eval.sh fid          # Only compute FID scores
#
# Requirements:
#   - Trained checkpoints in checkpoints/
#   - conda environment: diffusion-gpu

set -e

# Activate conda environment
PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"

# Configuration
DEVICE=0
NUM_IMAGES=1000
NUM_STEPS=50
BATCH_SIZE=32

# Directories
CHECKPOINT_BASELINE="checkpoints/baseline_final.pth"
CHECKPOINT_LIFT="checkpoints/lift_dual_timestep_final.pth"
HEATMAP_FILE="results/error_heatmap.pth"
RESULTS_DIR="results"
FID_REAL_DIR="results/fid_real"

# Create directories
mkdir -p $RESULTS_DIR
mkdir -p figures

echo "=========================================="
echo "LIFT vs Baseline Evaluation Pipeline"
echo "=========================================="

# Step 1: Prepare real images for FID (if not exists)
prepare_real_images() {
    if [ -d "$FID_REAL_DIR" ] && [ "$(ls -A $FID_REAL_DIR 2>/dev/null | wc -l)" -ge "$NUM_IMAGES" ]; then
        echo "[Skip] Real images already prepared in $FID_REAL_DIR"
    else
        echo "[Step 1] Preparing real images for FID..."
        $PYTHON prepare_fid_real.py \
            --output_dir $FID_REAL_DIR \
            --num_images $NUM_IMAGES
        echo "Real images saved to: $FID_REAL_DIR"
    fi
}

# Step 2: Compute error heatmap for LIFT model
compute_heatmap() {
    echo ""
    echo "[Step 2] Computing error heatmap for LIFT model..."
    $PYTHON compute_error_heatmap.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output $HEATMAP_FILE \
        --num_points 15 \
        --device $DEVICE
    echo "Heatmap saved to: $HEATMAP_FILE"
}

# Step 3: Plot heatmap with DP paths
plot_heatmaps() {
    echo ""
    echo "[Step 3] Plotting heatmaps with optimal paths..."

    # DP path on total error
    echo "  - Plotting DP Total path..."
    $PYTHON plot_heatmap.py --heatmap $HEATMAP_FILE --dp

    # DP path on 64×64 error only
    echo "  - Plotting DP 64×64 path..."
    $PYTHON plot_heatmap.py --heatmap $HEATMAP_FILE --dp_64

    # Copy to figures directory
    cp results/error_heatmap_with_path_dp.png figures/heatmap_dp_path.png 2>/dev/null || true
    cp results/error_heatmap_with_path_dp_64.png figures/heatmap_dp_64_path.png 2>/dev/null || true

    echo "Heatmap figures saved to figures/"
}

# Step 4: Generate images for FID evaluation
generate_images() {
    echo ""
    echo "[Step 4] Generating images for FID evaluation..."

    # Baseline model
    echo "  - Generating Baseline images..."
    $PYTHON generate_baseline_for_fid.py \
        --checkpoint $CHECKPOINT_BASELINE \
        --output_dir results/fid_baseline \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --device $DEVICE

    # LIFT Diagonal path
    echo "  - Generating LIFT Diagonal images..."
    $PYTHON generate_for_fid.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output_dir results/fid_lift_diagonal \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --mode diagonal \
        --device $DEVICE

    # LIFT DP Total path
    echo "  - Generating LIFT DP Total images..."
    $PYTHON generate_for_fid.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output_dir results/fid_lift_dp_total \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --mode dp \
        --heatmap $HEATMAP_FILE \
        --device $DEVICE

    # LIFT DP 64×64 path
    echo "  - Generating LIFT DP 64×64 images..."
    $PYTHON generate_for_fid.py \
        --checkpoint $CHECKPOINT_LIFT \
        --output_dir results/fid_lift_dp_64 \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --batch_size $BATCH_SIZE \
        --mode dp_64 \
        --heatmap $HEATMAP_FILE \
        --device $DEVICE

    echo "Generated images saved to results/fid_*/"
}

# Step 5: Compute FID scores
compute_fid() {
    echo ""
    echo "[Step 5] Computing FID scores..."
    echo ""

    echo "Computing FID for Baseline..."
    FID_BASELINE=$($PYTHON -m pytorch_fid $FID_REAL_DIR results/fid_baseline 2>&1 | grep -oP 'FID:\s*\K[\d.]+')
    echo "  Baseline FID: $FID_BASELINE"

    echo "Computing FID for LIFT Diagonal..."
    FID_DIAGONAL=$($PYTHON -m pytorch_fid $FID_REAL_DIR results/fid_lift_diagonal 2>&1 | grep -oP 'FID:\s*\K[\d.]+')
    echo "  LIFT Diagonal FID: $FID_DIAGONAL"

    echo "Computing FID for LIFT DP Total..."
    FID_DP_TOTAL=$($PYTHON -m pytorch_fid $FID_REAL_DIR results/fid_lift_dp_total 2>&1 | grep -oP 'FID:\s*\K[\d.]+')
    echo "  LIFT DP Total FID: $FID_DP_TOTAL"

    echo "Computing FID for LIFT DP 64×64..."
    FID_DP_64=$($PYTHON -m pytorch_fid $FID_REAL_DIR results/fid_lift_dp_64 2>&1 | grep -oP 'FID:\s*\K[\d.]+')
    echo "  LIFT DP 64×64 FID: $FID_DP_64"

    echo ""
    echo "=========================================="
    echo "FID Results Summary"
    echo "=========================================="
    echo "| Model / Path          | FID ↓        |"
    echo "|-----------------------|--------------|"
    echo "| Baseline (Non-LIFT)   | $FID_BASELINE |"
    echo "| LIFT Diagonal         | $FID_DIAGONAL |"
    echo "| LIFT DP Total         | $FID_DP_TOTAL |"
    echo "| LIFT DP 64×64         | $FID_DP_64 |"
    echo "=========================================="

    # Save results to file
    cat > results/fid_results.txt << EOF
FID Evaluation Results
======================
Date: $(date)
Num Images: $NUM_IMAGES
Num Steps: $NUM_STEPS

| Model / Path          | FID ↓        |
|-----------------------|--------------|
| Baseline (Non-LIFT)   | $FID_BASELINE |
| LIFT Diagonal         | $FID_DIAGONAL |
| LIFT DP Total         | $FID_DP_TOTAL |
| LIFT DP 64×64         | $FID_DP_64 |
EOF
    echo ""
    echo "Results saved to: results/fid_results.txt"
}

# Step 6: Create sample grids for visualization
create_sample_grids() {
    echo ""
    echo "[Step 6] Creating sample grids for visualization..."

    $PYTHON create_sample_grids.py \
        --baseline_dir results/fid_baseline \
        --diagonal_dir results/fid_lift_diagonal \
        --dp_total_dir results/fid_lift_dp_total \
        --dp_64_dir results/fid_lift_dp_64 \
        --output_dir figures

    echo "Sample grids saved to figures/"
}

# Parse command line arguments
case "${1:-all}" in
    heatmap)
        compute_heatmap
        plot_heatmaps
        ;;
    generate)
        prepare_real_images
        generate_images
        ;;
    fid)
        compute_fid
        ;;
    grids)
        create_sample_grids
        ;;
    all)
        prepare_real_images
        compute_heatmap
        plot_heatmaps
        generate_images
        compute_fid
        create_sample_grids
        echo ""
        echo "=========================================="
        echo "Full evaluation pipeline completed!"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [heatmap|generate|fid|grids|all]"
        exit 1
        ;;
esac
