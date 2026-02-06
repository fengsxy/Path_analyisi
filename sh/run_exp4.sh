#!/bin/bash
# Experiment 4: Test if 32×32 contributes during generation
# This experiment uses the existing trained LIFT model

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

CHECKPOINT="checkpoints/lift_dual_timestep_final.pth"
NUM_IMAGES=1000
BATCH_SIZE=32
NUM_STEPS=50
DEVICE=0
SEED=42

echo "========================================="
echo "Experiment 4: 32×32 Contribution Test"
echo "========================================="
echo ""
echo "Using checkpoint: $CHECKPOINT"
echo "Generating $NUM_IMAGES images per mode"
echo ""

# Baseline: Normal diagonal generation (both scales denoised)
echo "[1/3] Generating with diagonal path (baseline)..."
python generate_for_fid.py \
    --checkpoint $CHECKPOINT \
    --output_dir results/fid_exp4_diagonal \
    --num_images $NUM_IMAGES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --mode diagonal \
    --device $DEVICE \
    --seed $SEED

# Experiment 4a: 32×32 stays random noise
echo ""
echo "[2/3] Generating with random 32×32 (Exp 4a)..."
python generate_for_fid.py \
    --checkpoint $CHECKPOINT \
    --output_dir results/fid_exp4_random32 \
    --num_images $NUM_IMAGES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --mode random32 \
    --device $DEVICE \
    --seed $SEED

# Experiment 4b: 32×32 fixed noise
echo ""
echo "[3/3] Generating with fixed 32×32 (Exp 4b)..."
python generate_for_fid.py \
    --checkpoint $CHECKPOINT \
    --output_dir results/fid_exp4_fixed32 \
    --num_images $NUM_IMAGES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --mode fixed32 \
    --device $DEVICE \
    --seed $SEED

echo ""
echo "========================================="
echo "Computing FID scores..."
echo "========================================="
echo ""

# Compute FID for all three modes
echo "[1/3] FID for diagonal (baseline):"
python -m pytorch_fid results/fid_real results/fid_exp4_diagonal

echo ""
echo "[2/3] FID for random32 (Exp 4a):"
python -m pytorch_fid results/fid_real results/fid_exp4_random32

echo ""
echo "[3/3] FID for fixed32 (Exp 4b):"
python -m pytorch_fid results/fid_real results/fid_exp4_fixed32

echo ""
echo "========================================="
echo "Experiment 4 Complete!"
echo "========================================="
echo ""
echo "Results interpretation:"
echo "  - If random32/fixed32 FID ≈ diagonal FID → H4 holds (32×32 doesn't contribute)"
echo "  - If random32/fixed32 FID >> diagonal FID → H4 rejected (32×32 helps)"
