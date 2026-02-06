#!/bin/bash
# Generate images and compute FID for Experiments 1 & 2

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

NUM_IMAGES=1000
BATCH_SIZE=32
NUM_STEPS=50
DEVICE=0
SEED=42

echo "========================================="
echo "Generate Images & Compute FID"
echo "Experiments 1 & 2"
echo "========================================="
echo ""

# Experiment 1: loss_64 only
echo "[1/4] Generating images for Experiment 1 (loss_64 only)..."
python generate_for_fid.py \
    --checkpoint checkpoints/lift_exp1_loss64only.pth \
    --output_dir results/fid_exp1_loss64only \
    --num_images $NUM_IMAGES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --mode diagonal \
    --device $DEVICE \
    --seed $SEED

echo ""
echo "[2/4] Generating images for Experiment 2 (diagonal)..."
python generate_for_fid.py \
    --checkpoint checkpoints/lift_exp2_diagonal.pth \
    --output_dir results/fid_exp2_diagonal \
    --num_images $NUM_IMAGES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --mode diagonal \
    --device $DEVICE \
    --seed $SEED

echo ""
echo "========================================="
echo "Computing FID Scores"
echo "========================================="
echo ""

echo "[3/4] FID for Experiment 1 (loss_64 only):"
python -m pytorch_fid results/fid_real results/fid_exp1_loss64only

echo ""
echo "[4/4] FID for Experiment 2 (diagonal):"
python -m pytorch_fid results/fid_real results/fid_exp2_diagonal

echo ""
echo "========================================="
echo "Results Summary"
echo "========================================="
echo ""
echo "Baseline (30 epochs):        FID = 78.83"
echo "LIFT Original (30 epochs):   FID = 116.14"
echo ""
echo "Experiment 1 (loss_64 only): FID = [see above]"
echo "Experiment 2 (diagonal):     FID = [see above]"
echo ""
