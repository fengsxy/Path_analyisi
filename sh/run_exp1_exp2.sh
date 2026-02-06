#!/bin/bash
# Run Experiments 1 and 2 in parallel
# Experiment 1: loss_64 only (test multi-task loss conflict)
# Experiment 2: diagonal timesteps (test training space too large)

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

EPOCHS=30
BATCH_SIZE=64
LR=1e-4
HIDDEN_DIMS="64,128,256,512"
OUTPUT_DIR="checkpoints"
NUM_WORKERS=4
SEED=42

echo "========================================="
echo "Running Experiments 1 & 2 in Parallel"
echo "========================================="
echo ""
echo "Experiment 1: Train with loss_64 only"
echo "Experiment 2: Train with diagonal timesteps"
echo ""
echo "Settings:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Hidden dims: $HIDDEN_DIMS"
echo "  Random seed: $SEED"
echo ""

# Check available GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
    echo ""
fi

# Ask user which GPUs to use
read -p "Enter GPU ID for Experiment 1 (loss_64 only): " GPU1
read -p "Enter GPU ID for Experiment 2 (diagonal): " GPU2

echo ""
echo "Starting training..."
echo "  Experiment 1 on GPU $GPU1"
echo "  Experiment 2 on GPU $GPU2"
echo ""

# Create log directory
mkdir -p logs

# Run Experiment 1 in background
echo "[Exp 1] Starting on GPU $GPU1..."
python train_exp1_loss64only.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_dims $HIDDEN_DIMS \
    --output_dir $OUTPUT_DIR \
    --num_workers $NUM_WORKERS \
    --device $GPU1 \
    --seed $SEED \
    --fp16 \
    > logs/exp1_loss64only_$(date +%Y%m%d_%H%M%S).log 2>&1 &

EXP1_PID=$!
echo "[Exp 1] PID: $EXP1_PID"

# Wait a bit before starting second experiment
sleep 5

# Run Experiment 2 in background
echo "[Exp 2] Starting on GPU $GPU2..."
python train_exp2_diagonal.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_dims $HIDDEN_DIMS \
    --output_dir $OUTPUT_DIR \
    --num_workers $NUM_WORKERS \
    --device $GPU2 \
    --seed $SEED \
    --fp16 \
    > logs/exp2_diagonal_$(date +%Y%m%d_%H%M%S).log 2>&1 &

EXP2_PID=$!
echo "[Exp 2] PID: $EXP2_PID"

echo ""
echo "========================================="
echo "Both experiments started!"
echo "========================================="
echo ""
echo "Monitor progress:"
echo "  Experiment 1: tail -f logs/exp1_loss64only_*.log"
echo "  Experiment 2: tail -f logs/exp2_diagonal_*.log"
echo ""
echo "Check processes:"
echo "  ps aux | grep train_exp"
echo ""
echo "Waiting for both experiments to complete..."
echo ""

# Wait for both processes
wait $EXP1_PID
EXP1_EXIT=$?
echo "[Exp 1] Completed with exit code: $EXP1_EXIT"

wait $EXP2_PID
EXP2_EXIT=$?
echo "[Exp 2] Completed with exit code: $EXP2_EXIT"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""

if [ $EXP1_EXIT -eq 0 ]; then
    echo "✓ Experiment 1 (loss_64 only) succeeded"
else
    echo "✗ Experiment 1 (loss_64 only) failed"
fi

if [ $EXP2_EXIT -eq 0 ]; then
    echo "✓ Experiment 2 (diagonal) succeeded"
else
    echo "✗ Experiment 2 (diagonal) failed"
fi

echo ""
echo "Next steps:"
echo ""
echo "1. Generate images for FID evaluation:"
echo ""
echo "   # Experiment 1"
echo "   python generate_for_fid.py --checkpoint checkpoints/lift_exp1_loss64only.pth \\"
echo "     --output_dir results/fid_exp1_loss64only --num_images 1000 --mode diagonal"
echo ""
echo "   # Experiment 2"
echo "   python generate_for_fid.py --checkpoint checkpoints/lift_exp2_diagonal.pth \\"
echo "     --output_dir results/fid_exp2_diagonal --num_images 1000 --mode diagonal"
echo ""
echo "2. Compute FID scores:"
echo ""
echo "   python -m pytorch_fid results/fid_real results/fid_exp1_loss64only"
echo "   python -m pytorch_fid results/fid_real results/fid_exp2_diagonal"
echo ""
