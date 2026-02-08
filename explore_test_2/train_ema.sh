#!/bin/bash
# Train Baseline and LIFT models with EMA in parallel
# Usage: ./train_ema.sh
#
# This script runs:
#   - GPU 0: Baseline with EMA
#   - GPU 1: LIFT with EMA

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

cd "$(dirname "$0")/.."

EPOCHS=2000
SAVE_EVERY=200
EMA_DECAY=0.9999

echo "=============================================="
echo "Training with EMA (2 GPUs in parallel)"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "EMA decay: $EMA_DECAY"
echo "Save every: $SAVE_EVERY epochs"
echo ""

# Create directories
mkdir -p checkpoints
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run both trainings in parallel
echo "[GPU 0] Starting Baseline + EMA training..."
python explore_test_2/train_baseline_ema.py \
    --epochs $EPOCHS \
    --ema_decay $EMA_DECAY \
    --save_every $SAVE_EVERY \
    --device 0 \
    > logs/train_baseline_ema_${EPOCHS}ep_${TIMESTAMP}.log 2>&1 &
PID_BASELINE=$!

echo "[GPU 1] Starting LIFT + EMA training..."
python explore_test_2/train_lift_ema.py \
    --epochs $EPOCHS \
    --ema_decay $EMA_DECAY \
    --save_every $SAVE_EVERY \
    --device 1 \
    > logs/train_lift_ema_${EPOCHS}ep_${TIMESTAMP}.log 2>&1 &
PID_LIFT=$!

echo ""
echo "Jobs started:"
echo "  Baseline + EMA (GPU 0): PID $PID_BASELINE"
echo "  LIFT + EMA (GPU 1): PID $PID_LIFT"
echo ""
echo "Logs:"
echo "  tail -f logs/train_baseline_ema_${EPOCHS}ep_${TIMESTAMP}.log"
echo "  tail -f logs/train_lift_ema_${EPOCHS}ep_${TIMESTAMP}.log"
echo ""

# Wait for all jobs to complete
echo "Waiting for training to complete..."
wait $PID_BASELINE
echo "[GPU 0] Baseline + EMA training completed."

wait $PID_LIFT
echo "[GPU 1] LIFT + EMA training completed."

echo ""
echo "=============================================="
echo "All training completed!"
echo "=============================================="
echo ""
echo "Checkpoints saved to:"
echo "  checkpoints/baseline_ema_*.pth"
echo "  checkpoints/lift_ema_*.pth"
