#!/bin/bash
# Train both Baseline and LIFT models for 2000 epochs with checkpoints every 200 epochs
#
# Usage:
#   ./train2000.sh              # Train both models (Baseline on GPU 2, LIFT on GPU 3)
#   ./train2000.sh baseline     # Train only Baseline model
#   ./train2000.sh lift         # Train only LIFT model

set -e

# Activate conda environment
PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"

# Configuration - 2000 epochs with checkpoints every 200 epochs
EPOCHS=2000
SAVE_EVERY=200
BATCH_SIZE=256
HIDDEN_DIMS="64,128,256,512"
OUTPUT_DIR="checkpoints"
LOG_DIR="logs"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Get timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

train_baseline() {
    echo "=========================================="
    echo "Training Baseline Model - 2000 epochs"
    echo "  - Hidden dims: $HIDDEN_DIMS"
    echo "  - Epochs: $EPOCHS"
    echo "  - Save every: $SAVE_EVERY epochs"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 2"
    echo "=========================================="

    $PYTHON train_baseline.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dims $HIDDEN_DIMS \
        --output_dir $OUTPUT_DIR \
        --save_every $SAVE_EVERY \
        --device 2 \
        2>&1 | tee $LOG_DIR/train_baseline_2000ep_$TIMESTAMP.log
}

train_lift() {
    echo "=========================================="
    echo "Training LIFT Dual Timestep Model - 2000 epochs"
    echo "  - Hidden dims: $HIDDEN_DIMS"
    echo "  - Epochs: $EPOCHS"
    echo "  - Save every: $SAVE_EVERY epochs"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 3"
    echo "=========================================="

    $PYTHON train_lift.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dims $HIDDEN_DIMS \
        --output_dir $OUTPUT_DIR \
        --save_every $SAVE_EVERY \
        --device 3 \
        2>&1 | tee $LOG_DIR/train_lift_2000ep_$TIMESTAMP.log
}

# Parse command line arguments
case "${1:-both}" in
    baseline)
        train_baseline
        ;;
    lift)
        train_lift
        ;;
    both)
        echo "=========================================="
        echo "2000 Epoch Training Experiment"
        echo "=========================================="
        echo "Starting parallel training on 2 GPUs..."
        echo "  - Baseline on GPU 2"
        echo "  - LIFT on GPU 3"
        echo ""

        # Run both in parallel
        train_baseline &
        PID_BASELINE=$!

        train_lift &
        PID_LIFT=$!

        # Wait for both to complete
        echo "Waiting for training to complete..."
        wait $PID_BASELINE
        BASELINE_STATUS=$?

        wait $PID_LIFT
        LIFT_STATUS=$?

        echo ""
        echo "=========================================="
        echo "2000 Epoch Training Complete!"
        echo "=========================================="
        echo "  - Baseline: $([ $BASELINE_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo "  - LIFT: $([ $LIFT_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        ;;
    *)
        echo "Usage: $0 [baseline|lift|both]"
        exit 1
        ;;
esac
