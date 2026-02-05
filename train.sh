#!/bin/bash
# Train both Baseline and LIFT models in parallel
#
# Usage:
#   ./train.sh              # Train both models (Baseline on GPU 0, LIFT on GPU 1)
#   ./train.sh baseline     # Train only Baseline model
#   ./train.sh lift         # Train only LIFT model
#
# Requirements:
#   - 2 GPUs for parallel training
#   - conda environment with PyTorch, datasets, etc.

set -e

# Activate conda environment
CONDA_ENV="diffusion-gpu"
PYTHON="/home/ylong030/miniconda3/envs/${CONDA_ENV}/bin/python"

# Configuration
EPOCHS=30
BATCH_SIZE=64
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
    echo "Training Baseline Model (Non-LIFT)"
    echo "  - Hidden dims: $HIDDEN_DIMS"
    echo "  - Epochs: $EPOCHS"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 0"
    echo "=========================================="

    $PYTHON train_baseline.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dims $HIDDEN_DIMS \
        --output_dir $OUTPUT_DIR \
        --device 0 \
        2>&1 | tee $LOG_DIR/train_baseline_$TIMESTAMP.log
}

train_lift() {
    echo "=========================================="
    echo "Training LIFT Dual Timestep Model"
    echo "  - Hidden dims: $HIDDEN_DIMS"
    echo "  - Epochs: $EPOCHS"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 1"
    echo "=========================================="

    $PYTHON train_lift.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dims $HIDDEN_DIMS \
        --output_dir $OUTPUT_DIR \
        --device 1 \
        2>&1 | tee $LOG_DIR/train_lift_$TIMESTAMP.log
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
        echo "Starting parallel training on 2 GPUs..."
        echo "  - Baseline on GPU 0"
        echo "  - LIFT on GPU 1"
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
        echo "Training Complete!"
        echo "  - Baseline: $([ $BASELINE_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo "  - LIFT: $([ $LIFT_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo ""
        echo "Checkpoints saved to: $OUTPUT_DIR/"
        echo "  - baseline_final.pth"
        echo "  - lift_dual_timestep_final.pth"
        echo ""
        echo "Logs saved to: $LOG_DIR/"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [baseline|lift|both]"
        exit 1
        ;;
esac
