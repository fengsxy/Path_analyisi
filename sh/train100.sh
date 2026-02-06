#!/bin/bash
# Train both Baseline and LIFT models for 100 epochs with checkpoints every 20 epochs
#
# Usage:
#   ./train100.sh              # Train both models (Baseline on GPU 0, LIFT on GPU 1)
#   ./train100.sh baseline     # Train only Baseline model
#   ./train100.sh lift         # Train only LIFT model
#
# Purpose:
#   Verify if LIFT model needs more training epochs to converge
#   compared to Baseline model. Save checkpoints every 20 epochs
#   to track FID progression throughout training.

set -e

# Activate conda environment
PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"

# Configuration - 100 epochs with checkpoints every 20 epochs
EPOCHS=100
SAVE_EVERY=20
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
    echo "Training Baseline Model (Non-LIFT) - 100 epochs"
    echo "  - Hidden dims: $HIDDEN_DIMS"
    echo "  - Epochs: $EPOCHS"
    echo "  - Save every: $SAVE_EVERY epochs"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 0"
    echo "=========================================="

    $PYTHON train_baseline.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dims $HIDDEN_DIMS \
        --output_dir $OUTPUT_DIR \
        --save_every $SAVE_EVERY \
        --device 0 \
        2>&1 | tee $LOG_DIR/train_baseline_100ep_$TIMESTAMP.log

    # Rename final checkpoint to indicate 100 epochs
    if [ -f "$OUTPUT_DIR/baseline_final.pth" ]; then
        mv $OUTPUT_DIR/baseline_final.pth $OUTPUT_DIR/baseline_100ep.pth
        echo "Final checkpoint saved as: $OUTPUT_DIR/baseline_100ep.pth"
    fi
}

train_lift() {
    echo "=========================================="
    echo "Training LIFT Dual Timestep Model - 100 epochs"
    echo "  - Hidden dims: $HIDDEN_DIMS"
    echo "  - Epochs: $EPOCHS"
    echo "  - Save every: $SAVE_EVERY epochs"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 1"
    echo "=========================================="

    $PYTHON train_lift.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --hidden_dims $HIDDEN_DIMS \
        --output_dir $OUTPUT_DIR \
        --save_every $SAVE_EVERY \
        --device 1 \
        2>&1 | tee $LOG_DIR/train_lift_100ep_$TIMESTAMP.log

    # Rename final checkpoint to indicate 100 epochs
    if [ -f "$OUTPUT_DIR/lift_dual_timestep_final.pth" ]; then
        mv $OUTPUT_DIR/lift_dual_timestep_final.pth $OUTPUT_DIR/lift_100ep.pth
        echo "Final checkpoint saved as: $OUTPUT_DIR/lift_100ep.pth"
    fi
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
        echo "100 Epoch Training Experiment"
        echo "=========================================="
        echo "Hypothesis: LIFT model needs more epochs to converge"
        echo ""
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
        echo "100 Epoch Training Complete!"
        echo "=========================================="
        echo "  - Baseline: $([ $BASELINE_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo "  - LIFT: $([ $LIFT_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo ""
        echo "Checkpoints saved to: $OUTPUT_DIR/"
        echo "  Baseline:"
        echo "    - baseline_20ep.pth"
        echo "    - baseline_40ep.pth"
        echo "    - baseline_60ep.pth"
        echo "    - baseline_80ep.pth"
        echo "    - baseline_100ep.pth"
        echo "  LIFT:"
        echo "    - lift_dual_timestep_20ep.pth"
        echo "    - lift_dual_timestep_40ep.pth"
        echo "    - lift_dual_timestep_60ep.pth"
        echo "    - lift_dual_timestep_80ep.pth"
        echo "    - lift_100ep.pth"
        echo ""
        echo "Logs saved to: $LOG_DIR/"
        echo "  - train_baseline_100ep_$TIMESTAMP.log"
        echo "  - train_lift_100ep_$TIMESTAMP.log"
        echo ""
        echo "Next steps:"
        echo "  1. Run FID evaluation on each checkpoint to track progression"
        echo "  2. Compare FID curves: Baseline vs LIFT across epochs"
        echo "  3. Identify at which epoch LIFT surpasses Baseline"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [baseline|lift|both]"
        exit 1
        ;;
esac
