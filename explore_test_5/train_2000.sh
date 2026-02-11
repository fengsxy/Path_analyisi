#!/bin/bash
# Train explore_test_5 models (3 regimes) for 2000 epochs
#
# Usage:
#   ./train_2000.sh              # Train all 3 in parallel (3 GPUs)
#   ./train_2000.sh same_t       # Train only same_t
#   ./train_2000.sh dp_path      # Train only dp_path
#   ./train_2000.sh heuristic    # Train only heuristic

set -e

cd "$(dirname "$0")"

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
EPOCHS=2000
SAVE_EVERY=200
BATCH_SIZE=256

mkdir -p checkpoints
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

train_same_t() {
    echo "=========================================="
    echo "Training same_t - 2000 epochs (GPU 1)"
    echo "=========================================="

    $PYTHON train.py \
        --model same_t \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --save_every $SAVE_EVERY \
        --device 1 \
        2>&1 | tee logs/train_same_t_${EPOCHS}ep_$TIMESTAMP.log
}

train_dp_path() {
    echo "=========================================="
    echo "Training dp_path - 2000 epochs (GPU 2)"
    echo "=========================================="

    $PYTHON train.py \
        --model dp_path \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --save_every $SAVE_EVERY \
        --device 2 \
        2>&1 | tee logs/train_dp_path_${EPOCHS}ep_$TIMESTAMP.log
}

train_heuristic() {
    echo "=========================================="
    echo "Training heuristic - 2000 epochs (GPU 3)"
    echo "=========================================="

    $PYTHON train.py \
        --model heuristic \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --save_every $SAVE_EVERY \
        --device 3 \
        2>&1 | tee logs/train_heuristic_${EPOCHS}ep_$TIMESTAMP.log
}

case "${1:-all}" in
    same_t)
        train_same_t
        ;;
    dp_path)
        train_dp_path
        ;;
    heuristic)
        train_heuristic
        ;;
    all)
        echo "=========================================="
        echo "2000 Epoch Training - explore_test_5"
        echo "=========================================="
        echo "Parallel training on 3 GPUs..."
        echo "  GPU 1: same_t"
        echo "  GPU 2: dp_path"
        echo "  GPU 3: heuristic"
        echo ""

        train_same_t &
        PID1=$!

        train_dp_path &
        PID2=$!

        train_heuristic &
        PID3=$!

        echo "Waiting for training to complete..."
        wait $PID1; STATUS1=$?
        wait $PID2; STATUS2=$?
        wait $PID3; STATUS3=$?

        echo ""
        echo "=========================================="
        echo "Training Complete!"
        echo "=========================================="
        echo "  - same_t:     $([ $STATUS1 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo "  - dp_path:    $([ $STATUS2 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo "  - heuristic:  $([ $STATUS3 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo ""
        echo "Checkpoints: checkpoints/{same_t,dp_path,heuristic}_ema_{200..2000}ep.pth"
        ;;
    *)
        echo "Usage: $0 [same_t|dp_path|heuristic|all]"
        exit 1
        ;;
esac
