#!/bin/bash
# 训练 explore_test 模型 2000 epochs
#
# Usage:
#   ./train_2000.sh              # 并行训练两个模型
#   ./train_2000.sh single_t     # 只训练 single_t
#   ./train_2000.sh no_t         # 只训练 no_t

set -e

cd "$(dirname "$0")"

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
EPOCHS=2000
SAVE_EVERY=200
BATCH_SIZE=256

mkdir -p checkpoints
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

train_single_t() {
    echo "=========================================="
    echo "Training single_t model - 2000 epochs"
    echo "  - Epochs: $EPOCHS"
    echo "  - Save every: $SAVE_EVERY epochs"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 1"
    echo "=========================================="

    $PYTHON train.py \
        --model single_t \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --save_every $SAVE_EVERY \
        --device 1 \
        2>&1 | tee logs/train_single_t_2000ep_$TIMESTAMP.log
}

train_no_t() {
    echo "=========================================="
    echo "Training no_t model - 2000 epochs"
    echo "  - Epochs: $EPOCHS"
    echo "  - Save every: $SAVE_EVERY epochs"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Device: GPU 1"
    echo "=========================================="

    $PYTHON train.py \
        --model no_t \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --save_every $SAVE_EVERY \
        --device 1 \
        2>&1 | tee logs/train_no_t_2000ep_$TIMESTAMP.log
}

case "${1:-both}" in
    single_t)
        train_single_t
        ;;
    no_t)
        train_no_t
        ;;
    both)
        echo "=========================================="
        echo "2000 Epoch Training - explore_test"
        echo "=========================================="
        echo "Parallel training on GPU 1..."
        echo "  - single_t and no_t together"
        echo ""

        train_single_t &
        PID1=$!

        train_no_t &
        PID2=$!

        echo "Waiting for training to complete..."
        wait $PID1
        STATUS1=$?

        wait $PID2
        STATUS2=$?

        echo ""
        echo "=========================================="
        echo "Training Complete!"
        echo "=========================================="
        echo "  - single_t: $([ $STATUS1 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo "  - no_t: $([ $STATUS2 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
        echo ""
        echo "Checkpoints saved to: checkpoints/"
        echo "  - single_t_200ep.pth, single_t_400ep.pth, ..., single_t_final.pth"
        echo "  - no_t_200ep.pth, no_t_400ep.pth, ..., no_t_final.pth"
        echo ""
        echo "Next: python test_pipeline.py --device 1"
        ;;
    *)
        echo "Usage: $0 [single_t|no_t|both]"
        exit 1
        ;;
esac
