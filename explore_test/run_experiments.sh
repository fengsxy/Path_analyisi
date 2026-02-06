#!/bin/bash
# 运行 explore_test 实验
#
# Usage:
#   ./run_experiments.sh              # 运行两个实验（并行）
#   ./run_experiments.sh single_t     # 只运行 single_t
#   ./run_experiments.sh no_t         # 只运行 no_t

set -e

cd "$(dirname "$0")"

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
EPOCHS=10
BATCH_SIZE=64

mkdir -p checkpoints
mkdir -p results

run_single_t() {
    echo "=========================================="
    echo "Training single_t model (10 epochs)"
    echo "=========================================="
    $PYTHON train.py \
        --model single_t \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --device 2

    echo "Generating images..."
    $PYTHON generate.py \
        --checkpoint checkpoints/single_t_final.pth \
        --output_dir results/fid_single_t \
        --num_images 1000 \
        --device 2

    echo "Computing FID..."
    python -m pytorch_fid ../results/fid_real results/fid_single_t
}

run_no_t() {
    echo "=========================================="
    echo "Training no_t model (10 epochs)"
    echo "=========================================="
    $PYTHON train.py \
        --model no_t \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --device 3

    echo "Generating images..."
    $PYTHON generate.py \
        --checkpoint checkpoints/no_t_final.pth \
        --output_dir results/fid_no_t \
        --num_images 1000 \
        --device 3

    echo "Computing FID..."
    python -m pytorch_fid ../results/fid_real results/fid_no_t
}

case "${1:-both}" in
    single_t)
        run_single_t
        ;;
    no_t)
        run_no_t
        ;;
    both)
        echo "Running both experiments in parallel..."
        run_single_t &
        PID1=$!
        run_no_t &
        PID2=$!
        wait $PID1
        wait $PID2
        echo ""
        echo "=========================================="
        echo "All experiments completed!"
        echo "=========================================="
        ;;
    *)
        echo "Usage: $0 [single_t|no_t|both]"
        exit 1
        ;;
esac
