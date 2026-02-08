#!/bin/bash
# Evaluate all EMA models in parallel using 3 GPUs
# Usage:
#   ./scripts/eval_all_ema.sh              # Single run with seed=42
#   ./scripts/eval_all_ema.sh --force      # Force re-evaluation
#   ./scripts/eval_all_ema.sh --multi-seed # Run 5 seeds and average
#
# This script runs:
#   - GPU 0: Baseline EMA evaluation
#   - GPU 1: LIFT EMA Diagonal evaluation
#   - GPU 2: LIFT EMA DP (DP-64 + DP-Total) evaluation

set -e

# Parse arguments
FORCE_FLAG=""
MULTI_SEED=false
for arg in "$@"; do
    case $arg in
        --force)
            FORCE_FLAG="--force"
            echo "Force mode enabled: will re-evaluate all epochs"
            ;;
        --multi-seed)
            MULTI_SEED=true
            echo "Multi-seed mode enabled: will run 5 seeds and average"
            ;;
    esac
done

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

EPOCHS="200 400 600 800 1000 1200 1400 1600 1800 2000"
NUM_IMAGES=15803
NUM_STEPS=18

if [ "$MULTI_SEED" = true ]; then
    SEEDS="42 43 44 45 46"
else
    SEEDS="42"
fi

echo "=============================================="
echo "EMA Evaluation Pipeline (3 GPUs in parallel)"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Seeds: $SEEDS"
echo "Images: $NUM_IMAGES"
echo "Steps: $NUM_STEPS"
echo ""

mkdir -p results

for SEED in $SEEDS; do
    echo ""
    echo "############################################"
    echo "# Running with seed=$SEED"
    echo "############################################"
    echo ""

    if [ "$MULTI_SEED" = true ]; then
        SUFFIX="_seed${SEED}"
        OUTPUT_DIR="results/seed${SEED}"
        mkdir -p $OUTPUT_DIR
    else
        SUFFIX=""
        OUTPUT_DIR="results"
    fi

    echo "[GPU 0] Starting Baseline EMA evaluation (seed=$SEED)..."
    python eval_fid_batch.py \
        --model_type baseline --ema \
        --epochs $EPOCHS \
        --seed $SEED \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --device 0 \
        --output_dir $OUTPUT_DIR \
        $FORCE_FLAG \
        > results/eval_baseline_ema${SUFFIX}.log 2>&1 &
    PID_BASELINE=$!

    echo "[GPU 1] Starting LIFT EMA Diagonal evaluation (seed=$SEED)..."
    python eval_fid_batch.py \
        --model_type lift --ema \
        --epochs $EPOCHS \
        --seed $SEED \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --device 1 \
        --output_dir $OUTPUT_DIR \
        $FORCE_FLAG \
        > results/eval_lift_ema_diagonal${SUFFIX}.log 2>&1 &
    PID_LIFT=$!

    echo "[GPU 2] Starting LIFT EMA DP evaluation (seed=$SEED)..."
    python eval_lift_dp.py \
        --ema \
        --epochs $EPOCHS \
        --seed $SEED \
        --num_images $NUM_IMAGES \
        --num_steps $NUM_STEPS \
        --device 2 \
        --output_dir $OUTPUT_DIR \
        $FORCE_FLAG \
        > results/eval_lift_ema_dp${SUFFIX}.log 2>&1 &
    PID_DP=$!

    echo ""
    echo "Jobs started for seed=$SEED:"
    echo "  Baseline EMA (GPU 0): PID $PID_BASELINE"
    echo "  LIFT EMA Diagonal (GPU 1): PID $PID_LIFT"
    echo "  LIFT EMA DP (GPU 2): PID $PID_DP"
    echo ""

    echo "Waiting for seed=$SEED jobs to complete..."
    wait $PID_BASELINE
    echo "[GPU 0] Baseline EMA (seed=$SEED) completed."

    wait $PID_LIFT
    echo "[GPU 1] LIFT EMA Diagonal (seed=$SEED) completed."

    wait $PID_DP
    echo "[GPU 2] LIFT EMA DP (seed=$SEED) completed."

done

echo ""
echo "=============================================="
echo "All EMA evaluations completed!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  results/fid_baseline_ema_results.csv"
echo "  results/fid_lift_ema_results.csv"
echo "  results/fid_lift_ema_dp_results.csv"
