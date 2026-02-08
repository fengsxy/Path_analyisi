#!/bin/bash
# Evaluate all explore_test models (diagonal + DP) in parallel across GPUs.
#
# Usage:
#   ./eval_all.sh                    # Default: all epochs
#   ./eval_all.sh 200 400 600        # Specific epochs
#   ./eval_all.sh --force             # Force re-evaluation
#
# GPU assignment:
#   GPU 1: single_t diagonal → single_t DP (sequential)
#   GPU 3: no_t diagonal → no_t DP (sequential)

set -e

# Parse arguments
EPOCHS="200 400 600 800 1000 1200 1400 1600 1800 2000"
FORCE=""
CUSTOM_EPOCHS=""

for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE="--force"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        CUSTOM_EPOCHS="$CUSTOM_EPOCHS $arg"
    fi
done

if [ -n "$CUSTOM_EPOCHS" ]; then
    EPOCHS="$CUSTOM_EPOCHS"
fi

echo "=============================================="
echo "Explore Test - Full Evaluation Pipeline"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Force: ${FORCE:-no}"
echo ""

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

cd "$(dirname "$0")"

# GPU 1: single_t (diagonal then DP)
(
    echo "[GPU 1] Starting single_t evaluation..."
    python eval_fid_batch.py --model_type single_t --epochs $EPOCHS --device 1 $FORCE
    echo "[GPU 1] single_t diagonal done. Starting DP..."
    python eval_dp.py --model_type single_t --epochs $EPOCHS --device 1 $FORCE
    echo "[GPU 1] single_t DP done."
) &
PID1=$!

# GPU 3: no_t (diagonal then DP)
(
    echo "[GPU 3] Starting no_t evaluation..."
    python eval_fid_batch.py --model_type no_t --epochs $EPOCHS --device 3 $FORCE
    echo "[GPU 3] no_t diagonal done. Starting DP..."
    python eval_dp.py --model_type no_t --epochs $EPOCHS --device 3 $FORCE
    echo "[GPU 3] no_t DP done."
) &
PID2=$!

echo "Launched: single_t on GPU 1 (PID=$PID1), no_t on GPU 3 (PID=$PID2)"
echo "Waiting for all jobs to complete..."

wait $PID1
echo "single_t evaluation complete."

wait $PID2
echo "no_t evaluation complete."

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results in: results/"
echo "=============================================="
