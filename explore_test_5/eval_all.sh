#!/bin/bash
# Evaluate all explore_test_5 models (diagonal + DP) in parallel across GPUs.
#
# Usage:
#   ./eval_all.sh                    # Default: all epochs
#   ./eval_all.sh 200 400 600        # Specific epochs
#   ./eval_all.sh --force            # Force re-evaluation
#
# GPU assignment:
#   GPU 0: same_t (diagonal → DP)
#   GPU 1: dp_path (diagonal → DP)
#   GPU 2: heuristic (diagonal → DP)

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
echo "Explore Test 5 - Full Evaluation Pipeline"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Force: ${FORCE:-no}"
echo ""

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

cd "$(dirname "$0")"

# GPU 0: same_t (diagonal then DP)
(
    echo "[GPU 0] Starting same_t evaluation..."
    python eval_fid_batch.py --model_type same_t --epochs $EPOCHS --device 0 $FORCE
    echo "[GPU 0] same_t diagonal done. Starting DP..."
    python eval_dp.py --model_type same_t --epochs $EPOCHS --device 0 $FORCE
    echo "[GPU 0] same_t DP done."
) &
PID1=$!

# GPU 1: dp_path (diagonal then DP)
(
    echo "[GPU 1] Starting dp_path evaluation..."
    python eval_fid_batch.py --model_type dp_path --epochs $EPOCHS --device 1 $FORCE
    echo "[GPU 1] dp_path diagonal done. Starting DP..."
    python eval_dp.py --model_type dp_path --epochs $EPOCHS --device 1 $FORCE
    echo "[GPU 1] dp_path DP done."
) &
PID2=$!

# GPU 2: heuristic (diagonal then DP)
(
    echo "[GPU 2] Starting heuristic evaluation..."
    python eval_fid_batch.py --model_type heuristic --epochs $EPOCHS --device 2 $FORCE
    echo "[GPU 2] heuristic diagonal done. Starting DP..."
    python eval_dp.py --model_type heuristic --epochs $EPOCHS --device 2 $FORCE
    echo "[GPU 2] heuristic DP done."
) &
PID3=$!

echo "Launched: same_t on GPU 0 (PID=$PID1), dp_path on GPU 1 (PID=$PID2), heuristic on GPU 2 (PID=$PID3)"
echo "Waiting for all jobs to complete..."

wait $PID1
echo "same_t evaluation complete."

wait $PID2
echo "dp_path evaluation complete."

wait $PID3
echo "heuristic evaluation complete."

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results in: results/"
echo "=============================================="
