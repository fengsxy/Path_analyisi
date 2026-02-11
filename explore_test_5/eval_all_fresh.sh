#!/bin/bash
# Full re-evaluation of all explore_test_5 models on GPUs 1 and 3.
# Runs diagonal + DP for all 3 models across all epochs with --force.

set -e

EPOCHS="200 400 600 800 1000 1200 1400 1600 1800 2000"

echo "=============================================="
echo "Explore Test 5 - Full Re-Evaluation"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "GPUs: 1 (same_t + heuristic), 3 (dp_path)"
echo "Started: $(date)"
echo ""

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

cd "$(dirname "$0")"

# Clear old results
rm -f results/fid_*_results.csv

# GPU 1: same_t diagonal → same_t DP → heuristic diagonal → heuristic DP
(
    echo "[GPU 1] Starting same_t diagonal... $(date)"
    python eval_fid_batch.py --model_type same_t --epochs $EPOCHS --device 1 --force
    echo "[GPU 1] same_t diagonal done. Starting same_t DP... $(date)"
    python eval_dp.py --model_type same_t --epochs $EPOCHS --device 1 --force
    echo "[GPU 1] same_t DP done. Starting heuristic diagonal... $(date)"
    python eval_fid_batch.py --model_type heuristic --epochs $EPOCHS --device 1 --force
    echo "[GPU 1] heuristic diagonal done. Starting heuristic DP... $(date)"
    python eval_dp.py --model_type heuristic --epochs $EPOCHS --device 1 --force
    echo "[GPU 1] All done. $(date)"
) 2>&1 | tee results/eval_gpu1.log &
PID1=$!

# GPU 3: dp_path diagonal → dp_path DP
(
    echo "[GPU 3] Starting dp_path diagonal... $(date)"
    python eval_fid_batch.py --model_type dp_path --epochs $EPOCHS --device 3 --force
    echo "[GPU 3] dp_path diagonal done. Starting dp_path DP... $(date)"
    python eval_dp.py --model_type dp_path --epochs $EPOCHS --device 3 --force
    echo "[GPU 3] All done. $(date)"
) 2>&1 | tee results/eval_gpu3.log &
PID3=$!

echo "Launched: GPU 1 (PID=$PID1), GPU 3 (PID=$PID3)"
echo "Waiting for all jobs to complete..."

wait $PID1
echo "GPU 1 jobs complete. $(date)"

wait $PID3
echo "GPU 3 jobs complete. $(date)"

echo ""
echo "=============================================="
echo "All evaluations complete! $(date)"
echo "=============================================="
echo ""

# Print summary
echo "=== Diagonal FID Results ==="
for model in same_t dp_path heuristic; do
    echo "--- $model ---"
    if [ -f "results/fid_${model}_diag_results.csv" ]; then
        cat "results/fid_${model}_diag_results.csv"
    fi
    echo ""
done

echo "=== DP FID Results ==="
for model in same_t dp_path heuristic; do
    echo "--- $model ---"
    if [ -f "results/fid_${model}_dp_results.csv" ]; then
        cat "results/fid_${model}_dp_results.csv"
    fi
    echo ""
done
