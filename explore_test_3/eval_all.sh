#!/bin/bash
# Evaluate LIFT with Greedy Adaptive Path Sampling.
#
# Usage:
#   ./eval_all.sh [epochs...] [--force] [--ema] [--device N] [--skip-dp] [--bps VALUE]
#
# Examples:
#   ./eval_all.sh --ema                          # All epochs, EMA, default device
#   ./eval_all.sh --ema 200 400 --device 1       # Specific epochs on GPU 1
#   ./eval_all.sh --ema --force --skip-dp        # Force re-eval, skip DP comparison
#   ./eval_all.sh --ema --bps 2.5                # Custom BPS budget

set -e

cd "$(dirname "$0")"

# Parse arguments
EPOCHS=()
FORCE=""
EMA=""
DEVICE=0
SKIP_DP=""
BPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE="--force"; shift ;;
        --ema) EMA="--ema"; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --skip-dp) SKIP_DP="--skip_dp"; shift ;;
        --bps) BPS="--bps $2"; shift 2 ;;
        *) EPOCHS+=("$1"); shift ;;
    esac
done

# Default epochs if none specified
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)
fi

echo "=================================================="
echo "Greedy Adaptive Path Evaluation"
echo "=================================================="
echo "Epochs: ${EPOCHS[*]}"
echo "Device: $DEVICE"
echo "EMA: ${EMA:-no}"
echo "Force: ${FORCE:-no}"
echo "Skip DP: ${SKIP_DP:-no}"
echo "BPS: ${BPS:-auto}"
echo ""

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

python eval_greedy.py \
    --epochs ${EPOCHS[*]} \
    --device $DEVICE \
    $EMA $FORCE $SKIP_DP $BPS
