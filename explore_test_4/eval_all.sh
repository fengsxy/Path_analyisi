#!/bin/bash
# Evaluate LIFT as 32→64 super-resolution model.
#
# Usage:
#   ./eval_all.sh [epochs...] [--force] [--ema] [--device N] [--num_images N]
#
# Examples:
#   ./eval_all.sh --ema                          # All epochs, EMA
#   ./eval_all.sh --ema 200 400 --device 1       # Specific epochs on GPU 1
#   ./eval_all.sh --ema --force --num_images 50   # Force re-eval, 50 images

set -e

cd "$(dirname "$0")"

# Parse arguments
EPOCHS=()
FORCE=""
EMA=""
DEVICE=0
NUM_IMAGES=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE="--force"; shift ;;
        --ema) EMA="--ema"; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num_images) NUM_IMAGES="$2"; shift 2 ;;
        *) EPOCHS+=("$1"); shift ;;
    esac
done

# Default epochs if none specified
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)
fi

echo "=================================================="
echo "LIFT Super-Resolution Evaluation (32→64)"
echo "=================================================="
echo "Epochs: ${EPOCHS[*]}"
echo "Device: $DEVICE"
echo "EMA: ${EMA:-no}"
echo "Force: ${FORCE:-no}"
echo "Num images: $NUM_IMAGES"
echo ""

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion-gpu

python eval_sr.py \
    --epochs ${EPOCHS[*]} \
    --device $DEVICE \
    --num_images $NUM_IMAGES \
    $EMA $FORCE
