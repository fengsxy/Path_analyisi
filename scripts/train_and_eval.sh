#!/bin/bash
# Train models and automatically evaluate all checkpoints
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Automated Training and Evaluation Pipeline"
echo "=========================================="
echo "Step 1: Train models (100 epochs, ~40 minutes)"
echo "Step 2: Evaluate all checkpoints"
echo "=========================================="
echo ""

# Step 1: Train
echo "Starting training..."
./tmp/train100.sh

# Step 2: Evaluate
echo ""
echo "Training complete! Starting evaluation..."
./tmp/eval_checkpoints.sh

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Next: Run 'python tmp/analyze_fid_progression.py' to analyze results"
