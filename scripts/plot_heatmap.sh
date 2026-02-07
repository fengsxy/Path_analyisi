#!/bin/bash
# Step 2: Plot heatmap with optimal paths
#
# Usage:
#   ./scripts/plot_heatmap.sh 800 1000 2000    # Specific epochs
#   ./scripts/plot_heatmap.sh                   # All epochs
#
# Output:
#   - results/heatmap_30_XXXep.png (comparison plot)
#   - results/heatmap_30_XXXep_64.png (64x64 error)
#   - results/heatmap_30_XXXep_total.png (total error)

set -e

PYTHON="/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python"
NUM_STEPS=18
RESULTS_DIR="results"

ALL_EPOCHS=(200 400 600 800 1000 1200 1400 1600 1800 2000)

if [ $# -gt 0 ]; then
    EPOCHS=("$@")
else
    EPOCHS=("${ALL_EPOCHS[@]}")
fi

echo "=========================================="
echo "Step 2: Plot Heatmaps with Paths"
echo "=========================================="
echo "Epochs: ${EPOCHS[*]}"
echo ""

$PYTHON - ${EPOCHS[@]} << 'EOF'
import sys
import torch
import numpy as np
from compute_heatmap_30 import (
    find_optimal_path,
    find_optimal_path_n_steps,
    path_to_timesteps,
    plot_heatmap_with_path,
    plot_comparison
)

epochs = [int(e) for e in sys.argv[1:]]
num_steps = 18
results_dir = "results"

for epoch in epochs:
    print(f"\n=== Epoch {epoch} ===")

    heatmap_path = f"{results_dir}/heatmap_30_{epoch}ep.pth"

    try:
        data = torch.load(heatmap_path, weights_only=False)
    except FileNotFoundError:
        print(f"[Skip] Heatmap not found: {heatmap_path}")
        continue

    t_grid = data['t_grid']
    error_64 = data['error_64']
    error_total = data['error_total']

    # Find full paths (58 steps)
    path_64, cost_64 = find_optimal_path(error_64)
    path_total, cost_total = find_optimal_path(error_total)

    # Find N-step paths
    samples_64, cost_64_n = find_optimal_path_n_steps(error_64, num_steps)
    samples_total, cost_total_n = find_optimal_path_n_steps(error_total, num_steps)

    # Convert to timesteps
    ts_64_64, ts_64_32 = path_to_timesteps(samples_64, t_grid)
    ts_total_64, ts_total_32 = path_to_timesteps(samples_total, t_grid)

    print(f"DP-64: {len(samples_64)} points, t_64: {ts_64_64[0]}->{ts_64_64[-1]}, t_32: {ts_64_32[0]}->{ts_64_32[-1]}")
    print(f"DP-Total: {len(samples_total)} points, t_64: {ts_total_64[0]}->{ts_total_64[-1]}, t_32: {ts_total_32[0]}->{ts_total_32[-1]}")

    # Plot
    base = f"{results_dir}/heatmap_30_{epoch}ep"

    plot_heatmap_with_path(t_grid, error_64, path_64, samples_64,
                           f"{base}_64.png", f"64×64 Error (Epoch {epoch})")

    plot_heatmap_with_path(t_grid, error_total, path_total, samples_total,
                           f"{base}_total.png", f"Total Error (Epoch {epoch})")

    plot_comparison(t_grid, error_64, error_total, path_64, path_total,
                    samples_64, samples_total, f"{base}.png")

print("\nDone!")
EOF

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
