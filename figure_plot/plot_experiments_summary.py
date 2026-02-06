#!/usr/bin/env python3
"""
Create visualization for Experiments 1-4 results
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    'Baseline\n(单尺度)',
    'Exp2\nDiagonal',
    'LIFT\nDP 64×64',
    'LIFT\nDP Total',
    'LIFT\nOriginal',
    'Exp1\nloss_64 only',
    'Exp4\nRandom32',
    'Exp4\nFixed32'
]

fids = [78.83, 94.79, 101.53, 109.20, 116.14, 212.79, 300.68, 291.77]
colors = ['green', 'lightgreen', 'orange', 'orange', 'orange', 'red', 'darkred', 'darkred']

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Bar chart
bars = ax1.bar(range(len(models)), fids, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
ax1.set_title('Experiment Results: FID Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.axhline(y=78.83, color='green', linestyle='--', linewidth=2, label='Baseline')
ax1.axhline(y=116.14, color='orange', linestyle='--', linewidth=2, label='LIFT Original')
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# Add value labels on bars
for i, (bar, fid) in enumerate(zip(bars, fids)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{fid:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Relative improvement
baseline_fid = 78.83
lift_original_fid = 116.14

improvements = [(fid - lift_original_fid) / lift_original_fid * 100 for fid in fids]
colors2 = ['green' if imp < 0 else 'red' for imp in improvements]

bars2 = ax2.barh(range(len(models)), improvements, color=colors2, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Model', fontsize=12, fontweight='bold')
ax2.set_xlabel('Change vs LIFT Original (%)', fontsize=12, fontweight='bold')
ax2.set_title('Relative Performance vs LIFT Original', fontsize=14, fontweight='bold')
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, imp) in enumerate(zip(bars2, improvements)):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
             f'{imp:+.1f}%',
             ha='left' if width > 0 else 'right',
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/experiments_summary.png', dpi=300, bbox_inches='tight')
print("Saved: figures/experiments_summary.png")

# Create a second figure for training loss comparison
fig2, ax = plt.subplots(figsize=(10, 6))

experiments = ['LIFT\nOriginal', 'Exp1\nloss_64 only', 'Exp2\nDiagonal']
loss_64_values = [0.0440, 0.0433, 0.0446]  # Approximate from logs
loss_32_values = [0.0440, 1.0342, 0.0237]  # Approximate from logs

x = np.arange(len(experiments))
width = 0.35

bars1 = ax.bar(x - width/2, loss_64_values, width, label='loss_64', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, loss_32_values, width, label='loss_32', color='coral', alpha=0.8)

ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/training_loss_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: figures/training_loss_comparison.png")

print("\nSummary:")
print("=" * 60)
print(f"Best model: Baseline (FID={fids[0]:.2f})")
print(f"Best LIFT: Exp2 Diagonal (FID={fids[1]:.2f}, {improvements[1]:.1f}% vs Original)")
print(f"Worst: Exp4 Random32 (FID={fids[6]:.2f}, {improvements[6]:.1f}% vs Original)")
print("=" * 60)
