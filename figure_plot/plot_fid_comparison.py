#!/usr/bin/env python3
"""
Plot FID comparison across training epochs.
"""

import matplotlib.pyplot as plt
import numpy as np

# FID results
epochs = [20, 40, 60, 80, 100]

baseline = [99.11, 56.47, 70.18, 44.77, 40.53]
lift_diagonal = [136.96, 61.11, 64.46, 62.88, 66.53]
lift_dp64 = [149.13, 78.81, 60.24, 57.46, 71.66]

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot lines
ax.plot(epochs, baseline, 'o-', linewidth=2.5, markersize=10,
        label='Baseline', color='#2E86AB', markeredgewidth=2, markeredgecolor='white')
ax.plot(epochs, lift_diagonal, 's-', linewidth=2.5, markersize=10,
        label='LIFT Diagonal (γ₁=γ₀)', color='#A23B72', markeredgewidth=2, markeredgecolor='white')
ax.plot(epochs, lift_dp64, '^-', linewidth=2.5, markersize=10,
        label='LIFT DP-64 (optimize 64×64)', color='#F18F01', markeredgewidth=2, markeredgecolor='white')

# Mark best points
baseline_best_idx = np.argmin(baseline)
lift_dp64_best_idx = np.argmin(lift_dp64)

ax.scatter([epochs[baseline_best_idx]], [baseline[baseline_best_idx]],
           s=300, c='#2E86AB', marker='*', edgecolors='red', linewidths=2, zorder=5)
ax.scatter([epochs[lift_dp64_best_idx]], [lift_dp64[lift_dp64_best_idx]],
           s=300, c='#F18F01', marker='*', edgecolors='red', linewidths=2, zorder=5)

# Add value labels
for i, (e, b, ld, ldp) in enumerate(zip(epochs, baseline, lift_diagonal, lift_dp64)):
    ax.text(e, b-3, f'{b:.1f}', ha='center', va='top', fontsize=9, fontweight='bold', color='#2E86AB')
    ax.text(e, ld+3, f'{ld:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#A23B72')
    ax.text(e, ldp+3, f'{ldp:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#F18F01')

# Styling
ax.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
ax.set_ylabel('FID Score ↓ (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('FID Progression: Baseline vs LIFT (100 Epochs Training)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(epochs)
ax.set_ylim([35, 155])

# Add annotations
ax.annotate('Baseline Best\n40.53 @ 100ep',
            xy=(100, 40.53), xytext=(85, 50),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='#2E86AB',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E86AB', linewidth=2))

ax.annotate('LIFT Best\n57.46 @ 80ep',
            xy=(80, 57.46), xytext=(65, 75),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='#F18F01',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#F18F01', linewidth=2))

ax.annotate('LIFT overfitting?\n80→100ep degradation',
            xy=(100, 71.66), xytext=(105, 90),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=10, color='gray',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', linewidth=1))

plt.tight_layout()
plt.savefig('figures/fid_progression_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved to: figures/fid_progression_comparison.png")
plt.close()

# Create summary table
print("\n" + "="*70)
print("FID PROGRESSION SUMMARY")
print("="*70)
print(f"{'Epoch':<8} {'Baseline':<12} {'LIFT Diagonal':<16} {'LIFT DP-64':<12} {'Best':<10}")
print("-"*70)
for i, e in enumerate(epochs):
    scores = [
        ('Baseline', baseline[i]),
        ('Diagonal', lift_diagonal[i]),
        ('DP-64', lift_dp64[i])
    ]
    best_name, best_score = min(scores, key=lambda x: x[1])

    print(f"{e:<8} {baseline[i]:<12.2f} {lift_diagonal[i]:<16.2f} {lift_dp64[i]:<12.2f} {best_name:<10}")

print("-"*70)
print(f"\nOverall Best: Baseline 100ep = {min(baseline):.2f}")
print(f"LIFT Best: DP-64 80ep = {min(lift_dp64):.2f}")
print(f"Gap: {min(lift_dp64) - min(baseline):.2f} FID points")
print("="*70)
