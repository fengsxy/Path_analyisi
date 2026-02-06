#!/usr/bin/env python3
"""
Analyze and visualize FID progression across training epochs.

This script helps identify when LIFT surpasses Baseline performance.

Usage:
    python analyze_fid_progression.py --results_dir results
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_fid_from_output(output_text):
    """Extract FID score from pytorch-fid output."""
    # pytorch-fid outputs: "FID:  XX.XX"
    match = re.search(r'FID:\s+([\d.]+)', output_text)
    if match:
        return float(match.group(1))
    return None


def collect_fid_scores(results_dir):
    """
    Collect FID scores from result directories.

    Returns:
        dict: {
            'baseline': {20: fid_score, 40: fid_score, ...},
            'lift_diagonal': {20: fid_score, 40: fid_score, ...},
            'lift_dp64': {20: fid_score, 40: fid_score, ...}
        }
    """
    results = {
        'baseline': {},
        'lift_diagonal': {},
        'lift_dp64': {}
    }

    epochs = [20, 40, 60, 80, 100]

    # Note: This function expects FID scores to be manually entered
    # or computed separately. For automation, you'd need to parse
    # the output from the evaluation script.

    print("Please enter FID scores for each checkpoint:")
    print("(Press Enter to skip if not available)")
    print()

    # Baseline
    print("=== Baseline ===")
    for epoch in epochs:
        try:
            score = input(f"Baseline {epoch}ep FID: ").strip()
            if score:
                results['baseline'][epoch] = float(score)
        except ValueError:
            pass
    print()

    # LIFT Diagonal
    print("=== LIFT Diagonal Path ===")
    for epoch in epochs:
        try:
            score = input(f"LIFT Diagonal {epoch}ep FID: ").strip()
            if score:
                results['lift_diagonal'][epoch] = float(score)
        except ValueError:
            pass
    print()

    # LIFT DP-64
    print("=== LIFT DP-64 Path ===")
    for epoch in epochs:
        try:
            score = input(f"LIFT DP-64 {epoch}ep FID: ").strip()
            if score:
                results['lift_dp64'][epoch] = float(score)
        except ValueError:
            pass
    print()

    return results


def plot_fid_progression(results, output_path):
    """Plot FID progression curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Baseline
    if results['baseline']:
        epochs = sorted(results['baseline'].keys())
        fids = [results['baseline'][e] for e in epochs]
        ax.plot(epochs, fids, 'o-', linewidth=2, markersize=8,
                label='Baseline', color='#2E86AB')

    # Plot LIFT Diagonal
    if results['lift_diagonal']:
        epochs = sorted(results['lift_diagonal'].keys())
        fids = [results['lift_diagonal'][e] for e in epochs]
        ax.plot(epochs, fids, 's-', linewidth=2, markersize=8,
                label='LIFT Diagonal', color='#A23B72')

    # Plot LIFT DP-64
    if results['lift_dp64']:
        epochs = sorted(results['lift_dp64'].keys())
        fids = [results['lift_dp64'][e] for e in epochs]
        ax.plot(epochs, fids, '^-', linewidth=2, markersize=8,
                label='LIFT DP-64', color='#F18F01')

    ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID Score ↓', fontsize=12, fontweight='bold')
    ax.set_title('FID Progression: Baseline vs LIFT', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([20, 40, 60, 80, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def find_crossover_point(results):
    """Find when LIFT surpasses Baseline."""
    baseline_epochs = sorted(results['baseline'].keys())

    for model_name in ['lift_diagonal', 'lift_dp64']:
        lift_epochs = sorted(results[model_name].keys())

        if not lift_epochs:
            continue

        print(f"\n=== {model_name.replace('_', ' ').title()} ===")

        for epoch in lift_epochs:
            lift_fid = results[model_name][epoch]

            # Find closest baseline epoch
            baseline_epoch = min(baseline_epochs, key=lambda e: abs(e - epoch))
            baseline_fid = results['baseline'][baseline_epoch]

            diff = baseline_fid - lift_fid
            if diff > 0:
                print(f"✓ Epoch {epoch}: LIFT ({lift_fid:.2f}) beats Baseline ({baseline_fid:.2f}) by {diff:.2f}")
            else:
                print(f"  Epoch {epoch}: LIFT ({lift_fid:.2f}) behind Baseline ({baseline_fid:.2f}) by {-diff:.2f}")


def generate_summary_table(results):
    """Generate a markdown summary table."""
    print("\n" + "="*60)
    print("FID PROGRESSION SUMMARY")
    print("="*60)
    print()

    epochs = sorted(set(
        list(results['baseline'].keys()) +
        list(results['lift_diagonal'].keys()) +
        list(results['lift_dp64'].keys())
    ))

    print("| Epoch | Baseline | LIFT Diagonal | LIFT DP-64 | Best |")
    print("|-------|----------|---------------|------------|------|")

    for epoch in epochs:
        baseline = results['baseline'].get(epoch, None)
        diagonal = results['lift_diagonal'].get(epoch, None)
        dp64 = results['lift_dp64'].get(epoch, None)

        # Find best
        scores = []
        if baseline is not None:
            scores.append(('Baseline', baseline))
        if diagonal is not None:
            scores.append(('Diagonal', diagonal))
        if dp64 is not None:
            scores.append(('DP-64', dp64))

        best_name, best_score = min(scores, key=lambda x: x[1]) if scores else (None, None)

        baseline_str = f"{baseline:.2f}" if baseline is not None else "-"
        diagonal_str = f"{diagonal:.2f}" if diagonal is not None else "-"
        dp64_str = f"{dp64:.2f}" if dp64 is not None else "-"
        best_str = f"{best_name}" if best_name else "-"

        print(f"| {epoch:5d} | {baseline_str:8s} | {diagonal_str:13s} | {dp64_str:10s} | {best_str:4s} |")

    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze FID progression')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory')
    parser.add_argument('--output', type=str, default='figures/fid_progression.png',
                        help='Output plot path')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Collect FID scores
    results = collect_fid_scores(args.results_dir)

    # Generate summary table
    generate_summary_table(results)

    # Find crossover point
    find_crossover_point(results)

    # Plot progression
    plot_fid_progression(results, args.output)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
