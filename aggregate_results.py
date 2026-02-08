#!/usr/bin/env python3
"""
Aggregate FID results across multiple seeds.

Usage:
    python aggregate_results.py --seeds 42 43 44 45 46
"""

import os
import argparse
import pandas as pd
import numpy as np


def load_results_for_seed(seed, results_dir='results'):
    """Load all FID results for a given seed."""
    seed_dir = os.path.join(results_dir, f'seed{seed}')

    results = {}

    # Load baseline results
    baseline_path = os.path.join(seed_dir, 'fid_baseline_results.csv')
    if os.path.exists(baseline_path):
        df = pd.read_csv(baseline_path)
        for _, row in df.iterrows():
            epoch = int(row['Epoch'])
            if epoch not in results:
                results[epoch] = {}
            results[epoch]['Baseline'] = row['FID']

    # Load LIFT diagonal results
    lift_path = os.path.join(seed_dir, 'fid_lift_results.csv')
    if os.path.exists(lift_path):
        df = pd.read_csv(lift_path)
        for _, row in df.iterrows():
            epoch = int(row['Epoch'])
            if epoch not in results:
                results[epoch] = {}
            results[epoch]['LIFT_Diagonal'] = row['FID']

    # Load LIFT DP results
    dp_path = os.path.join(seed_dir, 'fid_lift_dp_results.csv')
    if os.path.exists(dp_path):
        df = pd.read_csv(dp_path)
        for _, row in df.iterrows():
            epoch = int(row['Epoch'])
            if epoch not in results:
                results[epoch] = {}
            results[epoch]['LIFT_DP64'] = row['FID_DP64']
            results[epoch]['LIFT_DP_Total'] = row['FID_DP_Total']

    return results


def aggregate_results(seeds, results_dir='results'):
    """Aggregate results across all seeds and compute mean/std."""
    all_results = {}

    for seed in seeds:
        seed_results = load_results_for_seed(seed, results_dir)
        for epoch, metrics in seed_results.items():
            if epoch not in all_results:
                all_results[epoch] = {k: [] for k in ['Baseline', 'LIFT_Diagonal', 'LIFT_DP64', 'LIFT_DP_Total']}
            for metric, value in metrics.items():
                if metric in all_results[epoch]:
                    all_results[epoch][metric].append(value)

    # Compute mean and std
    aggregated = []
    for epoch in sorted(all_results.keys()):
        row = {'Epoch': epoch}
        for metric in ['Baseline', 'LIFT_Diagonal', 'LIFT_DP64', 'LIFT_DP_Total']:
            values = all_results[epoch].get(metric, [])
            if values:
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
                row[f'{metric}_n'] = len(values)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_n'] = 0
        aggregated.append(row)

    return pd.DataFrame(aggregated)


def format_mean_std(mean, std):
    """Format as mean±std."""
    if pd.isna(mean):
        return '-'
    return f'{mean:.2f}±{std:.2f}'


def main():
    parser = argparse.ArgumentParser(description='Aggregate FID results across seeds')
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    print(f"Aggregating results for seeds: {args.seeds}")

    df = aggregate_results(args.seeds, args.results_dir)

    # Save raw results
    output_path = os.path.join(args.results_dir, 'fid_results_mean_std.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Print formatted table
    print("\n" + "=" * 80)
    print("FID Results (mean ± std)")
    print("=" * 80)
    print(f"{'Epoch':>6} {'Baseline':>16} {'LIFT Diagonal':>16} {'LIFT DP-64':>16} {'LIFT DP-Total':>16}")
    print("-" * 80)

    for _, row in df.iterrows():
        epoch = int(row['Epoch'])
        baseline = format_mean_std(row['Baseline_mean'], row['Baseline_std'])
        diagonal = format_mean_std(row['LIFT_Diagonal_mean'], row['LIFT_Diagonal_std'])
        dp64 = format_mean_std(row['LIFT_DP64_mean'], row['LIFT_DP64_std'])
        dp_total = format_mean_std(row['LIFT_DP_Total_mean'], row['LIFT_DP_Total_std'])
        print(f"{epoch:>6} {baseline:>16} {diagonal:>16} {dp64:>16} {dp_total:>16}")

    print("=" * 80)

    # Find best results
    print("\nBest Results:")
    for metric in ['Baseline', 'LIFT_Diagonal', 'LIFT_DP64', 'LIFT_DP_Total']:
        col = f'{metric}_mean'
        if col in df.columns and not df[col].isna().all():
            best_idx = df[col].idxmin()
            best_epoch = int(df.loc[best_idx, 'Epoch'])
            best_mean = df.loc[best_idx, f'{metric}_mean']
            best_std = df.loc[best_idx, f'{metric}_std']
            print(f"  {metric}: {best_mean:.2f}±{best_std:.2f} @ epoch {best_epoch}")


if __name__ == '__main__':
    main()
