"""2D DP path optimization for 2-scale SLOT model.

Finds optimal N-step path from (0,0) to (G-1,G-1) in 2D sigma space.
Uses the 2D error heatmap from compute_heatmap_2d.py.

Cost per step = (e_orig[cur] + e_orig[next])/2 × |Δλ_orig|
              + (e_2x[cur] + e_2x[next])/2 × |Δλ_2x|
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


def find_optimal_path_2d(error_orig, error_2x, log_snr,
                         num_steps=18, max_jump=3, max_diag_dist=None,
                         path_type='total'):
    """Find optimal 2D path using DP.

    Args:
        error_orig: [G, G] error for orig scale
        error_2x: [G, G] error for 2x scale
        log_snr: [G] logSNR values
        num_steps: number of steps
        max_jump: max grid cells per step per dimension
        max_diag_dist: if set, max distance from diagonal |i-j|
        path_type: 'total' (both errors) or 'orig' (orig error only)

    Returns:
        path: list of (i, j) tuples, length num_steps+1
        cost: total cost
    """
    eo = error_orig.numpy() if torch.is_tensor(error_orig) else error_orig
    e2 = error_2x.numpy() if torch.is_tensor(error_2x) else error_2x
    lsnr = log_snr.numpy() if torch.is_tensor(log_snr) else log_snr
    G = eo.shape[0]

    if path_type == 'orig':
        e2 = np.zeros_like(e2)

    INF = float('inf')
    dp = np.full((G, G, num_steps + 1), INF, dtype=np.float64)
    parent = np.full((G, G, num_steps + 1, 2), -1, dtype=np.int16)
    dp[0, 0, 0] = 0.0
# PLACEHOLDER_DP_LOOP
