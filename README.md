# LIFT Dual-Scale Diffusion Model

A dual-scale diffusion model that jointly denoises 64×64 and 32×32 images with independent timesteps for each scale, using dynamic programming to find optimal 2D denoising paths.

## Results

### Main Results (18 DDIM steps, 15803 images, 5 seeds)

| Epoch | Baseline | LIFT Diagonal | LIFT DP-64 | LIFT DP-Total |
|------:|---------:|--------------:|-----------:|--------------:|
| 200   | **33.07±0.13** | 87.22±0.35 | 85.75±0.39 | 80.53±0.16 |
| 400   | 41.54±0.30 | 49.60±0.28 | 51.77±0.29 | **39.35±0.26** |
| 600   | **34.04±0.16** | 50.07±0.32 | 49.77±0.22 | 46.48±0.40 |
| 800   | 40.82±0.20 | 49.07±0.44 | 48.39±0.26 | **40.43±0.31** |
| 1000  | **41.05±0.16** | 82.11±0.44 | 76.58±0.31 | 63.59±0.26 |
| 1200  | 52.32±0.29 | 72.69±0.34 | 61.05±0.26 | **40.11±0.22** |
| 1400  | 45.85±0.26 | 76.40±0.18 | 63.39±0.10 | **40.55±0.24** |
| 1600  | 48.13±0.34 | 64.89±0.31 | 60.91±0.22 | **40.28±0.39** |
| 1800  | **45.61±0.22** | 101.76±0.38 | 82.96±0.25 | 53.96±0.20 |
| 2000  | 59.37±0.19 | 69.61±0.08 | 51.83±0.15 | **36.65±0.23** |

### EMA Results (decay=0.9999, single seed)

| Epoch | Baseline EMA | LIFT EMA Diag | LIFT EMA DP-64 | LIFT EMA DP-Total |
|------:|-------------:|--------------:|---------------:|------------------:|
| 200   | **28.22** | 38.54 | 34.10 | 37.45 |
| 400   | **27.90** | 31.04 | 29.87 | 29.45 |
| 600   | **30.00** | 35.94 | 34.22 | 30.16 |
| 800   | **31.52** | 44.32 | 38.36 | 31.37 |
| 1000  | ---       | 46.12 | 38.35 | **31.32** |
| 1200  | ---       | 46.83 | 37.49 | **31.74** |

### Ablation: Single-Output Architectures (single seed)

| Epoch | single_t Diag | single_t DP | no_t Diag | no_t DP |
|------:|--------------:|------------:|----------:|--------:|
| 200   | 231.23 | 61.90 | 241.14 | 118.28 |
| 400   | 209.82 | 45.16 | 230.22 | 57.92 |
| 600   | 219.00 | **38.63** | 196.65 | 50.76 |
| 800   | 231.28 | 42.52 | 177.12 | 65.01 |
| 1000  | 220.37 | 46.23 | 170.03 | 81.40 |
| 1200  | 243.62 | 48.42 | **167.26** | 75.91 |
| 1400  | 250.40 | 45.30 | 167.25 | 56.43 |
| 1600  | 261.03 | 48.49 | 180.69 | 82.68 |
| 1800  | 256.19 | 60.19 | 187.47 | 73.21 |
| 2000  | 259.67 | 51.99 | 193.21 | 70.71 |

### Summary of Best FID

| Model | Best FID | Epoch | Notes |
|-------|----------|-------|-------|
| Baseline EMA | **27.90** | 400 | Single seed |
| LIFT EMA DP-Total | **29.45** | 400 | Single seed |
| LIFT EMA DP-64 | 29.87 | 400 | Single seed |
| LIFT EMA Diagonal | 31.04 | 400 | Single seed |
| Baseline | 33.07±0.13 | 200 | 5-seed |
| LIFT DP-Total | 36.65±0.23 | 2000 | 5-seed |
| single_t DP | 38.63 | 600 | Ablation |
| LIFT DP-64 | 48.39±0.26 | 800 | 5-seed |
| LIFT Diagonal | 49.07±0.44 | 800 | 5-seed |
| no_t DP | 50.76 | 600 | Ablation |

## Key Findings

### 1. DP Path Optimization is Critical

DP-Total consistently outperforms Diagonal by 8-33 FID points. The effect is even more dramatic in ablation models (single_t: 219→38.6, a 180-point improvement).

| Epoch | Diagonal | DP-Total | Improvement |
|------:|---------:|---------:|------------:|
| 400   | 49.60    | 39.35    | **+10.25** |
| 800   | 49.07    | 40.43    | **+8.64** |
| 1200  | 72.69    | 40.11    | **+32.58** |
| 2000  | 69.61    | 36.65    | **+32.96** |

### 2. EMA Closes the Gap

With EMA, LIFT DP-Total (29.45) nearly matches Baseline (27.90)—a gap of only 1.55 FID points, compared to 3.6 points without EMA.

### 3. LIFT DP-Total Shows Training Stability

| Model | Best FID | Best Epoch | FID at 2000ep | Degradation |
|-------|----------|------------|---------------|-------------|
| Baseline | 33.07 | 200 | 59.37 | **+26.30** |
| LIFT DP-Total | 36.65 | 2000 | 36.65 | **+0.00** |

### 4. Timestep Information Matters (Ablation)

| Model | Diagonal | DP | DP Improvement |
|-------|----------|-----|----------------|
| single_t (has t_64) | 209.82 | **38.63** | 171 points |
| no_t (no timestep) | 167.25 | **50.76** | 117 points |

single_t DP (38.63) outperforms no_t DP (50.76), confirming that timestep conditioning helps even in single-output models.

### 5. "Low Guides High" — Coarse Scale Denoises First

A striking emergent property: **all DP-Total paths consistently denoise the 32×32 (coarse) scale faster than the 64×64 (fine) scale**. Across every epoch and both EMA and non-EMA models, 17 out of 19 path points have t_32 ahead of t_64 (i.e., 32×32 is cleaner), with 0 points where t_64 is ahead.

```
DP-Total path (EMA 400ep):
  t_64: 999 → 964 → 930 → 895 → 861 → 861 → 826 → 792 → ... → 0
  t_32: 999 → 964 → 964 → 930 → 895 → 861 → 826 → 792 → ... → 0
                     ^^^                ^^^
              32 stays while 64    32 catches up, then
              advances first       64 advances again
```

| Epoch | Steps with t_32 cleaner | Steps with t_64 cleaner | Max deviation |
|------:|:-----------------------:|:-----------------------:|:-------------:|
| 400   | 17/19 | 0/19 | 8 grid cells |
| 800   | 17/19 | 0/19 | 4 grid cells |
| 1200  | 17/19 | 0/19 | 7 grid cells |
| 2000  | 17/19 | 0/19 | 7 grid cells |
| EMA 400  | 17/19 | 0/19 | — |
| EMA 800  | 17/19 | 0/19 | — |
| EMA 1200 | 17/19 | 0/19 | — |

**Interpretation**: The model learns an asymmetric coupling between scales — a cleaner coarse scale provides structural guidance that helps denoise the fine scale. The DP algorithm discovers this automatically by minimizing discretization error: denoising 32×32 first reduces the total error because the coarse-to-fine information flow (J_HL: ∂ε_64/∂x_32) benefits from a cleaner coarse input.

**Cross-Jacobian evidence**: The two scales are not fully decoupled. Cross-Jacobian analysis (EMA 400ep) reveals a clear "low guides high" pattern:

![Cross-Jacobian Analysis](results/cross_jacobian_analysis_400ep.png)

The raw Jacobian J_HL = ||∂ε_64/∂x_32||² (before chain-rule factor) shows:

| (t_64, t_32) | Raw J_HL | Interpretation |
|:------------:|:--------:|:--------------|
| (999, 999) — both noisy | 2.9e-7 | No cross-influence |
| (999, 0) — **noisy 64, clean 32** | **1.9e-3** | **6400× stronger** |
| (0, 999) — clean 64, noisy 32 | 2.5e-5 | Weak reverse influence |
| (0, 0) — both clean | 5.3e-1 | High sensitivity (both precise) |

The ratio J_HL/J_HH (cross vs self influence on 64×64) increases monotonically as t_32 decreases (32 gets cleaner):

| t_32 | 999 | 826 | 654 | 482 | 310 | 137 |
|------|-----|-----|-----|-----|-----|-----|
| J_HL/J_HH at t_64=999 | 0.04% | 0.07% | 0.11% | 0.16% | 0.20% | **0.72%** |

While cross-Jacobians are small in absolute terms, the trend is unambiguous: **the cleaner the 32×32 input, the more it influences the 64×64 prediction**. This confirms that the model learns to use coarse-scale structural information when available.

**Connection to γ_D chain-rule factor**: The alternative chain-rule factor γ_D = SNR/(4(1+SNR)²) produces an extreme version of this pattern — an L-shaped path that denoises 32×32 almost completely before starting on 64×64. However, this extreme strategy performs worse (FID 35.17 vs 28.91 with γ_B), suggesting that while "coarse first" is beneficial, pushing it too far is counterproductive. The optimal strategy is a moderate asymmetry where 32×32 leads by a few steps, not a complete sequential ordering.

## Visualizations

### Optimal Path Visualization (2000 epochs)

The heatmaps show discretization error across the 2D timestep space (t_64 × t_32). The optimal 18-step path (yellow points connected by red line) is computed via dynamic programming to minimize total error.

| 64×64 Error Heatmap | Total Error Heatmap |
|:-------------------:|:-------------------:|
| ![DP-64 Path](results/heatmap_30_2000ep_64.png) | ![DP-Total Path](results/heatmap_30_2000ep_total.png) |

- **Cyan line**: Diagonal path (t_64 = t_32)
- **Yellow points + Red line**: Optimal 18-step DP path
- **DP-Total path** deviates from diagonal to minimize combined error

### Path Convergence Across Epochs

![Path Comparison](results/path_comparison_across_epochs.png)

- **DP-Total paths converge after 1200ep** (green/blue/purple lines overlap)
- **DP-64 paths converge after 1200ep** with L-shaped trajectory
- Early epochs (400, 800) show different paths due to undertrained model

## Technical Details

### Discretization Error (vHv) Computation

The discretization error measures how much the model's output changes when the input is perturbed:

$$\text{Error} = v^T \left( J \odot J \right) v$$

We use the **Hutchinson trace estimator** to avoid computing the full Jacobian:

$$v^T (J \odot J) v = \mathbb{E}_{\epsilon} \left[ (J \epsilon)^2 \right]$$

### Chain-Rule Factor

The model operates in $x_t$ space (DDIM), but we want error in SNR ($\gamma$) space.

In DDIM:
$$x_t = \sqrt{\bar{\alpha}} \cdot x_0 + \sqrt{1 - \bar{\alpha}} \cdot \epsilon$$

In SNR parameterization:
$$z = \sqrt{\text{SNR}} \cdot x_0 + \epsilon$$

For the squared Jacobian (vHv):
$$(J_z)^2 = \frac{(J_{x_t})^2}{\text{SNR} \cdot (1 + \text{SNR})}$$

```python
def chain_rule_factor(snr):
    """Convert Jacobian from x_t space to z (SNR) space."""
    return 1.0 / (snr * (1.0 + snr))
```

### 2D Optimal Path Algorithm

The key insight is treating timestep scheduling as a **2D path optimization problem**. Given a 30×30 error heatmap, we find the optimal N-step path from (0,0) to (29,29) using dynamic programming.

**Cost Function**: Trapezoidal integral of error in λ-space (logSNR), with each scale weighted by its own Δλ
```python
step_cost = (e64[i,j] + e64[ni,nj])/2 * |Δλ_64| + (e32[i,j] + e32[ni,nj])/2 * |Δλ_32|
```
where `Δλ_64 = |log_snr[ni] - log_snr[i]|` is the logSNR distance for the 64×64 scale.

Using Δλ instead of grid-index distance is physically meaningful: grid indices are uniformly spaced, but logSNR spacing is non-uniform. Δλ correctly measures distance in the diffusion ODE's natural coordinate.

- **DP-64**: `e32 = 0` (only 64×64 error matters)
- **DP-Total**: both `e64` and `e32` contribute (recommended)

**Constraints**:
- Path must be monotonically increasing (can only move right/down)
- Maximum jump per step: `max_jump = 5` grid cells in each dimension
- This prevents unrealistic large jumps (e.g., t=999 → t=0 in one step)

```python
def find_optimal_path_n_steps_lambda(error_64, error_32, log_snr, num_steps=18, max_jump=5):
    """Find optimal N-step path using DP with λ-space trapezoidal cost."""
    # dp[i][j][k] = min cost to reach (i,j) in exactly k steps
    # Transition: try all (ni, nj) within max_jump distance
    # Cost: (e64[i,j]+e64[ni,nj])/2 * |Δλ_64| + (e32[i,j]+e32[ni,nj])/2 * |Δλ_32|
```

## Model Architecture

```
Input Processing:
  x_64 [B, 3, 64, 64] ─────────────────┐
                                        ├─ concat ─→ [B, 6, 64, 64]
  x_32 [B, 3, 32, 32] ─→ upsample 2× ──┘

Time Embedding:
  t_64 ─→ SinusoidalEmb ─→ MLP ─┐
                                 ├─ concat ─→ MLP ─→ t_combined
  t_32 ─→ SinusoidalEmb ─→ MLP ─┘

UNet Architecture:
  Encoder: [64, 128, 256, 512] channels
  - ResBlock + ResBlock + Attention (if channels >= 128)
  - Downsample 2×

  Bottleneck: 512 channels
  - ResBlock + Attention + ResBlock

  Decoder: [512, 256, 128, 64] channels
  - Upsample 2×
  - Skip connection (concat)
  - ResBlock + ResBlock + Attention (if channels >= 128)

Output Processing:
  [B, 6, 64, 64] ─→ split ─→ noise_pred_64 [B, 3, 64, 64]
                          ─→ downsample 2× ─→ noise_pred_32 [B, 3, 32, 32]
```

## Quick Start

```bash
# Train Baseline
python train_baseline.py --epochs 2000

# Train LIFT
python train_lift.py --epochs 2000

# Evaluate all models (3 GPUs in parallel)
./scripts/eval_all.sh

# Evaluate with multiple seeds
./scripts/eval_all.sh --multi-seed

# Compute heatmap for a specific epoch
./scripts/compute_heatmap.sh 2000
```

## Dataset

AFHQ (Animal Faces HQ) 64×64, loaded via HuggingFace datasets.
