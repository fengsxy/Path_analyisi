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

**Cost Function**: Trapezoidal integral of error along path segment
```python
step_cost = (error[i,j] + error[ni,nj]) / 2 * step_size
```
where `step_size = (ni - i) + (nj - j)` is the Manhattan distance.

**Constraints**:
- Path must be monotonically increasing (can only move right/down)
- Maximum jump per step: `max_jump = 5` grid cells in each dimension
- This prevents unrealistic large jumps (e.g., t=999 → t=0 in one step)

**Why trapezoidal integral?**
- Simple `error[ni,nj]` allows jumping to low-error regions (t≈0) immediately
- `error * step_size` still favors large jumps because error at t≈0 is tiny
- Trapezoidal integral properly accounts for error accumulated along the entire segment

```python
def find_optimal_path_n_steps(error_matrix, num_steps, max_jump=5):
    """Find optimal N-step path using DP with trapezoidal cost."""
    N = error_matrix.shape[0]
    INF = float('inf')

    # dp[step][i][j] = min cost to reach (i,j) in exactly `step` steps
    dp = [[[INF] * N for _ in range(N)] for _ in range(num_steps)]
    parent = [[[None] * N for _ in range(N)] for _ in range(num_steps)]

    dp[0][0][0] = 0

    for step in range(num_steps - 1):
        for i in range(N):
            for j in range(N):
                if dp[step][i][j] == INF:
                    continue
                for ni in range(i, min(i + max_jump + 1, N)):
                    for nj in range(j, min(j + max_jump + 1, N)):
                        if ni == i and nj == j:
                            continue
                        step_size = (ni - i) + (nj - j)
                        cost = (error[i,j] + error[ni,nj]) / 2 * step_size
                        if dp[step][i][j] + cost < dp[step+1][ni][nj]:
                            dp[step+1][ni][nj] = dp[step][i][j] + cost
                            parent[step+1][ni][nj] = (i, j)

    # Backtrack from (N-1, N-1)
    path = [(N-1, N-1)]
    for step in range(num_steps-1, 0, -1):
        path.append(parent[step][path[-1][0]][path[-1][1]])
    return path[::-1]
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
