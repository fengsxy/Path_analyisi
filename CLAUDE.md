# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LIFT (Learned Independent Frequency Timesteps) Dual-Scale Diffusion Model - a research project exploring dual-scale diffusion models that jointly denoise 64×64 and 32×32 images with independent timesteps for each scale.

**Dataset**: AFHQ (Animal Faces HQ) 64×64, loaded via HuggingFace `huggan/AFHQv2`.

## Common Commands

### Training
```bash
# Train both Baseline and LIFT models in parallel (requires 2 GPUs)
./scripts/train.sh

# Train only baseline (single-scale 64×64)
python train_baseline.py --epochs 30 --batch_size 64 --device 0

# Train LIFT dual-timestep model
python train_lift.py --epochs 30 --batch_size 64 --device 0
```

### Evaluation (Main Script)
```bash
# Full evaluation pipeline for all epochs
./scripts/eval_lift.sh

# Evaluate specific epochs
./scripts/eval_lift.sh 20 40 60 80 100
./scripts/eval_lift.sh 200 400 600 800 1000

# Output:
#   - results/heatmap_XXXep.png (100×100 heatmap with DP paths)
#   - results/fid_results.csv (FID scores and DP path errors)
```

### Individual Commands
```bash
# Compute 100×100 error heatmap with DP paths (18 steps)
python compute_heatmap_timestep.py \
    --checkpoint checkpoints/lift_dual_timestep_100ep.pth \
    --output results/heatmap_100ep.png \
    --num_steps 18 --device 0

# Generate with DP-64 path
python generate_with_dp_path.py \
    --checkpoint checkpoints/lift_dual_timestep_100ep.pth \
    --heatmap results/heatmap_100ep.pth \
    --output_dir results/fid_dp64_100ep \
    --path_type dp_64 --num_steps 18

# Compute FID
python -m pytorch_fid results/fid_real results/fid_dp64_100ep
```

## Architecture

### Two Model Types

1. **BaselineModel** (`baseline_model.py`): Single-scale 64×64 diffusion model
   - Input: `[B, 3, 64, 64]` noisy image + timestep
   - Output: `[B, 3, 64, 64]` noise prediction

2. **LIFTDualTimestepModel** (`model.py`): Dual-scale with independent timesteps
   - Input: `x_64 [B, 3, 64, 64]`, `x_32 [B, 3, 32, 32]`, `t_64`, `t_32`
   - Output: `noise_pred_64 [B, 3, 64, 64]`, `noise_pred_32 [B, 3, 32, 32]`
   - Key: Accepts TWO timesteps, enabling any (t_64, t_32) pair

### Generation Modes
- `diagonal`: Same timestep for both scales (t_64 = t_32)
- `dp_64`: DP optimal path minimizing 64×64 error
- `dp_total`: DP optimal path minimizing total error

## Two-Step Path Optimization (Align Your Path)

Based on "Align Your Steps" (AYS) paper. The key insight: **error大的区域需要更密集的采样**.

### Core Problem: 2D Timestep Scheduling

Unlike single-scale diffusion (1D problem), LIFT has two timesteps (t_64, t_32), making it a **2D path optimization problem**:
- Start: (t_64=999, t_32=999) - both scales at high noise
- End: (t_64=0, t_32=0) - both scales at clean image
- Goal: Find optimal path through 2D timestep space

### Error Measurement

We compute a 30×30 error heatmap using Hutchinson estimator:
```python
vHv = v^T (J ⊙ J) v  # Jacobian element-wise squared
```

**Chain-rule factor** converts from x_t space to SNR space:
```python
γ(SNR) = 1 / (SNR * (1 + SNR))
error_z = error_xt * γ(SNR)
```

This ensures error decays as SNR⁻² at high SNR (theoretical expectation from stochastic localization).

### Key Insight: Cost = Error × Step Size

Simply minimizing `Σ error[i]` leads to unrealistic paths (e.g., jumping from t=999 to t=0 in one step).

**The fix**: Cost should be the **integral of error along the path segment**:
```python
step_cost = (error_start + error_end) / 2 × step_size  # Trapezoidal rule
```

This penalizes large jumps through high-error regions, matching the continuous integral in AYS.

### DP Algorithm with Constraints

```python
def find_optimal_path_n_steps(error_matrix, num_steps, max_jump=5):
    """
    Find optimal N-step path from (0,0) to (29,29).

    Constraints:
    - Exactly num_steps steps (e.g., 18)
    - Each step moves at most max_jump cells in each direction
    - Must reach endpoint (guarantees t_64=0, t_32=0)

    Cost function:
    - Trapezoidal integral: (error_start + error_end) / 2 × step_size
    """
    # dp[i][j][k] = min cost to reach (i,j) in k steps
    # Transition: try all (ni, nj) within max_jump distance
```

### Why max_jump Constraint?

Without it, DP exploits the fact that error_64 only depends on t_64 (vertical stripes in heatmap):
- It would jump t_64 from 999→0 in one step
- Then slowly decrease t_32

With `max_jump=5` (~170 timesteps per step), the path is forced to be more gradual and realistic for DDIM.

### Two Path Types

1. **DP-64**: Optimize using error_64 only (64×64 scale error)
   - Path tends to prioritize t_64 reduction

2. **DP-Total**: Optimize using error_64 + error_32 (total error)
   - Path is more balanced, closer to diagonal
   - Generally recommended

### Implementation Files

```bash
# Step 1: Compute 30×30 error heatmap
./scripts/compute_heatmap.sh 800

# Step 2: Plot heatmap with optimal paths
./scripts/plot_heatmap.sh 800

# Step 3: Generate images and compute FID
./scripts/eval_lift_dp.sh 800
```

### Output

- `results/heatmap_30_XXXep.pth`: Error map + computed paths
- `results/heatmap_30_XXXep.png`: Visualization (red line = path, yellow dots = sampling points)
- `results/fid_lift_dp_results.csv`: FID scores for DP-64 and DP-Total

## Key Concepts

### SNR (Signal-to-Noise Ratio) and Timesteps
- SNR = α_bar / (1 - α_bar)
- High SNR = low noise = low timestep (t≈0)
- Low SNR = high noise = high timestep (t≈999)
- Heatmap index 0 → t=999 (high noise), index 99 → t=0 (low noise)

### Chain-Rule Factor
When computing discretization error in SNR space:
```python
def chain_rule_factor(snr):
    return 1.0 / (snr * (1.0 + snr))
```

## File Organization

- `model.py`: LIFT models (LIFTBaselineModel, LIFTDualTimestepModel)
- `baseline_model.py`: Single-scale baseline model
- `scheduler.py`: DDIM scheduler implementation
- `data.py`: Dataset classes (AFHQ64Dataset)
- `train_*.py`: Training scripts
- `compute_heatmap_30.py`: 30×30 error heatmap + N-step DP path optimization
- `eval_fid_batch.py`: Batch FID evaluation (in-memory, no disk I/O)
- `eval_lift_dp.py`: Evaluate LIFT with DP paths
- `generate_with_path_30.py`: Generate images using DP optimal path
- `generate_for_fid.py`: Generate images (diagonal mode)
- `scripts/compute_heatmap.sh`: Step 1 - Compute error heatmap
- `scripts/plot_heatmap.sh`: Step 2 - Plot heatmap with paths
- `scripts/eval_lift_dp.sh`: Step 3 - Generate images + FID
- `scripts/eval_baseline.sh`: Evaluate baseline model
- `scripts/eval_lift.sh`: Evaluate LIFT diagonal mode
- `scripts/eval_all.sh`: Parallel evaluation on 3 GPUs (non-EMA)
- `scripts/eval_all_ema.sh`: Parallel EMA evaluation on 3 GPUs
- `aggregate_results.py`: Aggregate multi-seed FID results (mean±std)
- `explore_test/`: Experimental single-output architectures (single_t, no_t)
- `explore_test/eval_fid_batch.py`: Diagonal FID evaluation for explore_test models
- `explore_test/eval_dp.py`: Heatmap + DP path FID evaluation for explore_test models
- `explore_test/eval_all.sh`: Parallel launcher (GPU 1: single_t, GPU 3: no_t)
- `explore_test_2/`: EMA training scripts

## Experiment Settings

- **Default generation steps**: 18
- **Heatmap resolution**: 30×30
- **Max jump per step**: 5 (≈170 timesteps)
- **Checkpoints**: 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000 epochs
- **FID evaluation**: 15803 images (full AFHQ training set for stable FID)

## EMA Evaluation

Both `eval_fid_batch.py` and `eval_lift_dp.py` support `--ema` flag to load EMA weights from checkpoints.

```bash
# Evaluate all EMA models in parallel (3 GPUs)
./scripts/eval_all_ema.sh              # Single seed
./scripts/eval_all_ema.sh --multi-seed # 5 seeds with aggregation
./scripts/eval_all_ema.sh --force      # Force re-evaluation

# Individual EMA evaluation
python eval_fid_batch.py --model_type baseline --ema --epochs 200 400 --device 0
python eval_fid_batch.py --model_type lift --ema --epochs 200 400 --device 1
python eval_lift_dp.py --ema --epochs 200 400 --device 2
```

EMA checkpoint naming: `baseline_ema_{epoch}ep.pth`, `lift_ema_{epoch}ep.pth`

## Development Guidelines

### Testing Requirement
Always test code after writing it. For evaluation scripts, run a quick smoke test with reduced parameters before full runs:
```bash
# Quick test: 1000 images, single epoch, force re-evaluation
source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffusion-gpu
python eval_fid_batch.py --model_type baseline --ema --epochs 200 --device 1 --num_images 1000 --force --output_dir /tmp/eval_test
python eval_lift_dp.py --ema --epochs 200 --device 1 --num_images 1000 --force --output_dir /tmp/eval_test
```

### Long-Running GPU Tasks
Use `screen` for long-running evaluation jobs:
```bash
screen -S eval-session
# Run commands inside screen, then detach with Ctrl+A, D
# Reattach: screen -r eval-session
```

Check GPU availability before running:
```bash
nvidia-smi --query-gpu=index,memory.free --format=csv
```

## Environment

- Conda environment: `diffusion-gpu`
- Python path: `/home/ylong030/miniconda3/envs/diffusion-gpu/bin/python`
- Default hidden dims: `[64, 128, 256, 512]` — Baseline: 58.3M params, LIFT: 58.5M params
- GPUs: 3 available (0, 1, 2); check `nvidia-smi` for free memory before use

## explore_test: Single-Output Experimental Models

### Two Architectures

1. **SingleTimestepModel** (`explore_test/model_single_t.py`): Receives t_64 only
   - Input: `x_64 [B, 3, 64, 64]`, `x_32_noisy [B, 3, 32, 32]`, `t_64 [B]`
   - Output: `noise_pred_64 [B, 3, 64, 64]` only
   - 32×32 provides structural info but model doesn't know its noise level

2. **NoTimestepModel** (`explore_test/model_no_t.py`): Blind denoiser, no timestep
   - Input: `x_64 [B, 3, 64, 64]`, `x_32_noisy [B, 3, 32, 32]`
   - Output: `noise_pred_64 [B, 3, 64, 64]` only
   - No time embedding at all

### Key Design: x_32 from Downsampled x_64

Unlike the main LIFT model where x_32 is independently denoised, explore_test models construct x_32 from x_64:

- **Diagonal mode**: `x_32 = F.interpolate(x_64, 32)` at each step
- **DP mode**: `x_32 = downsample(pred_x0_64) + noise at t_32_next`
  1. Get `noise_pred_64` → compute `pred_x0_64` via DDIM formula
  2. `pred_x0_32 = F.interpolate(pred_x0_64, 32)`
  3. Re-noise: `x_32 = sqrt(α_t32) * pred_x0_32 + sqrt(1-α_t32) * ε`

### Evaluation

```bash
cd explore_test

# Diagonal FID
python eval_fid_batch.py --model_type single_t --epochs 200 400 --device 1
python eval_fid_batch.py --model_type no_t --epochs 200 400 --device 3

# DP path FID (computes heatmap + optimal path + generates + FID)
python eval_dp.py --model_type single_t --epochs 200 400 --device 1
python eval_dp.py --model_type no_t --epochs 200 400 --device 3

# Full parallel evaluation (all epochs, both models)
./eval_all.sh                    # All epochs
./eval_all.sh 200 400 600        # Specific epochs
./eval_all.sh --force            # Force re-evaluation
```

### Differences from Main Eval Scripts

- Only one DP path type (error_64 only — no error_32 since models don't predict 32×32 noise)
- Real features cached at parent `results/real_features.npz` (shared)
- Checkpoints: `explore_test/checkpoints/{single_t,no_t}_{epoch}ep.pth`
- Results: `explore_test/results/`

## Patterns for Writing New Eval Scripts

When adapting eval scripts for new model architectures:

1. **Reuse shared infrastructure**: Import from parent (`compute_heatmap_30.py`, `scheduler.py`, `data.py`) via `sys.path.insert`
2. **Share real features cache**: Point to `results/real_features.npz` in parent dir to avoid recomputation
3. **Adapt generation functions**: The model-specific part is only the forward call; DDIM step, FID computation, grid saving are all reusable
4. **For heatmap computation**: Only change the closure `f_64(z_in)` to match the model's forward signature
5. **Smoke test first**: Always test with `--num_images 1000 --force --output_dir /tmp/test` before full runs
6. **ddim_step returning pred_x0**: When DP generation needs to construct auxiliary inputs (like x_32) from predicted clean images, modify `ddim_step` to also return `pred_original_sample`
