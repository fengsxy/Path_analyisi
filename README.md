# LIFT Dual-Scale Diffusion Model

A dual-scale diffusion model that jointly denoises 64×64 and 32×32 images with independent timesteps for each scale.

## Latest Results: 100 Epoch Training Study

### Performance Comparison (100 epochs, ~58M parameters, 1000 images)

| Epoch | Baseline | LIFT Diagonal | LIFT DP-64 | LIFT DP-Total | Best Model |
|-------|----------|---------------|------------|---------------|------------|
| 20    | 99.11    | 136.96        | 149.13     | **120.70**    | Baseline   |
| 40    | **56.47** | 61.11        | 78.81      | 77.66         | **Baseline** |
| 60    | 70.18    | **64.46**     | 60.24      | 64.56         | LIFT DP-64 |
| 80    | **44.77** | 62.88        | **57.46**  | 72.69         | **Baseline** |
| 100   | **40.53** | 66.53        | 71.66      | 74.44         | **Baseline** |

**Key Findings:**
- **Baseline wins overall**: 40.53 FID at 100 epochs
- **LIFT best**: 57.46 FID at 80 epochs (DP-64 path)
- **Gap**: 16.93 FID points between best models
- **LIFT overfitting**: Performance degrades from 80→100 epochs (57.46→71.66)
- **LIFT needs more training**: Very poor at 20 epochs (~140 FID), improves significantly by 40 epochs
- **DP-64 vs DP-Total**: DP-64 path outperforms DP-Total from 60 epochs onward
- **Different paths peak at different epochs**:
  - Diagonal: Best at 40ep (61.11)
  - DP-64: Best at 80ep (57.46) ⭐
  - DP-Total: Best at 40ep (77.66)

![FID Progression](figure_plot/fid_progression_comparison.png)

### Generation Path Analysis

**DP Path Variation Across Training:**

Different training epochs produce different optimal DP-64 paths:
- **20ep**: γ₁ stays very low (0.01 → 0.037)
- **40ep**: γ₁ jumps to medium (0.01 → 51.79)
- **60ep**: γ₁ moderate (0.01 → 7.20)
- **80ep**: γ₁ higher (0.01 → 26.83)
- **100ep**: γ₁ jumps to max (0.01 → 100)

This shows the model learns different error distributions at different training stages.

**Path Comparison:**
- **Diagonal Path**: Simple, consistent across epochs, works well early (40ep)
- **DP-64 Path**: Optimizes 64×64 error, best for later training (80ep)
- **DP-Total Path**: Optimizes total error, competitive early but falls behind later

### Training Configuration
- **Dataset**: AFHQ64 (15,803 training images)
- **Epochs**: 100 (with checkpoints every 20 epochs)
- **Batch size**: 64
- **Learning rate**: 1e-4 with cosine annealing
- **Architecture**: ~58M parameters, hidden dims [64, 128, 256, 512]
- **Hardware**: 2× GPUs (parallel training)

### Generation Paths Tested
1. **Diagonal Path** (γ₁ = γ₀): Simple diagonal timestep schedule
2. **DP-64 Path**: Dynamic programming path optimizing 64×64 error only

---

## Quick Start

**Best Model**: Diagonal Training LIFT (FID = 94.79)

```bash
# 1. Setup environment
conda create -n diffusion-gpu python=3.10
conda activate diffusion-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pillow tqdm datasets pytorch-fid

# 2. Generate images with best model
python generate_for_fid.py \
    --checkpoint checkpoints/lift_exp2_diagonal.pth \
    --output_dir results/generated \
    --num_images 1000 \
    --mode diagonal \
    --device 0

# 3. Evaluate FID
python prepare_fid_real.py --output_dir results/fid_real --num_images 1000
python -m pytorch_fid results/fid_real results/generated
```

**Expected Result**: FID ≈ 94.79

## Key Results

### 30 Epoch Study: Training Strategy Comparison

| Model / Training Strategy | Description | FID ↓ |
|---------------------------|-------------|-------|
| **Baseline (Non-LIFT)** | Single-scale 64×64 model | **78.83** |
| **LIFT Diagonal Training** | Train with t_64 = t_32 (diagonal timesteps) | **94.79** ⭐ |
| LIFT DP 64×64 Only | Train with independent timesteps, optimize 64×64 error | 101.53 |
| LIFT DP Total | Train with independent timesteps, optimize total error | 109.20 |
| LIFT Independent Training | Train with independent timesteps (original) | 116.14 |

### 100 Epoch Study: Extended Training Analysis

**Full Results Table:**

| Epoch | Baseline FID ↓ | LIFT Diagonal FID ↓ | LIFT DP-64 FID ↓ | LIFT DP-Total FID ↓ | Winner |
|-------|----------------|---------------------|------------------|---------------------|--------|
| 20    | 99.11          | 136.96              | 149.13           | 120.70              | Baseline |
| 40    | **56.47**      | 61.11               | 78.81            | 77.66               | **Baseline** |
| 60    | 70.18          | 64.46               | **60.24**        | 64.56               | **LIFT DP-64** |
| 80    | **44.77**      | 62.88               | 57.46            | 72.69               | **Baseline** |
| 100   | **40.53**      | 66.53               | 71.66            | 74.44               | **Baseline** |

**Summary:**
- **Overall Best**: Baseline 100ep = 40.53 FID
- **LIFT Best**: DP-64 80ep = 57.46 FID
- **Performance Gap**: 16.93 FID points
- **LIFT Peak**: 80 epochs (then degrades)
- **Training Time**: ~40 minutes on 2× GPUs

**Path-Specific Best Performance:**
- **Diagonal**: 40ep (61.11 FID)
- **DP-64**: 80ep (57.46 FID) ⭐ Best LIFT result
- **DP-Total**: 40ep (77.66 FID)

**DP-64 vs DP-Total Comparison:**
- Early training (20-40ep): DP-Total performs better
- Mid training (60ep): DP-64 starts to win (60.24 vs 64.56)
- Late training (80-100ep): DP-64 maintains advantage

This suggests that optimizing 64×64 error directly becomes more effective as the model matures.

### Core Findings

1. **Training space is the bottleneck** - Diagonal training (94.79) significantly outperforms independent training (116.14)
   - Independent training: 1M combinations (1000×1000), each seen ~5 times in 30 epochs
   - Diagonal training: 1K combinations, each seen ~5000 times in 30 epochs
   - **1000× more efficient learning per timestep combination**

2. **Diagonal training closes the gap** - Only 16 FID points behind Baseline (vs 37 points for original LIFT)
   - Gap reduction: 37 → 16 points (57% improvement)

3. **Training-generation consistency is critical** - The training strategy determines the optimal generation strategy
   - Diagonal train + diagonal generation = 94.79 ✅
   - Diagonal train + DP path generation = 248.78 ❌ (catastrophic failure)

4. **32×32 branch is essential** - Experiments show 32×32 must be trained and used correctly
   - Without training loss_32: FID = 212.79
   - With broken 32×32 generation: FID = 300.68

## Experiments & Analysis

### Ablation Studies

We conducted systematic experiments to understand why LIFT underperforms Baseline:

#### Experiment 1: Multi-task Loss Conflict (H1)

**Hypothesis**: loss_64 and loss_32 gradients conflict, hurting optimization

**Method**: Train with `loss = loss_64` only (ignore loss_32)

**Result**: FID = 212.79 (worse than original 116.14)

**Conclusion**: ❌ H1 rejected. loss_32 is essential, not harmful. Without it, the 32×32 branch fails to learn, providing incorrect information during generation.

#### Experiment 2: Training Space Too Large (H2)

**Hypothesis**: Independent timesteps create 1M combinations (1000×1000), making training inefficient

**Method**: Train with diagonal timesteps `t_64 = t_32`

**Result**: FID = 94.79 (18% better than original 116.14)

**Conclusion**: ✅ H2 confirmed! Training space is the main bottleneck.

**Analysis**:
- Independent training: 1M combinations, each seen ~5 times in 30 epochs
- Diagonal training: 1K combinations, each seen ~5000 times in 30 epochs
- 1000× more efficient learning per timestep combination

#### Experiment 4: 32×32 Branch Contribution

**Hypothesis**: 32×32 branch doesn't contribute during generation

**Method**: Generate with random or fixed 32×32 noise

**Results**:
- Normal generation: FID = 116.14
- Random 32×32: FID = 300.68 (+159%)
- Fixed 32×32: FID = 291.77 (+151%)

**Conclusion**: ❌ H4 rejected. 32×32 is critical for generation quality.

### Root Cause Analysis

**Why LIFT underperforms Baseline:**
1. **Primary cause**: Training space explosion (1M vs 1K combinations)
2. **Secondary cause**: 30 epochs may be insufficient for such large space
3. **Not the cause**: Multi-task loss conflict or 32×32 branch design

**Solution**: Diagonal training reduces the gap from 37 FID points to 16 points.

### Training-Generation Consistency

**Critical Finding**: The training strategy determines the optimal generation strategy.

| Training | Best Generation Path | FID | Reason |
|----------|---------------------|-----|--------|
| Independent | DP path | 101.53 | Model saw diverse (t₆₄, t₃₂) combinations |
| Diagonal | Diagonal path | 94.79 | Model only saw (t, t) combinations |
| Diagonal | DP path | 248.78 ⚠️ | Train-generation mismatch |

**Key Insight**: Even though error heatmap shows DP path is theoretically optimal, using it with diagonal-trained model causes catastrophic failure (FID 95→249). The model must generate with the same timestep strategy it was trained on.

### Why DP 64×64 Path Works for Independent Training

1. **Focus on what matters**: Since we evaluate FID on 64×64 images, optimizing 64×64 error directly improves output quality
2. **32×32 as auxiliary**: Keeping 32×32 noisy (low γ₁) means the model relies less on the auxiliary scale
3. **Avoid distribution shift**: DP Total's early γ₁ jump creates unusual (noisy 64×64, clean 32×32) combinations rarely seen in training

## Model Architecture

### LIFTDualTimestepModel

- **Type**: LIFT Dual Timestep Model
- **Parameters**: 58.5M
- **Hidden dims**: [64, 128, 256, 512]
- **Input**: 64×64 RGB + 32×32 RGB (concatenated after upsampling)
- **Output**: Noise predictions for both scales

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

**Key Design Choices:**
1. **Dual Timestep Embedding**: Model receives both t_64 and t_32, combined via MLP
2. **Shared UNet**: Single UNet processes both scales jointly
3. **Bilinear Interpolation**: Used for upsampling 32×32 input and downsampling 32×32 output
4. **Attention at Higher Channels**: Self-attention only at 128+ channel layers for efficiency

## Usage

### Training

```bash
# Train baseline model
python train_baseline.py --epochs 30 --batch_size 64 --device 0

# Train LIFT with diagonal timesteps (best)
python tmp/train_exp2_diagonal.py --epochs 30 --batch_size 64 --device 0

# Train LIFT with independent timesteps (original)
python train_lift.py --epochs 30 --batch_size 64 --device 0
```

### Generate Images

```bash
# Best model: Diagonal training
python generate_for_fid.py \
    --checkpoint checkpoints/lift_exp2_diagonal.pth \
    --output_dir results/generated \
    --num_images 1000 \
    --mode diagonal \
    --device 0

# Original LIFT with DP path
python generate_for_fid.py \
    --checkpoint checkpoints/lift_full_random_final.pth \
    --output_dir results/generated_dp \
    --num_images 1000 \
    --mode dp_64 \
    --heatmap results/error_heatmap_chainrule.pth \
    --device 0
```

### Compute Error Heatmap

```bash
python compute_error_heatmap.py \
    --checkpoint checkpoints/lift_exp2_diagonal.pth \
    --output results/error_heatmap_diagonal.pth \
    --num_points 15 --device 0
```

### Plot Heatmap with Path

```bash
# Greedy path on scale0 error
python plot_heatmap.py --heatmap results/error_heatmap_chainrule.pth

# DP path on total error
python plot_heatmap.py --heatmap results/error_heatmap_chainrule.pth --dp
```

### FID Evaluation

```bash
# Prepare real images (if not already done)
python prepare_fid_real.py --output_dir results/fid_real --num_images 1000

# Generate and evaluate
python generate_for_fid.py \
    --checkpoint checkpoints/lift_exp2_diagonal.pth \
    --output_dir results/fid_test \
    --num_images 1000 \
    --mode diagonal \
    --device 0

python -m pytorch_fid results/fid_real results/fid_test
```

### MSE Tests

```bash
python tests/test.py \
    --checkpoint checkpoints/lift_full_random_final.pth \
    --device 0 --timestep 500 --num_samples 500
```

**MSE Test Results:**

| Test | 64×64 Input | 32×32 Input | MSE |
|------|-------------|-------------|-----|
| **Test 1** | Noisy (t=500) | Noisy (t=500) | 0.0178 |
| **Test 2a** | Noisy (t=500) | Clean (t=0) | 0.0746 |
| **Test 2b** | Clean (t=0) | Noisy (t=500) | 0.0703 |

**Interpretation:**
- Test 1 shows the model can denoise both scales effectively when both are noisy
- Tests 2a/2b show higher MSE because the model expects both scales to have similar noise levels during training
- The model was trained with random independent timesteps, so it handles the "both noisy" case best

## Figures

### Optimal Path Evolution Across Training

![Path Evolution](figure_plot/path_evolution_across_epochs.png)

**Top row**: Optimal paths on 64×64 Error heatmaps
**Bottom row**: Optimal paths on Total Error heatmaps

**Key Observations**:
- **64×64 Error paths vary significantly** across epochs (0.04 → 51.79 → 7.20 → 26.83 → 100)
- **Total Error paths are consistent**: All jump to γ₁=100 immediately after start
- **Different error metrics favor different strategies**: 64×64 error prefers keeping 32×32 noisy (low γ₁), while total error wants both scales clean

### MSE Test Visualization
![MSE Chart](figure_plot/mse_test_chart.png)

### Generated Samples (100 Epochs)

**Baseline (FID=40.53):**
![Baseline 100ep](figure_plot/generated_baseline_100ep.png)

**LIFT Models at 100 Epochs:**

| Diagonal (FID=66.53) | DP-64 (FID=71.66) | DP-Total (FID=74.44) |
|----------------------|-------------------|----------------------|
| ![Diagonal 100ep](figure_plot/generated_diagonal_100ep.png) | ![DP64 100ep](figure_plot/generated_dp64_100ep.png) | ![DP Total 100ep](figure_plot/generated_dp_total_100ep.png) |

**Note**: While LIFT's best performance was at 80 epochs (DP-64: 57.46 FID), these 100 epoch samples show the model after slight overfitting.

## File Structure

```
simple_diffusion_clean/
├── Core Model Files
│   ├── model.py              # LIFTDualTimestepModel definition
│   ├── baseline_model.py     # Baseline single-scale model
│   ├── scheduler.py          # DDIM scheduler
│   └── data.py               # AFHQ64 dataset loader
│
├── Training Scripts
│   ├── train_baseline.py     # Train baseline model
│   └── train_lift.py         # Train LIFT dual-scale model
│
├── Generation & Evaluation
│   ├── generate.py           # Interactive image generation
│   ├── generate_for_fid.py   # Batch generation for FID evaluation
│   ├── prepare_fid_real.py   # Prepare real images for FID
│   └── compute_error_heatmap.py  # Error heatmap computation
│
├── Shell Scripts
│   └── sh/                   # Training and evaluation scripts
│       ├── train100.sh       # Train 100 epochs with checkpoints
│       ├── eval_checkpoints.sh  # Evaluate all checkpoints
│       └── eval_10k.sh       # 10k image evaluation
│
├── Plotting & Visualization
│   └── figure_plot/          # Plotting scripts and generated figures
│       ├── plot_path_evolution.py
│       ├── plot_fid_comparison.py
│       └── *.png             # All figures used in README
│
├── Results & Outputs
│   ├── checkpoints/          # Trained model checkpoints
│   ├── results/              # 1k evaluation results
│   ├── results_10k/          # 10k evaluation results
│   └── logs/                 # Training logs
│
└── Tests
    └── tests/                # MSE evaluation and other tests
```

## Technical Details

### SNR ↔ Timestep Conversion

```
SNR = α_bar / (1 - α_bar)

Low SNR (γ=0.01)  →  High timestep (t=943)  →  High noise
High SNR (γ=100)  →  Low timestep (t=54)    →  Low noise
```

### Discretization Error (vHv) Computation

#### What is vHv?

The discretization error measures how much the model's output changes when the input is perturbed. For a denoising model $\hat{x}(z)$, we want to compute:

$$\text{Error} = v^T \left( J \odot J \right) v$$

where:
- $J = \frac{\partial \hat{x}}{\partial z}$ is the Jacobian matrix
- $\odot$ denotes element-wise (Hadamard) product
- $v$ is a weighting vector (uniform in our case)

#### Hutchinson Estimator

Computing the full Jacobian is expensive ($O(n^2)$ for $n$ pixels). Instead, we use the **Hutchinson trace estimator**:

$$v^T (J \odot J) v = \mathbb{E}_{\epsilon} \left[ (J \epsilon)^2 \right]$$

where $\epsilon$ is a random Rademacher vector ($\pm 1$ with equal probability).

**Algorithm:**
```python
def get_vHv(f, z, v, K=8):
    """
    Compute vHv using Hutchinson estimator.

    Args:
        f: Model function z -> x_hat
        z: Input (noisy image)
        v: Weighting vector
        K: Number of random samples
    """
    scale = sqrt(v)
    vhv = 0

    for _ in range(K):
        # Random Rademacher vector
        eps = random_choice([-1, +1], size=z.shape)
        u = eps * scale

        # Compute Jacobian-vector product using autodiff
        Ju = jvp(f, z, u)  # ∂f/∂z · u

        # Accumulate squared JVP
        vhv += (Ju ** 2).sum()

    return vhv / K
```

#### Chain-Rule Factor

The model operates in $x_t$ space (DDIM parameterization), but we want error in SNR ($\gamma$) space. The conversion requires a chain-rule factor:

$$\gamma(\text{SNR}) = \frac{1}{\text{SNR} \times (1 + \text{SNR})}$$

**Derivation:**

In DDIM, the noisy sample is:
$$x_t = \sqrt{\bar{\alpha}} \cdot x_0 + \sqrt{1 - \bar{\alpha}} \cdot \epsilon$$

In SNR parameterization:
$$z = \sqrt{\text{SNR}} \cdot x_0 + \epsilon$$

The relationship is:
$$x_t = \frac{z}{\sqrt{\text{SNR} \cdot (1 + \text{SNR})}}$$

So the Jacobian transforms as:
$$\frac{\partial x_t}{\partial z} = \frac{1}{\sqrt{\text{SNR} \cdot (1 + \text{SNR})}}$$

For the squared Jacobian (vHv):
$$(J_z)^2 = \frac{(J_{x_t})^2}{\text{SNR} \cdot (1 + \text{SNR})}$$

#### Implementation

```python
def chain_rule_factor(snr):
    """Convert Jacobian from x_t space to z (SNR) space."""
    return 1.0 / (snr * (1.0 + snr))

def compute_error(model, z_64, z_32, t_64, t_32, snr_64, snr_32):
    # Define functions for each scale
    def f_64(z): return model(z, z_32, t_64, t_32)[0]
    def f_32(z): return model(z_64, z, t_64, t_32)[1]

    # Compute vHv in x_t space
    vhv_64_xt = get_vHv(f_64, z_64, v=1/(64*64))
    vhv_32_xt = get_vHv(f_32, z_32, v=1/(32*32))

    # Apply chain-rule factor
    error_64 = vhv_64_xt * chain_rule_factor(snr_64)
    error_32 = vhv_32_xt * chain_rule_factor(snr_32)

    return error_64, error_32
```

**Why This Matters:**
1. Error decays as $\text{SNR}^{-2}$ at high SNR (matching theoretical expectations)
2. Error is properly normalized across different noise levels
3. The DP optimal path correctly balances error across the generation trajectory

#### DP Path to Generation Steps

1. **DP Path**: Maps γ₀ → optimal γ₁
   ```
   γ₀=0.01  → γ₁=0.01   (start)
   γ₀=0.02  → γ₁=100    (jump to max)
   ...
   γ₀=100   → γ₁=100    (end)
   ```

2. **Generation Schedule**: Interpolate to N steps
   ```python
   gamma0_schedule = logspace(0.01, 100, num_steps)
   gamma1_schedule = interp(gamma0_schedule, dp_path)
   ```

3. **Convert to Timesteps**:
   ```python
   timesteps_64 = [snr_to_timestep(g) for g in gamma0_schedule]
   timesteps_32 = [snr_to_timestep(g) for g in gamma1_schedule]
   ```

## Dataset

AFHQ (Animal Faces HQ) 64×64, loaded via HuggingFace datasets.

## Citation

If you use this code, please cite:
```
@misc{lift-dual-scale-diffusion,
  title={LIFT Dual-Scale Diffusion Model},
  year={2024}
}
```
