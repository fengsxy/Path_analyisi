# explore_test_6 Memory

## Model
- Pretrained SLOT CIFAR model: `/home/ylong030/slot/network-snapshot-052685.pkl`
- 55.9M params, EDMPrecondSlot + SongUNet (model_channels=128, channel_mult=[2,2,2], augment_dim=9, dropout=0.13)
- 3 scales: orig 32×32, 2x (16×16→32×32), 4x (8×8→32×32)
- 18 torus channels (each RGB scale → 6 channels: cos, sin)
- EDM framework: sigma_data=0.5, sigma_min=0, sigma_max=inf
- Forward: `net(x [B,18,32,32], sigma_vec [B,3], class_labels=None)`

## CRITICAL: Torus Normalization
Latent initialization MUST normalize to torus manifold:
```python
latents = torch.randn(batch_size, 18, 32, 32, ...)
cos_part, sin_part = latents.chunk(2, dim=1)
norm = torch.sqrt(cos_part ** 2 + sin_part ** 2 + 1e-8)
latents = torch.cat([cos_part / norm, sin_part / norm], dim=1)
```
Without this, FID degrades from ~70 to ~212. The Heun sampler does `latents * 1.4` internally — do NOT double-scale.

## Path Exploration Results (10k images, final)
| Path | FID (10k) | FID (1k) |
|------|-----------|----------|
| lead_5 | **34.21** | 61.70 |
| lead_6 | 35.04 | — |
| lead_4 | 35.89 | 63.21 |
| lead_7 | 36.23 | — |
| lead_3 | 38.17 | 65.04 |
| lead_8 | 39.80 | — |
| sync | 43.22 | 70.21 |
| lead_2 | — | 66.67 |
| lead_1 | — | 68.01 |
| ratio_1.2 | — | 69.71 |
| ratio_1.5 | — | 75.43 |
| dp_diag_2 | — | 79.46 |

## Key Findings
1. **Lead-follow is the best strategy**: 2x scale leads orig by N steps on the same rho schedule. lead_5 beats sync by 9 FID points (34.21 vs 43.22, 21% improvement).
2. **Sweet spot is lead_4 to lead_6**: FID is relatively flat in this range (34.2-35.9). Beyond lead_7, returns diminish.
3. **DP paths are degenerate**: Heatmap errors span 15 orders of magnitude and are nearly independent across scales → DP finds L-shaped paths (fully denoise one scale first). Even constrained DP (diagonal proximity) doesn't help.
4. **Speed ratio doesn't work**: ratio_1.2 barely matches sync; higher ratios are worse.
5. **"Low guides high" confirmed**: The best paths have the smaller scale (2x/4x) denoising ahead of the larger scale (orig).

## Heatmap
- 30×30 grid, sigma from 80.0 to 0.002
- Stored at `results/heatmap.pth`
- Keys: error_orig, error_2x, error_total, sigma_grid, path_total, path_orig, cost_total, cost_orig

## Files
- `generate_and_fid.py`: Main generation + FID (supports sync, coarse_to_fine, multi_offset, orig, total)
- `explore_paths.py`: Path exploration (lead-follow, speed ratio, constrained DP)
- `train_slot.py`: Training script (smoke-tested, ~3.1 it/s on single GPU)
- `results/cifar10_real_features.npz`: Cached real CIFAR-10 inception features (50k images)

## TODO
- Launch training for 2000 epochs
- Consider 50k image FID for final numbers
