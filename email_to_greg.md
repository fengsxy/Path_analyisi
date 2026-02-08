Subject: LIFT dual-scale update — path optimization results and some observations

Hi Greg,

I wanted to update you on the dual-scale diffusion experiments. The full results with visualizations are in the README: [GitHub link]

**Key findings:**

1. **Path optimization is critical.** Synchronized denoising (diagonal path, t_64 = t_32) gives FID 49-88. Optimizing the path based on total discretization error (AYS, in logSNR space) drops FID to 36-40 — an improvement of 8-33 points. This also explains why the slot framework gave poor results (FID ~20 vs baseline ~2): without path optimization, the synchronized denoising path is far from optimal in the 2D timestep space.

2. **With optimal paths, we nearly match baseline.** Best: Baseline EMA = 27.90, LIFT EMA (path optimized on total error) = 28.91 (both at 400ep, matched parameter count ~58M). The gap is only ~1 FID point.

3. **"Low guides high" emerges consistently.** The optimized paths consistently have 32×32 denoising slightly ahead of 64×64 (17/19 steps across all epochs). Cross-Jacobian analysis confirms this: the influence of 32→64 is 6400× stronger when 32 is clean vs when both are noisy. But in absolute terms, the cross-influence remains small (J_HL/J_HH peaks at 0.72%).

4. **Training stability.** Baseline peaks at 200ep (33.07) then degrades to 59.37 at 2000ep. LIFT (path optimized on total error) improves monotonically to its best (36.65) at 2000ep.

**Some observations I'd like your thoughts on:**

The optimized paths are only mildly asymmetric — 32×32 leads by a few steps, but it's far from an L-shaped path. I've been thinking about why, and I suspect it relates to the over-completeness of our representation: since 32×32 is just a downsampled 64×64, denoising it first doesn't provide information that 64×64 doesn't already contain. The cross-Jacobian seems to support this — the coupling exists but is weak.

This made me think about the contrast with VAR-style residual representations, where each scale encodes genuinely different information (low-frequency structure vs high-frequency details). In that setting, I'd expect the coarse-first strategy to have a much stronger effect, and the L-shaped path to be more natural.

It also connects to the anisotropic diffusion paper you mentioned — they work with orthogonal (DCT) bases rather than over-complete ones. I'm curious whether you think the over-complete case has a fundamentally different character from the orthogonal case, or if there's an angle I'm missing where the redundancy could be beneficial.

In any case, the path optimization framework (AYS discretization error in 2D) seems to work well and should transfer to other multi-scale architectures.

Would love to hear your thoughts.

Best,
Longxuan
