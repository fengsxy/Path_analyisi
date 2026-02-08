#!/usr/bin/env python3
"""
Greedy Adaptive Path Sampler for LIFT dual-scale diffusion.

Algorithm 1: At each step, estimate per-scale Jacobian and MSE,
compute optimal step direction in SNR space, giving each sample
its own adaptive path.
"""

import torch
import torch.nn.functional as F
import numpy as np


def snr_to_timestep(target_snr, all_snr):
    """Convert target SNR values to nearest discrete timesteps.

    Args:
        target_snr: [B] tensor of target SNR values
        all_snr: [1000] tensor, all_snr[t] = alphas_cumprod[t] / (1 - alphas_cumprod[t])
                 Decreasing: t=0 has highest SNR, t=999 has lowest.

    Returns:
        [B] integer tensor of timesteps, clamped to [0, 999]
    """
    # Flip to ascending for searchsorted
    snr_ascending = all_snr.flip(0)  # [1000], ascending
    # searchsorted finds insertion point in ascending array
    idx_asc = torch.searchsorted(snr_ascending.contiguous(), target_snr.contiguous())
    # Flip index back: t = 999 - idx_asc
    t = 999 - idx_asc
    return t.clamp(0, 999).long()


def estimate_denoiser_output(model, scheduler, x_64, x_32, t_64, t_32):
    """Run model forward and convert eps prediction to x_hat (denoised estimate).

    Args:
        model: LIFTDualTimestepModel
        scheduler: DDIMScheduler (for alphas_cumprod)
        x_64: [B, 3, 64, 64] noisy images
        x_32: [B, 3, 32, 32] noisy images
        t_64: [B] integer timesteps for 64-scale
        t_32: [B] integer timesteps for 32-scale

    Returns:
        x_hat_64, x_hat_32, eps_pred_64, eps_pred_32
    """
    eps_pred_64, eps_pred_32 = model(x_64, x_32, t_64, t_32)

    acp = scheduler.alphas_cumprod.to(x_64.device)
    alpha_bar_64 = acp[t_64].view(-1, 1, 1, 1)  # [B,1,1,1]
    alpha_bar_32 = acp[t_32].view(-1, 1, 1, 1)

    # x_hat = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
    x_hat_64 = (x_64 - (1 - alpha_bar_64).sqrt() * eps_pred_64) / alpha_bar_64.sqrt()
    x_hat_32 = (x_32 - (1 - alpha_bar_32).sqrt() * eps_pred_32) / alpha_bar_32.sqrt()

    return x_hat_64, x_hat_32, eps_pred_64, eps_pred_32


def estimate_diagonal_jacobian_sq(model, scheduler, x_64, x_32, t_64, t_32, eps=1e-3, K=1):
    """Estimate squared diagonal Jacobian norm per scale via finite differences.

    For each scale, perturb input with Rademacher vector, compute finite-difference
    approximation of Jacobian-vector product, then estimate ||diag(J)||^2.

    Args:
        model, scheduler, x_64, x_32, t_64, t_32: as in estimate_denoiser_output
        eps: perturbation magnitude for finite differences
        K: number of random vectors to average over

    Returns:
        J_sq_64: [B] estimated ||diag(J_64)||^2 (normalized by spatial dims)
        J_sq_32: [B] estimated ||diag(J_32)||^2 (normalized by spatial dims)
    """
    B = x_64.shape[0]
    device = x_64.device
    d_64 = 3 * 64 * 64
    d_32 = 3 * 32 * 32

    J_sq_64_acc = torch.zeros(B, device=device)
    J_sq_32_acc = torch.zeros(B, device=device)

    # Get baseline prediction
    x_hat_64_0, x_hat_32_0, _, _ = estimate_denoiser_output(
        model, scheduler, x_64, x_32, t_64, t_32
    )

    for _ in range(K):
        # --- Perturb 64-scale ---
        w_64 = torch.randint(0, 2, x_64.shape, device=device).float() * 2 - 1  # Rademacher
        x_64_pert = x_64 + eps * w_64
        x_hat_64_p, _, _, _ = estimate_denoiser_output(
            model, scheduler, x_64_pert, x_32, t_64, t_32
        )
        # Jw ≈ (x_hat_pert - x_hat_0) / eps
        Jw_64 = (x_hat_64_p - x_hat_64_0) / eps
        # ||diag(J)||^2 ≈ mean over spatial of (w * Jw)^2
        # Since w is ±1, w*Jw = diag(J) element-wise (in expectation)
        elem_sq = (w_64 * Jw_64) ** 2  # [B, 3, 64, 64]
        J_sq_64_acc += elem_sq.view(B, -1).mean(dim=1)  # [B]

        # --- Perturb 32-scale ---
        w_32 = torch.randint(0, 2, x_32.shape, device=device).float() * 2 - 1
        x_32_pert = x_32 + eps * w_32
        _, x_hat_32_p, _, _ = estimate_denoiser_output(
            model, scheduler, x_64, x_32_pert, t_64, t_32
        )
        Jw_32 = (x_hat_32_p - x_hat_32_0) / eps
        elem_sq = (w_32 * Jw_32) ** 2
        J_sq_32_acc += elem_sq.view(B, -1).mean(dim=1)

    J_sq_64 = J_sq_64_acc / K
    J_sq_32 = J_sq_32_acc / K

    return J_sq_64, J_sq_32


def ddim_step_per_sample(scheduler, eps_pred, t_curr, t_next, x_curr):
    """DDIM step with per-sample integer timesteps.

    Args:
        scheduler: DDIMScheduler
        eps_pred: [B, C, H, W] noise prediction
        t_curr: [B] current timesteps (integers)
        t_next: [B] next timesteps (integers), 0 means final step
        x_curr: [B, C, H, W] current noisy sample

    Returns:
        x_next: [B, C, H, W] denoised sample at t_next
    """
    device = x_curr.device
    acp = scheduler.alphas_cumprod.to(device)

    # Per-sample alpha_bar values
    alpha_bar_t = acp[t_curr].view(-1, 1, 1, 1)  # [B,1,1,1]

    # For t_next, handle t_next=0 → use final_alpha_cumprod
    # We index acp[t_next] but clamp to valid range, then fix t_next=0 case
    t_next_clamped = t_next.clamp(min=0)
    alpha_bar_t_next = acp[t_next_clamped].view(-1, 1, 1, 1)
    # Where t_next == 0, use final_alpha_cumprod (which is 1.0 for set_alpha_to_one=True)
    # Actually t=0 is a valid index in alphas_cumprod, so acp[0] is correct.
    # The "final" case is when prev_timestep < 0, but we clamp to 0.

    beta_prod_t = 1 - alpha_bar_t

    # Predict x0
    pred_x0 = (x_curr - beta_prod_t.sqrt() * eps_pred) / alpha_bar_t.sqrt()

    if scheduler.clip_sample:
        pred_x0 = torch.clamp(pred_x0, -1, 1)

    # Re-derive eps from clipped x0
    pred_eps = (x_curr - alpha_bar_t.sqrt() * pred_x0) / beta_prod_t.sqrt()

    # DDIM deterministic step (eta=0)
    pred_dir = (1 - alpha_bar_t_next).sqrt() * pred_eps
    x_next = alpha_bar_t_next.sqrt() * pred_x0 + pred_dir

    return x_next


@torch.no_grad()
def generate_batch_greedy(model, scheduler, batch_size, num_steps, device,
                          bps=None, eps_jac=1e-3, K_jac=1, eps_reg=1e-8,
                          verbose=False):
    """Generate images using greedy adaptive path sampling.

    At each step:
    1. Forward pass → eps_pred, x_hat, compute per-scale MSE (E_k)
    2. Estimate diagonal Jacobian squared norm (J²_k) via finite differences
    3. Compute optimal step: Δsnr_k = 1 / (J²_k + eps_reg)
    4. Scale by BPS budget: snr_new = snr + λ * Δsnr
    5. Snap to nearest timestep, ensure progress

    Args:
        model: LIFTDualTimestepModel
        scheduler: DDIMScheduler
        batch_size: number of images to generate
        num_steps: number of denoising steps
        device: torch device
        bps: bits-per-step budget (None = auto-compute)
        eps_jac: perturbation for Jacobian estimation
        K_jac: number of random vectors for Jacobian estimation
        eps_reg: regularization for Δsnr computation
        verbose: print step-by-step info

    Returns:
        images: [B, 3, 64, 64] in [0, 1]
        paths_64: [B, num_steps+1] integer timesteps
        paths_32: [B, num_steps+1] integer timesteps
    """
    acp = scheduler.alphas_cumprod.to(device)
    all_snr = acp / (1 - acp)  # [1000], decreasing (t=0 highest)
    all_log_snr = torch.log(all_snr)

    # Total log-SNR range
    log_snr_max = all_log_snr[0].item()    # t=0, clean
    log_snr_min = all_log_snr[999].item()  # t=999, noisy
    total_range = log_snr_max - log_snr_min  # positive

    # Base step size per scale: uniform Δλ that would traverse the range in num_steps-1 steps
    # (last step is forced to t=0)
    base_delta = total_range / (num_steps - 1)

    if bps is not None:
        base_delta = bps  # override with user-specified budget

    if verbose:
        print(f"  Base delta_lambda per step: {base_delta:.4f} (total range: {total_range:.4f})")

    # Initialize
    x_64 = torch.randn(batch_size, 3, 64, 64, device=device)
    x_32 = torch.randn(batch_size, 3, 32, 32, device=device)
    t_64 = torch.full((batch_size,), 999, device=device, dtype=torch.long)
    t_32 = torch.full((batch_size,), 999, device=device, dtype=torch.long)

    # Record paths
    paths_64 = [t_64.cpu().clone()]
    paths_32 = [t_32.cpu().clone()]

    for step in range(num_steps):
        is_last = (step == num_steps - 1)

        if is_last:
            # Force to t=0
            t_next_64 = torch.zeros_like(t_64)
            t_next_32 = torch.zeros_like(t_32)
        else:
            # --- Greedy step computation in log-SNR space ---
            # 1. Forward pass → x_hat, eps_pred
            x_hat_64, x_hat_32, eps_pred_64, eps_pred_32 = estimate_denoiser_output(
                model, scheduler, x_64, x_32, t_64, t_32
            )

            # 2. Per-sample error proxy: E_k
            E_64 = eps_pred_64.view(batch_size, -1).pow(2).mean(dim=1)  # [B]
            E_32 = eps_pred_32.view(batch_size, -1).pow(2).mean(dim=1)  # [B]

            # 3. Estimate Jacobian
            J_sq_64, J_sq_32 = estimate_diagonal_jacobian_sq(
                model, scheduler, x_64, x_32, t_64, t_32,
                eps=eps_jac, K=K_jac
            )

            # 4. Error density in λ-space: ρ_k = E_k * J²_k
            rho_64 = E_64 * J_sq_64 + eps_reg  # [B]
            rho_32 = E_32 * J_sq_32 + eps_reg  # [B]

            # 5. Each scale gets base_delta as its Δλ, then we redistribute
            # between scales based on error density ratio.
            # Scale with higher ρ gets smaller Δλ (needs denser sampling).
            # Scale with lower ρ gets larger Δλ (can afford bigger steps).
            #
            # Redistribution: total budget = 2 * base_delta
            # Δλ_k = 2 * base_delta * (1/sqrt(ρ_k)) / (1/sqrt(ρ_64) + 1/sqrt(ρ_32))
            inv_sqrt_rho_64 = 1.0 / rho_64.sqrt()
            inv_sqrt_rho_32 = 1.0 / rho_32.sqrt()
            total_inv = inv_sqrt_rho_64 + inv_sqrt_rho_32 + eps_reg

            delta_lambda_64 = 2.0 * base_delta * inv_sqrt_rho_64 / total_inv  # [B]
            delta_lambda_32 = 2.0 * base_delta * inv_sqrt_rho_32 / total_inv  # [B]

            # Clamp: each scale's Δλ should not exceed remaining distance
            remaining_64 = log_snr_max - all_log_snr[t_64]  # [B]
            remaining_32 = log_snr_max - all_log_snr[t_32]  # [B]
            delta_lambda_64 = torch.min(delta_lambda_64, remaining_64)
            delta_lambda_32 = torch.min(delta_lambda_32, remaining_32)

            # 6. New log-SNR (increasing = denoising)
            log_snr_64 = all_log_snr[t_64]  # [B]
            log_snr_32 = all_log_snr[t_32]  # [B]

            new_log_snr_64 = log_snr_64 + delta_lambda_64
            new_log_snr_32 = log_snr_32 + delta_lambda_32

            # 7. Convert back to SNR, then snap to nearest timestep
            new_snr_64 = new_log_snr_64.exp()
            new_snr_32 = new_log_snr_32.exp()

            t_next_64 = snr_to_timestep(new_snr_64, all_snr)
            t_next_32 = snr_to_timestep(new_snr_32, all_snr)

            # 8. Ensure progress: t_next < t_curr (at least 1 step forward)
            t_next_64 = torch.min(t_next_64, t_64 - 1).clamp(min=0)
            t_next_32 = torch.min(t_next_32, t_32 - 1).clamp(min=0)

        # DDIM step
        if not is_last:
            # We already have eps_pred from the forward pass above
            x_64 = ddim_step_per_sample(scheduler, eps_pred_64, t_64, t_next_64, x_64)
            x_32 = ddim_step_per_sample(scheduler, eps_pred_32, t_32, t_next_32, x_32)
        else:
            # Last step: need a fresh forward pass
            eps_64, eps_32 = model(x_64, x_32, t_64, t_32)
            x_64 = ddim_step_per_sample(scheduler, eps_64, t_64, t_next_64, x_64)
            x_32 = ddim_step_per_sample(scheduler, eps_32, t_32, t_next_32, x_32)

        t_64 = t_next_64
        t_32 = t_next_32
        paths_64.append(t_64.cpu().clone())
        paths_32.append(t_32.cpu().clone())

        if verbose and step % 3 == 0:
            print(f"  Step {step}: t_64={t_64[0].item()}, t_32={t_32[0].item()}")

    # Stack paths: [B, num_steps+1]
    paths_64 = torch.stack(paths_64, dim=1)
    paths_32 = torch.stack(paths_32, dim=1)

    # Normalize to [0, 1]
    images = (x_64 + 1) * 0.5

    return images, paths_64, paths_32
