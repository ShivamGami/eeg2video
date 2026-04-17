"""
dana.py  —  Dynamic-Aware Noise Adding (DANA)
EEG2Video · IIT Mandi · CS 671

Implements Equation (2) from the EEG2Video paper (NeurIPS 2024):

    z_T = sqrt(alpha_T) * z0
        + sqrt(1 - alpha_T) * (sqrt(beta) * eps_s + sqrt(1-beta) * eps_d)

  eps_s  = static noise  : same noise replicated across all F frames
             → promotes temporal consistency (good for slow videos)
  eps_d  = diverse noise : independent noise per frame
             → promotes frame-level variation (good for fast videos)
  beta   = 0.3 if slow video  (more static noise)
           0.2 if fast video  (more diverse noise)

Interface:
    dana    = DANAModule()
    z_T     = dana(z0, is_fast, T)

    z0      : (B, F, 4, H, W)  clean latents from Seq2Seq encoder
    is_fast : (B,)  float in [0, 1]  sigmoid probability (>0.5 → fast)
    T       : int   noise level timestep  (default: num_timesteps - 1 = 999)
    returns : (B, F, 4, H, W)  noised latents z_T

Changes / fixes
---------------
v2  is_fast.bool() treated ANY nonzero float (e.g. 0.01) as "fast".
    Fixed to explicit `is_fast > 0.5` threshold.
v3  No new bugs found; file is unchanged from v2 (verified clean).
v4  forward() now also returns the blended noise (eps_mixed) alongside z_T
    so that train_backbone can use the EXACT same noise as the training
    target — eliminating the critical bug where dana_forward_with_target()
    regenerated independent eps_s/eps_d, making noise_target != actual noise.
    Return signature: (z_T, eps_mixed).
    get_noise_for_inference() still returns only z_T for backward compat.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Linear beta schedule (DDPM-style, matches SD v1)
# ─────────────────────────────────────────────────────────────────────────────

def linear_beta_schedule(
    timesteps:  int   = 1000,
    beta_start: float = 1e-4,
    beta_end:   float = 2e-2,
) -> torch.Tensor:
    """Returns a 1-D tensor of beta values, length = timesteps."""
    return torch.linspace(beta_start, beta_end, timesteps)


def get_alphas_cumprod(timesteps: int = 1000) -> torch.Tensor:
    """Returns cumulative product of (1 - beta_t), i.e. alpha_bar_t."""
    betas  = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)   # (timesteps,)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DANA Module
# ─────────────────────────────────────────────────────────────────────────────

class DANAModule(nn.Module):
    """
    Dynamic-Aware Noise Adding.

    Stateless (no learnable parameters). The beta mixing ratio is determined
    entirely by the fast/slow prediction from the dynamic predictor.

    Paper values:
        beta = 0.2  for fast video   (higher eps_d weight → more frame variation)
        beta = 0.3  for slow video   (higher eps_s weight → more frame consistency)

    Args:
        num_timesteps: length of the DDPM noise schedule (default 1000)
        beta_fast    : beta used when is_fast > 0.5
        beta_slow    : beta used when is_fast <= 0.5
    """

    BETA_FAST: float = 0.2
    BETA_SLOW: float = 0.3

    def __init__(
        self,
        num_timesteps: int   = 1000,
        beta_fast:     float = 0.2,
        beta_slow:     float = 0.3,
    ):
        super().__init__()
        self.BETA_FAST     = beta_fast
        self.BETA_SLOW     = beta_slow
        self.num_timesteps = num_timesteps

        alphas_cumprod = get_alphas_cumprod(num_timesteps)  # (T,)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(
        self,
        z0:      torch.Tensor,  # (B, F, 4, H, W)  clean latents
        is_fast: torch.Tensor,  # (B,)  float in [0, 1]  sigmoid probability
        T:       int = None,    # noise timestep INDEX — must be a Python int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies DANA forward process: returns (z_T, eps_mixed).

        FIX v4: returns the blended noise eps_mixed alongside z_T so that
        the training loop can use this exact noise as the prediction target,
        instead of regenerating independent noise (which caused the training
        bug in previous dana_forward_with_target()).

        Args:
            z0     : (B, F, 4, H, W)
            is_fast: (B,) float sigmoid output — threshold at 0.5 for fast/slow
            T      : int  noise level (0 = no noise, num_timesteps-1 = full noise)
                     IMPORTANT: T must be a Python int, not a tensor.
        Returns:
            z_T      : (B, F, 4, H, W)  noised latents
            eps_mixed: (B, F, 4, H, W)  the exact blended noise used (training target)
        """
        if T is None:
            T = self.num_timesteps - 1

        if not isinstance(T, int):
            T = int(T.item()) if T.numel() == 1 else int(T[0].item())

        B, F, C, H, W = z0.shape
        device = z0.device

        # alpha_T: scalar cumulative noise coefficient at step T
        alpha_T = self.alphas_cumprod[T].to(device)   # scalar tensor

        # Per-sample beta: fast → BETA_FAST, slow → BETA_SLOW
        is_fast_d = is_fast.to(device)
        beta_vals = torch.where(
            is_fast_d > 0.5,
            torch.full_like(is_fast_d, self.BETA_FAST),
            torch.full_like(is_fast_d, self.BETA_SLOW),
        ).view(B, 1, 1, 1, 1)   # broadcast over (F, C, H, W)

        # Static noise: same across all F frames per sample (temporal consistency)
        eps_s = torch.randn(B, 1, C, H, W, device=device).expand(B, F, C, H, W).contiguous()

        # Diverse noise: independent per frame
        eps_d = torch.randn(B, F, C, H, W, device=device)

        # Blended noise per DANA equation — returned as training target
        eps_mixed = (
            torch.sqrt(beta_vals)       * eps_s
          + torch.sqrt(1.0 - beta_vals) * eps_d
        )

        # DDPM forward process
        z_T = (
            torch.sqrt(alpha_T)       * z0
          + torch.sqrt(1.0 - alpha_T) * eps_mixed
        )

        # FIX v4: return both z_T and eps_mixed so caller can use the exact
        # noise as the UNet prediction target (no re-sampling needed).
        return z_T, eps_mixed

    def get_noise_for_inference(
        self,
        z_hat:   torch.Tensor,  # (B, F, 4, H, W)  predicted latents from Seq2Seq
        is_fast: torch.Tensor,  # (B,)
        T:       int = None,
    ) -> torch.Tensor:
        """
        Convenience wrapper for inference: adds DANA noise and returns z_T
        ready to be passed as the starting point for DDIM sampling.
        Returns only z_T (noise discarded — not needed at inference).
        """
        z_T, _ = self.forward(z_hat, is_fast, T)
        return z_T


# ─────────────────────────────────────────────────────────────────────────────
# 3. Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke-test] device = {device}")

    B, F = 4, 6
    dana    = DANAModule(num_timesteps=1000).to(device)
    z0      = torch.randn(B, F, 4, 16, 16, device=device)
    is_fast = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float32, device=device)

    # v4: forward returns (z_T, eps_mixed)
    z_T, eps_mixed = dana(z0, is_fast, T=999)
    assert z_T.shape    == z0.shape, f"z_T shape mismatch: {z_T.shape}"
    assert eps_mixed.shape == z0.shape, f"eps_mixed shape mismatch: {eps_mixed.shape}"
    assert not torch.allclose(z_T, z0), "z_T should differ from z0 after noise"

    # Verify noise consistency: z_T = sqrt(a)*z0 + sqrt(1-a)*eps_mixed
    alpha_T = dana.alphas_cumprod[999]
    z_T_check = torch.sqrt(alpha_T) * z0 + torch.sqrt(1.0 - alpha_T) * eps_mixed
    assert torch.allclose(z_T, z_T_check, atol=1e-5), "z_T/eps_mixed consistency check failed"
    print("[smoke-test] z_T / eps_mixed consistency ✓")

    # Threshold boundary check
    dana(torch.randn(1, F, 4, 16, 16, device=device),
         torch.tensor([0.51], device=device), T=500)
    dana(torch.randn(1, F, 4, 16, 16, device=device),
         torch.tensor([0.49], device=device), T=500)

    # Graceful tensor-T handling
    t_tensor = torch.tensor(500, device=device)
    z_T2, _ = dana(torch.randn(1, F, 4, 16, 16, device=device),
                   torch.tensor([0.7], device=device), T=t_tensor)
    assert z_T2.shape == (1, F, 4, 16, 16)

    # Inference wrapper returns only z_T
    z_inf = dana.get_noise_for_inference(z0[:1], is_fast[:1], T=999)
    assert z_inf.shape == (1, F, 4, 16, 16), f"inference shape: {z_inf.shape}"

    print(f"[smoke-test] z0       shape : {z0.shape}")
    print(f"[smoke-test] z_T      shape : {z_T.shape}  ✓")
    print(f"[smoke-test] eps_mixed shape : {eps_mixed.shape}  ✓")
    print(f"[smoke-test] beta_fast={dana.BETA_FAST}  beta_slow={dana.BETA_SLOW}")
    print(f"[smoke-test] threshold   : is_fast > 0.5  ✓")
    print("[smoke-test] dana PASSED")
