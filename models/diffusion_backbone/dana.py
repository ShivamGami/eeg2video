"""
dana.py — Dynamic-Aware Noise Addition (DANA) Module
=====================================================
Purpose:
  DANA selects the diffusion noise schedule beta parameter
  based on the Fast/Slow dynamics prediction from Sub-team 4.

Logic (from project spec):
  - Fast motion (is_fast=1.0): use higher beta → more aggressive
    noise addition → UNet generates more dynamic, varied frames
  - Slow motion (is_fast=0.0): use lower beta → gentler noise
    → UNet generates smoother, less dynamic frames

How it works:
  1. Take visual latent from Sub-team 3: (6, 4, 32, 32)
  2. Select beta based on is_fast flag
  3. Add scaled Gaussian noise to the latent
  4. Return noised latent as the starting point for VideoLDM denoising

  The noised latent is passed to Sub-team 2's UNet as the
  initial noisy input instead of pure Gaussian noise.
  This means the UNet "continues" from Sub-team 3's latent
  rather than generating from scratch — preserving structural
  information from the ViT while applying EEG conditioning.

Beta values:
  BETA_FAST = 0.85  — strong noise, fast/dynamic content
  BETA_SLOW = 0.35  — gentle noise, slow/static content

  These values were chosen so that:
  - BETA_SLOW preserves ~65% of the original latent signal
  - BETA_FAST preserves ~15% (mostly structure, UNet fills the rest)
"""

import torch
import torch.nn as nn


# Noise strength per dynamics class
BETA_FAST = 0.85   # fast motion → more noise → UNet generates dynamic content
BETA_SLOW = 0.35   # slow motion → less noise → preserve structure


class DANAModule(nn.Module):
    """
    Dynamic-Aware Noise Addition module.

    Takes Sub-team 3's visual latent and adds dynamics-aware noise
    to produce the starting point for Sub-team 2's denoising.

    Inputs:
        visual_latent : (B, 6, 4, 32, 32)  from Sub-team 3 ViT
        is_fast       : (B, 1)              from Sub-team 4 Dynamics MLP
                        0.0 = slow, 1.0 = fast

    Output:
        noised_latent : (B, 6, 4, 32, 32)  ready for VideoLDM denoising
        beta_used     : (B,) float          the beta value applied per sample
    """

    def __init__(self, beta_fast: float = BETA_FAST, beta_slow: float = BETA_SLOW):
        super().__init__()
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

    def forward(
        self,
        visual_latent: torch.Tensor,
        is_fast:       torch.Tensor,
    ):
        """
        Args:
            visual_latent : (B, 6, 4, 32, 32)
            is_fast       : (B, 1) values in {0.0, 1.0}

        Returns:
            noised_latent : (B, 6, 4, 32, 32)
            beta_used     : (B,)
        """
        B = visual_latent.shape[0]
        device = visual_latent.device

        # Select beta per sample based on dynamics flag
        # is_fast=1 → beta_fast, is_fast=0 → beta_slow
        is_fast_flag = is_fast.squeeze(1).float()           # (B,)
        beta = (
            is_fast_flag * self.beta_fast
            + (1 - is_fast_flag) * self.beta_slow
        )  # (B,) each value is either beta_fast or beta_slow

        # Reshape beta for broadcasting: (B, 1, 1, 1, 1)
        beta_view = beta.view(B, 1, 1, 1, 1)

        # Noise addition: x_noisy = sqrt(1-beta)*x + sqrt(beta)*eps
        # This is the standard diffusion forward process formula
        noise         = torch.randn_like(visual_latent)
        noised_latent = (
            torch.sqrt(1 - beta_view) * visual_latent
            + torch.sqrt(beta_view)   * noise
        )

        return noised_latent, beta


def verify_dana():
    dana   = DANAModule()
    latent = torch.randn(4, 6, 4, 32, 32)

    # Mixed fast/slow batch
    is_fast = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    noised, betas = dana(latent, is_fast)

    assert noised.shape == latent.shape
    assert len(betas) == 4
    assert abs(betas[0].item() - BETA_FAST) < 1e-5
    assert abs(betas[1].item() - BETA_SLOW) < 1e-5

    print(f"DANA verified | output shape: {noised.shape}")
    print(f"Betas applied : {betas.tolist()}")
    print(f"  Sample 0 (fast) beta={betas[0]:.2f} | "
          f"noise retained: {betas[0]:.0%}")
    print(f"  Sample 1 (slow) beta={betas[1]:.2f} | "
          f"noise retained: {betas[1]:.0%}")


if __name__ == "__main__":
    verify_dana()