"""
dummy_data.py
=============
Sub-team 2 – Generative Backbone | Phase 1 Dummy Data
CS 671 EEG2Video Reproduction | Team 22

Generates all dummy tensors matching the Phase 1 interface contracts agreed
upon by all sub-teams. Sub-team 2 uses visual_latents and text_embeddings
as inputs to the SD fine-tuning backbone.

NO real data needed – all tensors are torch.randn() stubs.

Usage:
    from dummy_data import get_dummy_batch
    batch = get_dummy_batch(batch_size=2)
"""

import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
#  Environment Safeguard – MUST be eeg2video_env (server & local)
# ─────────────────────────────────────────────────────────────────────────────
_REQUIRED_ENV = "eeg2video_env"
_active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
if _active_env != _REQUIRED_ENV:
    print("\n" + "!"*55)
    print(f"  [BLOCKED] Wrong conda environment detected.")
    print(f"  Active  : '{_active_env or 'none'}'")
    print(f"  Required: '{_REQUIRED_ENV}'")
    print(f"\n  Fix: conda activate {_REQUIRED_ENV}")
    print("!"*55 + "\n")
    sys.exit(1)

import torch


# ─────────────────────────────────────────────────────────────────────────────
#  Interface Contracts  (agreed in Phase 1 – DO NOT change these shapes)
# ─────────────────────────────────────────────────────────────────────────────
#
#   EEG Input        : (Batch, 7, Channels, 100)
#   Visual Latents   : (Batch, 6, 4, H, W)      ← 6 frames, 4 latent channels
#   Text Embeddings  : (Batch, 77, 768)          ← CLIP token space
#
#  Sub-team 2 consumes: visual_latents + text_embeddings
#  Sub-team 3 produces: visual_latents  (from EEG)
#  Sub-team 4 produces: text_embeddings (from EEG)
# ─────────────────────────────────────────────────────────────────────────────

# SD VAE latent spatial size (SD 1.x encodes 256×256 → 32×32 latents)
LATENT_H = 32
LATENT_W = 32
LATENT_C = 4       # SD VAE latent channels
NUM_FRAMES = 6     # 2-sec clip at 3 FPS
EEG_CHANNELS = 62  # SEED-DV electrode count
EEG_SEGMENTS = 7
EEG_TIMESTEPS = 100
TEXT_SEQ_LEN = 77  # CLIP max token length
TEXT_DIM = 768     # CLIP text embedding dim (SD 1.x)


def get_dummy_batch(batch_size: int = 2, device: str = "cpu") -> dict:
    """
    Returns a dict of all dummy tensors for one training batch.

    Args:
        batch_size: number of samples in the batch (keep ≤ 2 on CPU)
        device:     'cpu' for local testing, 'cuda' on server

    Returns:
        dict with keys:
            eeg_signal      – (B, 7, 62, 100)   raw EEG input
            visual_latents  – (B, 6, 4, 32, 32) VAE-encoded video frames
            text_embeddings – (B, 77, 768)       SD text encoder output
            noise_target    – (B, 6, 4, 32, 32) noise SD tries to predict
            timesteps       – (B,)               diffusion timestep indices
    """
    B = batch_size

    eeg_signal      = torch.randn(B, EEG_SEGMENTS, EEG_CHANNELS, EEG_TIMESTEPS).to(device)
    visual_latents  = torch.randn(B, NUM_FRAMES, LATENT_C, LATENT_H, LATENT_W).to(device)
    text_embeddings = torch.randn(B, TEXT_SEQ_LEN, TEXT_DIM).to(device)

    # Noise target: what the UNet should predict (same shape as latents)
    noise_target    = torch.randn_like(visual_latents)

    # Diffusion timesteps: random integers in [0, 1000)
    timesteps       = torch.randint(0, 1000, (B,), device=device)

    return {
        "eeg_signal":      eeg_signal,
        "visual_latents":  visual_latents,
        "text_embeddings": text_embeddings,
        "noise_target":    noise_target,
        "timesteps":       timesteps,
    }


def get_dummy_noisy_latents(
    visual_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Simulates the SD forward diffusion process (q(x_t | x_0)).
    Adds scaled noise to clean latents at the given timestep.

    In real training this is handled by DDPMScheduler.add_noise(),
    but this stub lets you test shapes without loading the full scheduler.

    Args:
        visual_latents : (B, 6, 4, 32, 32) clean latents from VAE
        timesteps      : (B,) integer timesteps in [0, 1000)
        noise          : optional pre-generated noise (same shape as latents)

    Returns:
        noisy_latents  : (B, 6, 4, 32, 32)
    """
    if noise is None:
        noise = torch.randn_like(visual_latents)

    # Simplified linear noise schedule for dummy purposes only
    # Real training will use scheduler.add_noise()
    alpha = 1.0 - timesteps.float() / 1000.0  # (B,)
    # Broadcast alpha over all latent dims: (B,) → (B,1,1,1,1)
    alpha = alpha.view(-1, 1, 1, 1, 1)
    noisy_latents = alpha.sqrt() * visual_latents + (1 - alpha).sqrt() * noise
    return noisy_latents


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nDummy Data – Shape Verification")
    print("="*45)

    batch = get_dummy_batch(batch_size=2)
    for key, tensor in batch.items():
        print(f"  {key:<22} : {tuple(tensor.shape)}")

    # Test noisy latent generation
    noisy = get_dummy_noisy_latents(
        batch["visual_latents"],
        batch["timesteps"],
        batch["noise_target"],
    )
    print(f"\n  noisy_latents (output)  : {tuple(noisy.shape)}")
    print("\n  [OK] All dummy tensors match interface contracts.")
