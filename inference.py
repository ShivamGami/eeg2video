"""
inference.py  —  EEG-to-Video Inference Pipeline (v4 FINAL)
EEG2Video · IIT Mandi · CS 671  ·  Group 4

End-to-end sampling pipeline:
    EEG signal  →  CARD Encoder  →  DANA noise init  →  DDIM denoising
                →  LatentDecoder  →  uint8 video (B, F, 3, 128, 128)

Alternatively, pre-computed conditioning tensors stored in real_inputs/ can
be loaded directly, bypassing the CARD encoder (Phase 4 production path).

Key design decisions (aligned with v4 backbone):
  ▸ DDIM sampler uses the SAME alphas_cumprod buffer registered inside
    DANAModule — single source of truth for the noise schedule.
  ▸ is_fast threshold is EXPLICITLY 0.5 (v4 fix — never bool() cast).
  ▸ All tensors are moved to the correct device before use, preventing
    cross-device RuntimeErrors.
  ▸ LATENT_CH / LATENT_H / LATENT_W / N_FRAMES / TEXT_SEQ / TEXT_DIM
    imported from sd_backbone — no hardcoded spatial dimensions.
  ▸ decode_to_uint8() from LatentDecoder returns (B, F, 3, 128, 128) uint8.
  ▸ autocast used during the denoising loop for CUDA efficiency.
  ▸ All submodules set to eval() during inference to disable dropout/BN.
  ▸ from_real_inputs() factory loads pre-computed real_inputs/ tensors
    (visual_latents.pt, text_embeddings.pt, is_fast.pt) directly, bypassing
    the CARD encoder — the production path for Phase 4.
  ▸ from_checkpoints() factory supports optional real_models/ weights for
    all three subteam model files.

CARD Transformer architecture (project spec §Stage 3):
    3 stacked CARD blocks: intra-channel MHSA → inter-channel MHSA
    → 1D Conv token blending → residual + LayerNorm.
    Produces z_eeg ∈ R^512 per sample, projected into CLIP space.

Usage (from precomputed real_inputs — Phase 4 production):
    pipeline = EEG2VideoPipeline.from_checkpoints(device="cuda")
    video = pipeline.run_from_real_inputs(
        visual_latents_path  = "real_inputs/visual_latents.pt",
        text_embeddings_path = "real_inputs/text_embeddings.pt",
        is_fast_path         = "real_inputs/is_fast.pt",
    )
    # video: (B, F, 3, 128, 128) uint8

Usage (from raw EEG):
    pipeline = EEG2VideoPipeline.from_checkpoints(
        unet_ckpt        = "real_models/vit_real_data.pth",
        eeg_encoder_ckpt = None,
        device           = "cuda",
    )
    with torch.no_grad():
        video = pipeline(eeg_signal, text_emb=None)
        # video: (B, F, 3, 128, 128) uint8
"""

from __future__ import annotations

import sys
import os

# ── Path bootstrap so imports resolve whether run from project root or subdir ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ── v4 backbone imports (single source of truth for constants) ─────────────────
from sd_backbone import (
    TemporalUNet,
    LATENT_CH,
    LATENT_H,
    LATENT_W,
    N_FRAMES,
    TEXT_SEQ,
    TEXT_DIM,
)
from decoder import LatentDecoder
from dana import DANAModule


# ── autocast: prefer the non-deprecated torch.amp API (PyTorch ≥ 2.0) ─────────
def _autocast(device_type: str):
    """Returns an autocast context for the given device type."""
    try:
        return torch.amp.autocast(device_type=device_type)
    except AttributeError:
        # Fallback for PyTorch < 2.0
        from torch.cuda.amp import autocast
        return autocast(enabled=(device_type == "cuda"))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CARD Transformer Encoder
#     Implements the three-stage architecture from project spec §Stage 2–4:
#       (a) Temporal tokenisation (N=10 non-overlapping patches per channel)
#       (b) CARD blocks: intra-channel MHSA → inter-channel MHSA → Conv1D blend
#       (c) Flatten + linear projection → z_eeg ∈ R^512
#       (d) Two-layer MLP → CLIP-aligned embedding z ∈ R^512
#       (e) Dynamic predictor head: sigmoid → is_fast ∈ (0, 1)
# ─────────────────────────────────────────────────────────────────────────────

class CARDBlock(nn.Module):
    """
    Single CARD (Channel-Aligned Robust Blend) Transformer block.

    Per paper §3.2:
      1. Intra-channel MHSA  — each channel's N=10 temporal tokens attend to
         each other independently (shared weights across channels).
      2. Inter-channel MHSA  — each channel's single summary token attends to
         all other channel summaries (captures spatial brain correlations).
      3. 1D Conv token blending — depth-wise Conv1D across the N tokens per
         channel for multi-scale temporal abstraction.
      4. Residual + LayerNorm at every sub-block.

    Args:
        d_model : token embedding dimension (default 512)
        n_heads : number of attention heads for both MHSA layers
        n_tokens: number of temporal patches per channel N (default 10)
        n_ch    : number of EEG channels C (default 128)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_tokens: int = 10,
        n_ch: int = 128,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # (a) Intra-channel: attention across N temporal tokens, per channel
        self.intra_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.intra_norm = nn.LayerNorm(d_model)

        # (b) Inter-channel: attention across C channel summaries (mean-pooled)
        self.inter_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.inter_norm = nn.LayerNorm(d_model)

        # (c) 1D Conv token blending along the token axis (groups=d_model for
        #     depth-wise conv to keep parameters manageable)
        self.conv_blend = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        self.conv_norm = nn.LayerNorm(d_model)

        self.n_tokens = n_tokens
        self.n_ch = n_ch

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, C, N, d_model)   C=n_ch, N=n_tokens
        Returns:
            H: (B, C, N, d_model)   same shape
        """
        B, C, N, D = H.shape

        # ── (a) Intra-channel MHSA ─────────────────────────────────────────
        # Reshape: treat each (B, c) independently → (B*C, N, D)
        H_flat = H.reshape(B * C, N, D)
        attn_out, _ = self.intra_attn(H_flat, H_flat, H_flat)
        H_flat = self.intra_norm(H_flat + attn_out)
        H = H_flat.reshape(B, C, N, D)

        # ── (b) Inter-channel MHSA ─────────────────────────────────────────
        # Channel summary: mean over token axis → (B, C, D)
        ch_summary = H.mean(dim=2)                           # (B, C, D)
        inter_out, _ = self.inter_attn(ch_summary, ch_summary, ch_summary)
        ch_summary = self.inter_norm(ch_summary + inter_out) # (B, C, D)
        # Broadcast back and add as residual to H
        H = H + ch_summary.unsqueeze(2)                     # (B, C, N, D)

        # ── (c) 1D Conv token blending ─────────────────────────────────────
        # Treat token axis as sequence: (B*C, D, N) for Conv1d
        H_conv = H.reshape(B * C, N, D).permute(0, 2, 1)   # (B*C, D, N)
        conv_out = self.conv_blend(H_conv).permute(0, 2, 1) # (B*C, N, D)
        H = self.conv_norm(H.reshape(B * C, N, D) + conv_out).reshape(B, C, N, D)

        return H


class CARDEncoder(nn.Module):
    """
    Full CARD Transformer encoder.

    Stages (per project spec §Stage 2–5):
      1. Linear patch embedding: X_{c,n} ∈ R^P → h_{c,n} ∈ R^d  (N=10, P=44)
      2. 3× CARDBlock
      3. Flatten + linear → z_eeg ∈ R^512
      4. Two-layer MLP + L2-norm → z_clip ∈ R^512 (CLIP-aligned)
      5. Sigmoid head → is_fast ∈ (0, 1)  (dynamic predictor)

    Args:
        n_ch      : EEG channels (default 128)
        n_tokens  : temporal patches per channel (default 10, P=44 samples each)
        patch_dim : raw patch length in samples (default 44)
        d_model   : latent dimension (default 512)
        n_blocks  : number of stacked CARD blocks (default 3, per paper)
        clip_dim  : output CLIP embedding dimension (default 512)
    """

    def __init__(
        self,
        n_ch: int = 128,
        n_tokens: int = 10,
        patch_dim: int = 44,
        d_model: int = 512,
        n_blocks: int = 3,
        clip_dim: int = 512,
    ):
        super().__init__()
        self.n_ch = n_ch
        self.n_tokens = n_tokens

        # Stage 1: linear patch embedding (shared across channels, per paper)
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # Stage 2: stacked CARD blocks
        self.card_blocks = nn.ModuleList([
            CARDBlock(d_model=d_model, n_heads=8, n_tokens=n_tokens, n_ch=n_ch)
            for _ in range(n_blocks)
        ])

        # Stage 3: flatten + project → z_eeg ∈ R^512
        self.latent_proj = nn.Linear(n_ch * n_tokens * d_model, clip_dim)

        # Stage 4: two-layer CLIP projection MLP (per paper Eq. 3)
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, clip_dim * 2),
            nn.GELU(),
            nn.Linear(clip_dim * 2, clip_dim),
        )

        # Stage 5: dynamic predictor (sigmoid → is_fast probability)
        self.dynamic_head = nn.Sequential(
            nn.Linear(clip_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.clip_dim = clip_dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T)  raw EEG — C channels, T samples (440 at 880 Hz)
        Returns:
            z_clip  : (B, 512)  CLIP-aligned EEG embedding (L2-normalised)
            z_eeg   : (B, 512)  pre-projection latent (used for composite loss)
            is_fast : (B,)      sigmoid fast/slow probability — threshold at 0.5
        """
        B, C, T = x.shape

        # Reshape into patches: (B, C, N, P)
        N, P = self.n_tokens, T // self.n_tokens
        x_patches = x[:, :, : N * P].reshape(B, C, N, P)

        # Stage 1: patch embedding → (B, C, N, d_model)
        H = self.patch_embed(x_patches)

        # Stage 2: CARD blocks
        for block in self.card_blocks:
            H = block(H)

        # Stage 3: flatten + project
        z_eeg = self.latent_proj(H.reshape(B, -1))          # (B, 512)

        # Stage 4: CLIP projection + L2 normalisation
        z_clip = F.normalize(self.clip_proj(z_eeg), dim=-1) # (B, 512)

        # Stage 5: dynamic predictor — is_fast > 0.5 → fast video (v4 fix)
        is_fast = self.dynamic_head(z_eeg).squeeze(-1)      # (B,)

        return z_clip, z_eeg, is_fast


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DDIM Sampler (synchronized with DANA alpha schedule)
#
#     DDIM (Song et al., 2020) deterministic sampling using the SAME
#     alphas_cumprod that DANAModule registers as a buffer, so the
#     forward (DANA) and reverse (DDIM) processes share identical schedules.
#
#     Reverse step (DDIM, η=0 deterministic):
#       x_{t-1} = sqrt(α_{t-1}) * x0_pred
#               + sqrt(1 - α_{t-1}) * eps_pred
#
#     where:
#       eps_pred = (x_t - sqrt(α_t) * x0_pred) / sqrt(1 - α_t)
#       x0_pred  = (x_t - sqrt(1 - α_t) * unet_out) / sqrt(α_t)
# ─────────────────────────────────────────────────────────────────────────────

class DDIMSampler:
    """
    Deterministic DDIM sampler synchronized with the DANA noise schedule.

    The alphas_cumprod tensor is taken directly from a DANAModule instance
    to guarantee that the forward (noising) and reverse (denoising) schedules
    are bit-for-bit identical — eliminating a common class of schedule mismatch
    bugs that arise when these are defined in separate files.

    Args:
        dana        : DANAModule whose alphas_cumprod buffer we borrow
        n_steps     : number of DDIM inference steps (default 50)
        device      : target device
    """

    def __init__(
        self,
        dana: DANAModule,
        n_steps: int = 50,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_steps = n_steps
        self.device = device

        # Borrow the schedule from DANA — single source of truth
        alphas_full = dana.alphas_cumprod.to(device)       # (1000,)
        total_T = len(alphas_full)

        # Uniformly subsample timestep indices from T-1 down to 0
        step_size = total_T // n_steps
        ts = list(reversed(range(0, total_T, step_size)))[:n_steps]
        self.timesteps = torch.tensor(ts, dtype=torch.long, device=device)

        # Pre-index alpha values at selected timesteps
        self.alphas = alphas_full[self.timesteps]          # (n_steps,)

        # Alpha at the step BEFORE each selected timestep (for DDIM update)
        prev_ts = torch.cat([
            self.timesteps[1:],
            torch.tensor([0], device=device)
        ])
        self.alphas_prev = alphas_full[prev_ts]            # (n_steps,)

    @torch.no_grad()
    def step(
        self,
        unet_out: torch.Tensor,   # (B, F, 4, H, W)  predicted noise from UNet
        x_t: torch.Tensor,         # (B, F, 4, H, W)  current noisy latents
        step_idx: int,             # index into self.timesteps (0 = most noisy)
    ) -> torch.Tensor:
        """
        Performs one DDIM reverse step.

        Returns:
            x_{t-1}: (B, F, 4, H, W)  less-noisy latents
        """
        alpha_t    = self.alphas[step_idx]
        alpha_prev = self.alphas_prev[step_idx]

        # Predict x0 from current x_t and UNet noise estimate
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_t) * unet_out) / torch.sqrt(alpha_t)
        x0_pred = x0_pred.clamp(-1.0, 1.0)                    # stabilise

        # Direction pointing to x_t
        eps_pred = (x_t - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1.0 - alpha_t)

        # DDIM deterministic update (η=0)
        x_prev = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1.0 - alpha_prev) * eps_pred
        return x_prev


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EEG2VideoPipeline
# ─────────────────────────────────────────────────────────────────────────────

class EEG2VideoPipeline(nn.Module):
    """
    Complete EEG-to-Video inference pipeline (v4 FINAL).

    Architecture (Group 4 spec):
      EEG → CARDEncoder → z_clip + is_fast
                       ↓
              DANA initialises z_T using EXPLICIT is_fast > 0.5 threshold:
                  is_fast > 0.5  →  beta=0.2  (more diverse noise, fast video)
                  is_fast ≤ 0.5  →  beta=0.3  (more static noise, slow video)
                       ↓
           DDIM denoising (TemporalUNet conditioned on z_clip as text_emb proxy)
                       ↓
           LatentDecoder.decode_to_uint8 → (B, F, 3, 128, 128) uint8

    The DDIM sampler reads its alpha schedule from the same DANAModule buffer
    registered at init, keeping forward and reverse processes on the same schedule.

    Production path (Phase 4):
      run_from_real_inputs() loads pre-computed tensors from real_inputs/ and
      runs DANA + DDIM + decode directly — bypassing the CARD encoder entirely.
      This uses the actual Sub-team outputs:
          real_inputs/visual_latents.pt   → (B, 6, 4, 32, 32)
          real_inputs/text_embeddings.pt  → (B, 77, 512)
          real_inputs/is_fast.pt          → (B,)

    Args:
        unet         : TemporalUNet (denoising backbone)
        decoder      : LatentDecoder (latent → RGB frames)
        eeg_encoder  : CARDEncoder (EEG → CLIP embedding + is_fast)
        dana         : DANAModule  (DANA noise init + shared alpha schedule)
        n_ddim_steps : DDIM inference steps (default 50)
        device       : inference device
    """

    def __init__(
        self,
        unet: TemporalUNet,
        decoder: LatentDecoder,
        eeg_encoder: CARDEncoder,
        dana: DANAModule,
        n_ddim_steps: int = 50,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.unet = unet.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.eeg_encoder = eeg_encoder.to(device).eval()
        self.dana = dana.to(device)
        self.device = device
        self.n_frames = N_FRAMES  # from sd_backbone constants

        self.sampler = DDIMSampler(dana=dana, n_steps=n_ddim_steps, device=device)

    @classmethod
    def from_checkpoints(
        cls,
        unet_ckpt: Optional[str] = None,
        decoder_ckpt: Optional[str] = None,
        eeg_encoder_ckpt: Optional[str] = None,
        n_ddim_steps: int = 50,
        device: str = "cuda",
    ) -> "EEG2VideoPipeline":
        """
        Factory constructor: builds pipeline from optional checkpoint paths.
        If a checkpoint is None, the component is randomly initialised (Phase 4).

        Supports the real_models/ checkpoint naming from the project spec:
            real_models/vit_real_data.pth   → UNet or EEG encoder (sub-team 3)
            real_models/text_mlp_final.pth  → (not used here; consumed by real_inputs)
            real_models/dynamics_model.pth  → (not used here; consumed by real_inputs)

        Args:
            unet_ckpt        : path to TemporalUNet state dict (.pth)
            decoder_ckpt     : path to LatentDecoder VAE weights (.pth) or None
            eeg_encoder_ckpt : path to CARDEncoder state dict (.pth) or None
            n_ddim_steps     : DDIM inference steps
            device           : "cuda" or "cpu"
        Returns:
            EEG2VideoPipeline ready for inference
        """
        _device = torch.device(device if torch.cuda.is_available() else "cpu")

        unet = TemporalUNet(latent_ch=LATENT_CH, text_dim=TEXT_DIM, n_frames=N_FRAMES)
        if unet_ckpt is not None and os.path.isfile(unet_ckpt):
            state = torch.load(unet_ckpt, map_location="cpu")
            unet.load_state_dict(state.get("state_dict", state), strict=False)
            print(f"[EEG2VideoPipeline] loaded UNet from {unet_ckpt}")

        decoder = LatentDecoder(latent_ch=LATENT_CH, vae_ckpt=decoder_ckpt)

        eeg_encoder = CARDEncoder()
        if eeg_encoder_ckpt is not None and os.path.isfile(eeg_encoder_ckpt):
            state = torch.load(eeg_encoder_ckpt, map_location="cpu")
            eeg_encoder.load_state_dict(state.get("state_dict", state), strict=False)
            print(f"[EEG2VideoPipeline] loaded CARDEncoder from {eeg_encoder_ckpt}")

        dana = DANAModule(num_timesteps=1000)

        return cls(
            unet=unet,
            decoder=decoder,
            eeg_encoder=eeg_encoder,
            dana=dana,
            n_ddim_steps=n_ddim_steps,
            device=_device,
        )

    # ── Production path: load pre-computed real_inputs/ tensors ───────────

    @torch.no_grad()
    def run_from_real_inputs(
        self,
        visual_latents_path:  str = "real_inputs/visual_latents.pt",
        text_embeddings_path: str = "real_inputs/text_embeddings.pt",
        is_fast_path:         str = "real_inputs/is_fast.pt",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Production inference using pre-computed Sub-team conditioning tensors.

        Loads:
            visual_latents  : (B, 6, 4, 32, 32) — Sub-team 3 ViT output
            text_embeddings : (B, 77, 512)        — Sub-team 4 TextMLP output
            is_fast         : (B,)                — Sub-team 4 DynamicsClassifier

        Runs DANA noise mixing → DDIM denoising → LatentDecoder.

        The is_fast > 0.5 threshold is applied INSIDE DANAModule (BETA_FAST=0.2
        vs BETA_SLOW=0.3) — this function just passes the raw sigmoid values.

        Returns:
            video: (B, F, 3, 128, 128) uint8 in [0, 255]
        """
        if seed is not None:
            torch.manual_seed(seed)

        dev = self.device

        # ── Load pre-computed tensors ───────────────────────────────────────
        for path in (visual_latents_path, text_embeddings_path, is_fast_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[run_from_real_inputs] Required file not found: {path}\n"
                    f"  Generate it by running the Sub-team 3/4 model scripts first."
                )

        visual_latents  = torch.load(visual_latents_path,  map_location=dev).float()
        text_embeddings = torch.load(text_embeddings_path, map_location=dev).float()
        is_fast         = torch.load(is_fast_path,         map_location=dev).float()

        # ── Validate shapes ─────────────────────────────────────────────────
        B = visual_latents.shape[0]
        assert visual_latents.shape  == (B, N_FRAMES, LATENT_CH, visual_latents.shape[-2], visual_latents.shape[-1]), \
            f"visual_latents: expected (B,{N_FRAMES},{LATENT_CH},H,W), got {visual_latents.shape}"
        assert text_embeddings.shape == (B, TEXT_SEQ, TEXT_DIM), \
            f"text_embeddings: expected ({B},{TEXT_SEQ},{TEXT_DIM}), got {text_embeddings.shape}"
        assert is_fast.shape == (B,), \
            f"is_fast: expected ({B},), got {is_fast.shape}"

        # ── DANA noise mixing using visual_latents as z0 ───────────────────
        # is_fast > 0.5 → BETA_FAST=0.2 (diverse noise), else BETA_SLOW=0.3
        x_t = self.dana.get_noise_for_inference(
            z_hat=visual_latents,
            is_fast=is_fast,
            T=self.dana.num_timesteps - 1,
        )   # (B, F, LATENT_CH, H, W)

        # ── DDIM denoising loop ─────────────────────────────────────────────
        x_t = self._ddim_loop(x_t, text_embeddings, B)

        # ── Decode latents → uint8 video ────────────────────────────────────
        return self.decoder.decode_to_uint8(x_t)

    # ── Internal DDIM denoising loop ──────────────────────────────────────

    def _ddim_loop(
        self,
        x_t: torch.Tensor,        # (B, F, CH, H, W) noisy latents
        text_emb: torch.Tensor,   # (B, 77, 512)
        B: int,
    ) -> torch.Tensor:
        """
        Runs the DDIM denoising loop for n_steps steps.

        Uses the DANA alphas_cumprod schedule (single source of truth).
        All timestep tensors are already on self.device (set in DDIMSampler.__init__).

        Returns:
            x_0: (B, F, CH, H, W) denoised latents
        """
        dev = self.device
        dev_type = dev.type if hasattr(dev, "type") else str(dev).split(":")[0]

        for step_idx in range(self.sampler.n_steps):
            t_global = self.sampler.timesteps[step_idx]          # scalar, on dev
            t_batch  = t_global.expand(B).to(dev)                # (B,)

            with _autocast(dev_type):
                noise_pred = self.unet(
                    latents=x_t.float(),
                    text_emb=text_emb,
                    timestep=t_batch,
                )   # (B, F, CH, H, W)

            # DDIM reverse step (deterministic, η=0)
            x_t = self.sampler.step(
                unet_out=noise_pred.float(),
                x_t=x_t.float(),
                step_idx=step_idx,
            )   # (B, F, CH, H, W)

        return x_t

    # ── Standard forward: raw EEG → video ─────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        eeg_signal: torch.Tensor,               # (B, C_eeg, T)
        text_emb: Optional[torch.Tensor] = None, # (B, 77, 512) or None
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run full EEG-to-video inference from raw EEG signals.

        Args:
            eeg_signal : (B, C_eeg, T)  preprocessed EEG (z-scored, bandpassed)
            text_emb   : (B, 77, 512)   CLIP text embeddings (optional; if None,
                         the z_clip embedding is used as a proxy via projection)
            seed       : optional random seed for reproducibility
        Returns:
            video: (B, F, 3, 128, 128)  uint8 RGB frames in [0, 255]
        """
        if seed is not None:
            torch.manual_seed(seed)

        B = eeg_signal.shape[0]
        dev = self.device
        dev_type = dev.type if hasattr(dev, "type") else str(dev).split(":")[0]

        # ── Move inputs to device ───────────────────────────────────────────
        eeg_signal = eeg_signal.to(dev)

        # ── Stage 1: CARD encoder → z_clip, is_fast ────────────────────────
        # is_fast ∈ (0, 1) — dynamic predictor output (sigmoid)
        # Explicit threshold 0.5 (v4 fix: never use .bool() casting)
        with _autocast(dev_type):
            z_clip, z_eeg, is_fast = self.eeg_encoder(eeg_signal)
            # z_clip: (B, 512), is_fast: (B,) float in (0, 1)

        # ── Build text_emb for UNet cross-attention ─────────────────────────
        # UNet expects (B, TEXT_SEQ=77, TEXT_DIM=512).
        # If no text prompt provided, broadcast z_clip as a single-token
        # proxy and zero-pad to 77 tokens.
        if text_emb is None:
            # Shape: (B, 1, 512) → (B, 77, 512)
            proxy   = z_clip.unsqueeze(1)                                # (B, 1, 512)
            padding = torch.zeros(B, TEXT_SEQ - 1, TEXT_DIM, device=dev)
            text_emb = torch.cat([proxy, padding], dim=1).float()         # (B, 77, 512)
        else:
            text_emb = text_emb.to(dev).float()
            if text_emb.shape != (B, TEXT_SEQ, TEXT_DIM):
                raise ValueError(
                    f"text_emb must be (B={B}, {TEXT_SEQ}, {TEXT_DIM}), "
                    f"got {tuple(text_emb.shape)}"
                )

        # ── Stage 2: DANA noise initialisation ─────────────────────────────
        # DANA adds structured noise to a zero latent start point.
        # The is_fast > 0.5 threshold selects the noise mixing coefficient:
        #   is_fast > 0.5  →  beta=0.2  →  more diverse noise (fast dynamics)
        #   is_fast ≤ 0.5  →  beta=0.3  →  more static noise  (slow dynamics)
        # This matches DANA Equation (2) from the EEG2Video paper.
        z0 = torch.zeros(B, self.n_frames, LATENT_CH, LATENT_H, LATENT_W, device=dev)
        is_fast_dev = is_fast.to(dev).float()
        x_t = self.dana.get_noise_for_inference(
            z_hat=z0,
            is_fast=is_fast_dev,   # explicit float tensor — threshold inside DANAModule
            T=self.dana.num_timesteps - 1,
        )   # (B, F, LATENT_CH, LATENT_H, LATENT_W)

        # ── Stage 3: DDIM denoising loop ───────────────────────────────────
        x_t = self._ddim_loop(x_t, text_emb, B)

        # ── Stage 4: decode latents → uint8 video ──────────────────────────
        # decode_to_uint8 returns (B, F, 3, 128, 128) uint8 in [0, 255]
        return self.decoder.decode_to_uint8(x_t)

    def __repr__(self) -> str:
        def _n(m):
            return sum(p.numel() for p in m.parameters())

        return (
            f"EEG2VideoPipeline(\n"
            f"  CARDEncoder  : {_n(self.eeg_encoder):>12,} params\n"
            f"  TemporalUNet : {_n(self.unet):>12,} params\n"
            f"  LatentDecoder: {_n(self.decoder):>12,} params\n"
            f"  DDIM steps   : {self.sampler.n_steps}\n"
            f"  Output       : (B, {N_FRAMES}, 3, 128, 128) uint8\n"
            f"  Device       : {self.device}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Quick sanity-check  (run directly: python inference.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inference smoke-test] device = {device_str}")

    pipeline = EEG2VideoPipeline.from_checkpoints(
        unet_ckpt=None,
        decoder_ckpt=None,
        eeg_encoder_ckpt=None,
        n_ddim_steps=5,     # tiny for speed
        device=device_str,
    )
    print(pipeline)

    B = 2
    # EEG: 128 channels, 440 samples (880 Hz × 500 ms, per paper spec)
    eeg = torch.randn(B, 128, 440)

    with torch.no_grad():
        video = pipeline(eeg, text_emb=None, seed=42)

    assert video.shape == (B, N_FRAMES, 3, 128, 128), (
        f"Expected ({B}, {N_FRAMES}, 3, 128, 128), got {video.shape}"
    )
    assert video.dtype == torch.uint8, f"Expected uint8, got {video.dtype}"
    print(f"[inference smoke-test] output shape : {video.shape}  dtype={video.dtype}  ✓")
    print("[inference smoke-test] PASSED")
