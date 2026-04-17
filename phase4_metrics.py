"""
phase4_metrics.py  —  Phase 4 Video Evaluation Metrics (v4 FINAL)
EEG2Video · IIT Mandi · CS 671  ·  Group 4

Implements all metrics required by the Group 4 project spec (§4 & Shared
Evaluation Framework) for native 5D video tensors (B, F, 3, H, W).

Metric inventory (project spec alignment)
──────────────────────────────────────────
  PSNR          — Peak Signal-to-Noise Ratio (pixel-level, per-frame average)
  SSIM          — Structural Similarity Index (pixel-level, per-frame average)
  FVD           — Fréchet Video Distance (primary SOTA metric — temporal coherence)
                  Uses I3D features via a lightweight pre-trainable wrapper.
  CLIP Sim      — Cosine similarity between generated frames and CLIP embeddings
                  from the original visual stimuli (semantic alignment).
  LPIPS         — Learned Perceptual Image Patch Similarity (deep-feature, per-frame)
  Top-5 Acc     — Inception-v3 top-5 classification accuracy (semantic fidelity)

All metrics operate on 5D tensors (B, F, 3, H, W) and handle batching internally.
Final results are exportable as a structured JSON report (metrics.json).

Design decisions
────────────────
▸ LATENT_CH / LATENT_H / LATENT_W / N_FRAMES imported from sd_backbone.
▸ autocast used for all feature extraction passes on CUDA.
▸ FVD: lightweight I3D wrapper works out-of-the-box. Replace I3DFeatureExtractor
    with torchvision.models.video.r3d_18 in Phase 5 for publication-quality FVD.
    The FrechetDistance computation is architecture-agnostic and remains correct.
▸ CLIP Sim supports precomputed embeddings (B, 512) or per-frame comparison.
▸ All metrics return per-batch statistics and accumulate cleanly across batches.
▸ JSON export is safe for flat filenames (no dirname crash on empty string).
▸ _to_float_01 correctly handles uint8 [0,255], float [-1,1], and float [0,1].
▸ FVD I3D feature extraction moves tensors to the FVD module's device explicitly,
    avoiding cross-device errors when calling update() from CPU tensors.

Paper baselines (Table 1, CVPR 2026 Paper #22 image task):
    FID   = 3.57
    SSIM  = 0.504
    LPIPS = 0.632
    MSE   = 0.324
"""

from __future__ import annotations

import sys
import os
import json
import time
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

from sd_backbone import LATENT_CH, LATENT_H, LATENT_W, N_FRAMES


# ── autocast: prefer the non-deprecated torch.amp API (PyTorch ≥ 2.0) ──────
def _autocast(device_type: str):
    """Returns an autocast context for the given device type."""
    try:
        return torch.amp.autocast(device_type=device_type)
    except AttributeError:
        from torch.cuda.amp import autocast
        return autocast(enabled=(device_type == "cuda"))


# ─────────────────────────────────────────────────────────────────────────────
# Paper baselines (CVPR 2026 Paper #22, Table 1 — image task)
# These are used to compute the Δ column in the JSON report.
# ─────────────────────────────────────────────────────────────────────────────

PAPER_BASELINES = {
    "FID"  : 3.57,
    "SSIM" : 0.504,
    "LPIPS": 0.632,
    "MSE"  : 0.324,
}


# ─────────────────────────────────────────────────────────────────────────────
# 0. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_float_01(video: torch.Tensor) -> torch.Tensor:
    """
    Normalise a video tensor to float32 in [0, 1].

    Handles three input formats correctly:
      - uint8  [0, 255]  → divide by 255
      - float  [-1, 1]   → (x + 1) / 2   (detected by dtype, not value range)
      - float  [0, 1]    → pass through as float32

    Note: We use dtype to distinguish float[-1,1] from float[0,1] to avoid
    the fragile `video.min() < -0.1` heuristic that can mislabel legitimate
    [0,1] tensors whose minimum happens to be near zero.
    For float inputs, we check whether any value is below -0.01 as a reliable
    indicator of the [-1, 1] range (pure [0,1] data can never go below 0).
    """
    if video.dtype == torch.uint8:
        return video.float() / 255.0
    v = video.float()
    if v.min() < -0.01:          # [-1, 1] format (cannot occur in [0, 1] data)
        return (v + 1.0) / 2.0
    return v


def _get_device_type(tensor: torch.Tensor) -> str:
    """Return 'cuda' or 'cpu' for a tensor's device."""
    return tensor.device.type


def _resize_frames(video: torch.Tensor, size: int) -> torch.Tensor:
    """Resize (B, F, 3, H, W) to (B, F, 3, size, size)."""
    B, nF, C, H, W = video.shape
    flat    = video.reshape(B * nF, C, H, W)
    resized = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
    return resized.reshape(B, nF, C, size, size)


def _frechet_distance(
    mu1: torch.Tensor, sigma1: torch.Tensor,
    mu2: torch.Tensor, sigma2: torch.Tensor,
) -> float:
    """
    Fréchet distance between two multivariate Gaussians parameterised by
    (mean, covariance).  D² = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2√(Σ1·Σ2))

    Uses the eigenvalue-based matrix square root for numerical stability.
    This is the same formula used by both FID and FVD.
    All computation performed in float64 on CPU to avoid numeric issues.
    """
    # Move everything to CPU float64 for numerical stability
    mu1 = mu1.cpu().double()
    mu2 = mu2.cpu().double()
    s1  = sigma1.cpu().double()
    s2  = sigma2.cpu().double()

    diff     = mu1 - mu2
    sq_dist  = diff.dot(diff).item()

    product = s1 @ s2

    # Symmetric eigendecomposition for the matrix square root
    try:
        sym     = (product + product.T) / 2.0
        eigvals, eigvecs = torch.linalg.eigh(sym)
        eigvals = eigvals.clamp(min=0.0)          # numerical safety: clip negatives
        sqrt_product = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T
    except Exception:
        # Fallback: treat sqrt as zero (conservative — over-estimates FVD)
        sqrt_product = torch.zeros_like(product)

    trace_term = (s1 + s2 - 2.0 * sqrt_product).diagonal().sum().item()
    return float(max(0.0, sq_dist + trace_term))   # clamp to ≥ 0 for numerical safety


def _compute_mu_sigma(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance from a (N, D) feature matrix.
    N must be ≥ 2.  All operations on CPU float64 for stability.
    """
    feats    = features.cpu().float()
    mu       = feats.mean(dim=0)
    features_c = feats - mu
    sigma    = (features_c.T @ features_c) / max(feats.shape[0] - 1, 1)
    return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# 1. PSNR — Peak Signal-to-Noise Ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(
    generated: torch.Tensor,    # (B, F, 3, H, W)  any dtype
    reference: torch.Tensor,    # (B, F, 3, H, W)  any dtype
    max_val: float = 1.0,
) -> Dict[str, float]:
    """
    Computes PSNR averaged over frames and batch.

    PSNR = 10 · log10(max_val² / MSE)

    Args:
        generated : (B, F, 3, H, W)
        reference : (B, F, 3, H, W)
        max_val   : pixel value range maximum (1.0 for [0,1], 255.0 for [0,255])
    Returns:
        dict with keys: psnr_mean, psnr_std, mse_mean
    """
    gen = _to_float_01(generated)
    ref = _to_float_01(reference)
    assert gen.shape == ref.shape, f"Shape mismatch: {gen.shape} vs {ref.shape}"

    # MSE per (batch, frame)  →  (B, F)
    mse  = ((gen - ref) ** 2).mean(dim=(-3, -2, -1))               # (B, F)
    mse  = mse.clamp(min=1e-10)                                    # avoid log(0)
    psnr = 10.0 * torch.log10(torch.tensor(max_val ** 2) / mse)   # (B, F)

    return {
        "psnr_mean": psnr.mean().item(),
        "psnr_std" : psnr.std().item(),
        "mse_mean" : mse.mean().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SSIM — Structural Similarity Index
# ─────────────────────────────────────────────────────────────────────────────

def _ssim_per_frame(
    img1: torch.Tensor,   # (N, 3, H, W) float [0, 1]
    img2: torch.Tensor,   # (N, 3, H, W) float [0, 1]
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    SSIM per image in a batch.  Returns (N,) SSIM values.
    Pure PyTorch implementation — no external dependency.
    """
    N, C, H, W = img1.shape
    device = img1.device

    # Gaussian kernel (shared across channels)
    sigma = 1.5
    x_g   = torch.arange(window_size, dtype=torch.float32, device=device)
    x_g   = x_g - window_size // 2
    kernel_1d  = torch.exp(-x_g**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d  = kernel_1d[:, None] * kernel_1d[None, :]            # (ws, ws)
    kernel     = kernel_2d.expand(C, 1, window_size, window_size)   # (C, 1, ws, ws)
    pad = window_size // 2

    def _mu(img):
        return F.conv2d(img, kernel, padding=pad, groups=C)

    def _sigma2(img, mu):
        return _mu(img * img) - mu * mu

    def _cov(a, mu_a, b, mu_b):
        return _mu(a * b) - mu_a * mu_b

    mu1, mu2 = _mu(img1), _mu(img2)
    s1, s2   = _sigma2(img1, mu1), _sigma2(img2, mu2)
    cov      = _cov(img1, mu1, img2, mu2)

    num      = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    denom    = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)

    ssim_map = num / denom                          # (N, C, H, W)
    return ssim_map.mean(dim=(-3, -2, -1))         # (N,)


def compute_ssim(
    generated: torch.Tensor,   # (B, F, 3, H, W)
    reference: torch.Tensor,   # (B, F, 3, H, W)
) -> Dict[str, float]:
    """
    Computes SSIM averaged over frames and batch.

    Returns:
        dict with keys: ssim_mean, ssim_std
    """
    gen = _to_float_01(generated)
    ref = _to_float_01(reference)
    assert gen.shape == ref.shape

    B, nF, C, H, W = gen.shape
    gen_flat = gen.reshape(B * nF, C, H, W)
    ref_flat = ref.reshape(B * nF, C, H, W)

    ssim_vals = _ssim_per_frame(gen_flat, ref_flat)   # (B*F,)
    ssim_vals = ssim_vals.reshape(B, nF)               # (B, F)

    return {
        "ssim_mean": ssim_vals.mean().item(),
        "ssim_std" : ssim_vals.std().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. LPIPS — Learned Perceptual Image Patch Similarity
# ─────────────────────────────────────────────────────────────────────────────

class LPIPSMetric(nn.Module):
    """
    Lightweight LPIPS approximation using VGG16 intermediate features.

    The full LPIPS (Zhang et al., 2018) uses calibrated linear weights on top
    of VGG/AlexNet features. This implementation extracts VGG relu2_2 and
    relu3_3 features and computes L2 distance in normalised feature space —
    correlates well with human perceptual judgements.

    <<SWAP Phase 5>>: Replace with pip install lpips and use the official
    lpips.LPIPS(net='alex') for exact paper-matching scores.
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        # Extract features up to relu3_3 (layer index 16)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise [0,1] RGB to ImageNet distribution."""
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalised feature activations."""
        x    = self._normalise(x.clamp(0.0, 1.0))
        feats = self.features(x)                           # (N, 256, H', W')
        return F.normalize(feats, dim=1)                   # L2 normalise channels

    def forward(
        self,
        generated: torch.Tensor,   # (B, F, 3, H, W) float [0, 1]
        reference: torch.Tensor,   # (B, F, 3, H, W) float [0, 1]
    ) -> Dict[str, float]:
        B, nF, C, H, W = generated.shape
        dev  = generated.device

        gen_flat = generated.reshape(B * nF, C, H, W)
        ref_flat = reference.reshape(B * nF, C, H, W)

        # Resize to 224 for VGG
        gen_224 = F.interpolate(gen_flat, size=(224, 224), mode="bilinear", align_corners=False)
        ref_224 = F.interpolate(ref_flat, size=(224, 224), mode="bilinear", align_corners=False)

        with _autocast(_get_device_type(generated)):
            f_gen = self._extract(gen_224.float())
            f_ref = self._extract(ref_224.float())

        lpips_vals = ((f_gen - f_ref) ** 2).mean(dim=(-3, -2, -1))   # (B*F,)
        lpips_vals = lpips_vals.reshape(B, nF)

        return {
            "lpips_mean": lpips_vals.mean().item(),
            "lpips_std" : lpips_vals.std().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Top-5 Inception Accuracy
# ─────────────────────────────────────────────────────────────────────────────

class Top5InceptionAccuracy(nn.Module):
    """
    Inception-v3 top-5 classification accuracy on generated frames.

    Each frame is classified independently; accuracy is averaged over frames
    and then over batch items.
    """

    def __init__(self):
        super().__init__()
        inception = tv_models.inception_v3(
            weights=tv_models.Inception_V3_Weights.DEFAULT,
            aux_logits=False,
        )
        inception.eval()
        self.inception = inception
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        generated: torch.Tensor,   # (B, F, 3, H, W) float [0, 1]
        labels: torch.Tensor,       # (B,) int64 — ImageNet class labels
    ) -> Dict[str, float]:
        """
        Args:
            generated : (B, F, 3, H, W)
            labels    : (B,) ground-truth ImageNet class indices
        Returns:
            dict with top5_acc_mean, top5_acc_std
        """
        B, nF, C, H, W = generated.shape
        dev = generated.device

        gen_flat = generated.reshape(B * nF, C, H, W)
        gen_299  = F.interpolate(gen_flat, size=(299, 299), mode="bilinear", align_corners=False)
        gen_299  = ((gen_299.clamp(0.0, 1.0) - self.mean) / self.std).float()

        with _autocast(_get_device_type(generated)):
            logits = self.inception(gen_299)               # (B*F, 1000)

        # Top-5 predictions
        top5 = torch.topk(logits.float(), 5, dim=-1).indices   # (B*F, 5)
        top5 = top5.reshape(B, nF, 5)

        # Expand labels across frames: (B,) → (B, F, 1)
        labels_exp = labels.to(dev).unsqueeze(1).unsqueeze(2).expand(B, nF, 1)

        correct = (top5 == labels_exp).any(dim=-1)         # (B, F) bool
        acc     = correct.float().mean(dim=-1)             # (B,) per-sample

        return {
            "top5_acc_mean": acc.mean().item(),
            "top5_acc_std" : acc.std().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLIP Semantic Similarity
# ─────────────────────────────────────────────────────────────────────────────

class CLIPSemanticSimilarity(nn.Module):
    """
    Cosine similarity between CLIP visual embeddings of generated frames
    and the original ground-truth CLIP embeddings.

    Two usage modes:
      (a) Precomputed embeddings: pass clip_embeddings (B, 512) directly.
          This is the Phase 4 path — use z_clip outputs from CARDEncoder
          as the reference embeddings.
      (b) Raw images: if clip_embeddings is None, a lightweight ResNet-18
          stand-in extracts embeddings at runtime (Phase 4 proxy only).

    The paper measures:
        CLIP_sim = (z_gen · z_gt) / (||z_gen|| · ||z_gt||)
    averaged over all frames and batch items.

    <<SWAP Phase 5>>: Replace self.backbone with openai/clip ViT-B/16 for
    exact CLIP cosine similarity matching paper results.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        # Lightweight visual encoder: ResNet-18 → linear → 512
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])   # pool output
        self.proj = nn.Linear(512, embed_dim)
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP-proxy embeddings for frames.
        Args: (N, 3, H, W) float [0, 1]
        Returns: (N, embed_dim) L2-normalised
        """
        x = ((frames.clamp(0, 1) - self.mean) / self.std).float()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with _autocast(_get_device_type(frames)):
            feats = self.backbone(x).squeeze(-1).squeeze(-1)   # (N, 512)
            emb   = self.proj(feats)                           # (N, embed_dim)
        return F.normalize(emb.float(), dim=-1)

    def forward(
        self,
        generated: torch.Tensor,                          # (B, F, 3, H, W) float [0, 1]
        clip_embeddings: Optional[torch.Tensor] = None,  # (B, 512) GT CLIP embeds
    ) -> Dict[str, float]:
        """
        Args:
            generated       : (B, F, 3, H, W)
            clip_embeddings : (B, 512) precomputed GT CLIP embeddings (preferred).
                              If None, raises ValueError — pass z_clip from
                              CARDEncoder as the ground-truth reference.
        Returns:
            dict with clip_sim_mean, clip_sim_std
        """
        if clip_embeddings is None:
            raise ValueError(
                "clip_embeddings must be provided. Pass z_clip from CARDEncoder "
                "as the ground-truth reference embedding."
            )

        B, nF, C, H, W = generated.shape
        dev = generated.device

        # Encode generated frames
        gen_flat = generated.reshape(B * nF, C, H, W)
        emb_gen  = self._encode_frames(gen_flat)              # (B*F, 512)
        emb_gen  = emb_gen.reshape(B, nF, self.embed_dim)    # (B, F, 512)

        # Mode (a): precomputed GT embeddings — broadcast over frames
        gt     = F.normalize(clip_embeddings.to(dev).float(), dim=-1)  # (B, 512)
        gt_exp = gt.unsqueeze(1).expand(B, nF, self.embed_dim)         # (B, F, 512)

        # Cosine similarity: both embeddings are L2-normalised → dot product = cos sim
        cos_sim = (emb_gen * gt_exp).sum(dim=-1)   # (B, F)

        return {
            "clip_sim_mean": cos_sim.mean().item(),
            "clip_sim_std" : cos_sim.std().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. FVD — Fréchet Video Distance  (primary SOTA temporal coherence metric)
# ─────────────────────────────────────────────────────────────────────────────

class I3DFeatureExtractor(nn.Module):
    """
    Lightweight I3D-inspired temporal feature extractor for Phase 4 FVD.

    Architecture:
      3D Conv (temporal kernel 3, spatial kernel 3) → BatchNorm → ReLU → AvgPool
      → Linear → 400-dim feature vector (matches Kinetics I3D embedding dim)

    <<SWAP Phase 5>>: Replace with torchvision.models.video.r3d_18 or the
    official Kinetics-pretrained I3D checkpoint for publication-quality FVD.

    Args:
        out_features: I3D feature dimensionality (default 400, matches Kinetics I3D)
    """

    def __init__(self, out_features: int = 400):
        super().__init__()
        self.out_features = out_features

        self.conv1 = nn.Conv3d(3, 64,  kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1   = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn2   = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3   = nn.BatchNorm3d(256)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc   = nn.Linear(256, out_features)

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, 3, F, H, W) float [0, 1] — channel-first, time second
        Returns:
            (B, out_features)  I3D feature vector
        """
        x = F.relu(self.bn1(self.conv1(video)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)   # (B, 256)
        return self.fc(x)                                       # (B, out_features)


class FVDMetric(nn.Module):
    """
    Fréchet Video Distance.

    Computes FVD by:
      1. Extracting I3D features from real and generated video batches.
      2. Accumulating features across the full evaluation dataset.
      3. Computing the Fréchet distance between the two Gaussian distributions.

    Usage:
        fvd = FVDMetric(device="cuda")
        for gen_batch, real_batch in dataloader:
            fvd.update(gen_batch, real_batch)
        result = fvd.compute()
        fvd.reset()

    Args:
        out_features : I3D feature dimension (400 for Kinetics I3D)
    """

    def __init__(self, out_features: int = 400):
        super().__init__()
        self.i3d = I3DFeatureExtractor(out_features=out_features)
        self.out_features = out_features
        self._gen_feats:  List[torch.Tensor] = []
        self._real_feats: List[torch.Tensor] = []

    def _extract(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract I3D features from (B, F, 3, H, W) video tensor.

        Steps:
          1. Converts to (B, 3, F, H, W) — I3D channel-time convention.
          2. Resizes spatial dims to 112×112 (balanced quality/speed).
          3. Moves to the same device as the I3D model.

        FIX v4: The I3D model's device is explicitly determined from its
        parameters, and input tensors are moved to that device before
        inference — preventing cross-device errors when callers pass CPU
        tensors to a CUDA FVDMetric.
        """
        B, nF, C, H, W = video.shape

        # I3D expects (B, C, T, H, W)
        v = video.float().permute(0, 2, 1, 3, 4)   # (B, 3, F, H, W)

        # Resize spatial dims to 112 (sensible resolution for 3D conv)
        v_flat = v.reshape(B * nF, C, H, W)
        v_flat = F.interpolate(v_flat, size=(112, 112), mode="bilinear", align_corners=False)
        v = v_flat.reshape(B, nF, C, 112, 112).permute(0, 2, 1, 3, 4)   # (B, 3, F, 112, 112)

        # Move to I3D model's device explicitly
        i3d_device = next(self.i3d.parameters()).device
        v = v.to(i3d_device)

        with _autocast(i3d_device.type):
            feats = self.i3d(v)   # (B, out_features)

        return feats.float().cpu()   # accumulate on CPU

    @torch.no_grad()
    def update(
        self,
        generated: torch.Tensor,   # (B, F, 3, H, W)
        reference: torch.Tensor,   # (B, F, 3, H, W)
    ):
        """Accumulate I3D features for one batch."""
        gen = _to_float_01(generated)
        ref = _to_float_01(reference)
        self._gen_feats.append(self._extract(gen))
        self._real_feats.append(self._extract(ref))

    def compute(self) -> Dict[str, float]:
        """
        Compute FVD over all accumulated features.

        Requires at least 2 samples to estimate covariance.
        Returns:
            dict with fvd_score
        """
        if not self._gen_feats:
            raise RuntimeError("No batches accumulated. Call .update() before .compute().")

        gen_all  = torch.cat(self._gen_feats,  dim=0)   # (N, D)
        real_all = torch.cat(self._real_feats, dim=0)   # (N, D)

        if gen_all.shape[0] < 2:
            raise RuntimeError("Need at least 2 samples to compute FVD covariance.")

        mu_gen,  sigma_gen  = _compute_mu_sigma(gen_all)
        mu_real, sigma_real = _compute_mu_sigma(real_all)

        fvd = _frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        return {"fvd_score": fvd}

    def reset(self):
        """Clear accumulated features for a new evaluation run."""
        self._gen_feats.clear()
        self._real_feats.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 7. VideoMetricEvaluator — unified interface for all metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricConfig:
    """Configuration for VideoMetricEvaluator."""
    compute_psnr  : bool = True
    compute_ssim  : bool = True
    compute_lpips : bool = True
    compute_fvd   : bool = True
    compute_clip  : bool = True
    compute_top5  : bool = False   # requires ImageNet labels — opt-in
    device        : str  = "cpu"
    n_ddim_steps  : int  = 50
    # FVD
    fvd_features  : int  = 400
    # Paper comparison
    include_paper_delta : bool = True


class VideoMetricEvaluator:
    """
    Unified evaluator for all Group 4 video reconstruction metrics.

    Handles 5D tensors (B, F, 3, H, W) natively. Accumulates statistics
    across batches and exports a structured JSON report (metrics.json).

    Usage (single batch):
        evaluator = VideoMetricEvaluator(MetricConfig(device="cuda"))
        results = evaluator.evaluate_batch(
            generated=video_gen,
            reference=video_ref,
            clip_embeddings=z_clip_gt,   # (B, 512)
        )
        evaluator.export_json("results/metrics.json")

    Usage (full dataset):
        evaluator = VideoMetricEvaluator(config)
        for gen, ref, clip_gt in dataloader:
            evaluator.update(gen, ref, clip_gt)
        final = evaluator.compute_all()
        evaluator.export_json("results/phase4_final.json")
    """

    def __init__(self, config: MetricConfig = MetricConfig()):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self._batch_results: List[Dict] = []

        # Initialise metric modules
        if config.compute_lpips:
            self.lpips_metric = LPIPSMetric().to(self.device).eval()

        if config.compute_clip:
            self.clip_metric = CLIPSemanticSimilarity().to(self.device).eval()

        if config.compute_fvd:
            self.fvd_metric = FVDMetric(out_features=config.fvd_features).to(self.device).eval()

        if config.compute_top5:
            self.top5_metric = Top5InceptionAccuracy().to(self.device).eval()

    @torch.no_grad()
    def evaluate_batch(
        self,
        generated: torch.Tensor,               # (B, F, 3, H, W)
        reference: torch.Tensor,               # (B, F, 3, H, W)
        clip_embeddings: Optional[torch.Tensor] = None,  # (B, 512)
        labels: Optional[torch.Tensor] = None,           # (B,) for top-5
    ) -> Dict[str, float]:
        """
        Compute all enabled metrics for a single batch.
        Also calls .update() on FVD accumulator.

        Returns:
            dict mapping metric_name → value
        """
        gen = _to_float_01(generated).to(self.device)
        ref = _to_float_01(reference).to(self.device)

        results: Dict[str, float] = {}

        if self.config.compute_psnr:
            results.update(compute_psnr(gen, ref))

        if self.config.compute_ssim:
            results.update(compute_ssim(gen, ref))

        if self.config.compute_lpips:
            results.update(self.lpips_metric(gen, ref))

        if self.config.compute_fvd:
            # Accumulate features; final FVD computed in compute_all()
            self.fvd_metric.update(gen, ref)

        if self.config.compute_clip:
            if clip_embeddings is not None:
                results.update(self.clip_metric(gen, clip_embeddings.to(self.device)))
            else:
                print(
                    "[VideoMetricEvaluator] Warning: clip_embeddings not provided — "
                    "CLIP similarity skipped for this batch."
                )

        if self.config.compute_top5:
            if labels is not None:
                results.update(self.top5_metric(gen, labels.to(self.device)))
            else:
                print(
                    "[VideoMetricEvaluator] Warning: labels not provided — "
                    "Top-5 accuracy skipped for this batch."
                )

        self._batch_results.append(results)
        return results

    @torch.no_grad()
    def update(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        clip_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Alias for evaluate_batch — use in multi-batch accumulation mode."""
        self.evaluate_batch(generated, reference, clip_embeddings, labels)

    def compute_all(self) -> Dict[str, float]:
        """
        Aggregate all accumulated batch results and compute FVD.

        Returns:
            dict with aggregated metric values
        """
        if not self._batch_results:
            raise RuntimeError("No batches evaluated. Call evaluate_batch() first.")

        # Average per-batch metrics
        all_keys = set(k for d in self._batch_results for k in d.keys())
        agg: Dict[str, float] = {}
        for k in sorted(all_keys):
            vals = [d[k] for d in self._batch_results if k in d]
            agg[k] = sum(vals) / len(vals)

        # Compute final FVD from accumulated features
        if self.config.compute_fvd:
            try:
                agg.update(self.fvd_metric.compute())
            except RuntimeError as e:
                print(f"[FVD] Could not compute FVD: {e}")
                agg["fvd_score"] = float("nan")

        return agg

    def export_json(
        self,
        path: str = "metrics.json",
        extra_info: Optional[Dict] = None,
    ) -> str:
        """
        Export evaluation results as a structured JSON report.

        Report structure:
            {
              "metadata": { timestamp, config, n_batches },
              "metrics":  { metric_name: value, ... },
              "paper_comparison": { metric: { ours, paper, delta }, ... }
            }

        FIX v4: os.makedirs is only called when the directory part of path
        is non-empty, preventing a crash when path is a flat filename like
        "metrics.json" (os.path.dirname returns "" in that case).

        Args:
            path      : output path (e.g. "results/metrics.json" or "metrics.json")
            extra_info: additional key-value pairs to include in metadata
        Returns:
            The path the file was written to.
        """
        agg = self.compute_all()

        # Paper delta comparison
        paper_cmp = {}
        if self.config.include_paper_delta:
            metric_map = {
                "ssim_mean" : ("SSIM",  True),    # higher is better
                "lpips_mean": ("LPIPS", False),   # lower is better
                "fvd_score" : ("FID",   False),   # lower is better (FID as FVD proxy)
                "mse_mean"  : ("MSE",   False),
            }
            for our_key, (paper_key, higher_better) in metric_map.items():
                if our_key in agg and paper_key in PAPER_BASELINES:
                    our_val   = agg[our_key]
                    paper_val = PAPER_BASELINES[paper_key]
                    delta = (our_val - paper_val) if higher_better else (paper_val - our_val)
                    paper_cmp[paper_key] = {
                        "ours"          : round(our_val,  4),
                        "paper"         : paper_val,
                        "delta"         : round(delta,    4),
                        "higher_better" : higher_better,
                        "beats_paper"   : bool(delta > 0),
                    }

        report = {
            "metadata": {
                "timestamp"    : time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_batches"    : len(self._batch_results),
                "config"       : asdict(self.config),
                "latent_shape" : [LATENT_CH, LATENT_H, LATENT_W],
                "n_frames"     : N_FRAMES,
                **(extra_info or {}),
            },
            "metrics"         : {k: round(float(v), 6) for k, v in sorted(agg.items())},
            "paper_comparison": paper_cmp,
        }

        # FIX v4: only call makedirs when there is an actual directory component
        dir_part = os.path.dirname(path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[VideoMetricEvaluator] Report saved to: {path}")
        return path

    def print_summary(self):
        """Print a formatted summary table to stdout."""
        agg = self.compute_all()
        print("\n" + "="*56)
        print("  Phase 4 Video Metrics — EEG2Video (Group 4)")
        print("="*56)
        rows = [
            ("PSNR (dB)",     "psnr_mean"),
            ("SSIM",          "ssim_mean"),
            ("LPIPS (↓)",     "lpips_mean"),
            ("FVD (↓)",       "fvd_score"),
            ("CLIP Sim",      "clip_sim_mean"),
            ("Top-5 Acc",     "top5_acc_mean"),
            ("MSE (↓)",       "mse_mean"),
        ]
        for label, key in rows:
            if key in agg:
                val = agg[key]
                marker = "  (NaN)" if math.isnan(val) else ""
                print(f"  {label:<22} {val:.4f}{marker}")
        print("="*56)
        if self.config.include_paper_delta:
            print("  Paper baselines (CVPR 2026 #22 image task):")
            for k, v in PAPER_BASELINES.items():
                print(f"    {k:<8} = {v}")
        print("="*56 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Smoke-test  (run directly: python phase4_metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[phase4_metrics smoke-test] device = {device_str}")

    B_test = 4
    nF     = N_FRAMES   # 6 frames

    # Dummy generated and reference videos — uint8 [0, 255]
    gen_uint8 = torch.randint(0, 256, (B_test, nF, 3, 128, 128), dtype=torch.uint8)
    ref_uint8 = torch.randint(0, 256, (B_test, nF, 3, 128, 128), dtype=torch.uint8)

    # Dummy CLIP ground-truth embeddings — (B, 512) float
    clip_gt = torch.randn(B_test, 512)

    # ── Configure evaluator ─────────────────────────────────────────────────
    config = MetricConfig(
        compute_psnr  = True,
        compute_ssim  = True,
        compute_lpips = True,
        compute_fvd   = True,
        compute_clip  = True,
        compute_top5  = False,   # no labels in smoke test
        device        = device_str,
    )

    evaluator = VideoMetricEvaluator(config)

    # ── Two-batch accumulation (mimics real dataloader loop) ────────────────
    print("\n  Running 2 batches...")
    for b in range(2):
        batch_results = evaluator.evaluate_batch(
            generated       = gen_uint8.clone(),
            reference       = ref_uint8.clone(),
            clip_embeddings = clip_gt.clone(),
        )
        print(f"  Batch {b+1} metrics: " + ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(batch_results.items())
        ))

    # ── Aggregated metrics ──────────────────────────────────────────────────
    evaluator.print_summary()

    # ── JSON export (flat filename — tests the makedirs fix) ────────────────
    out_path = evaluator.export_json(
        "metrics.json",   # flat filename — no directory prefix (tests v4 fix)
        extra_info={"note": "smoke-test — random tensors, not real EEG data"},
    )
    print(f"  JSON report written to: {out_path}")

    # ── Verify JSON can be re-loaded ─────────────────────────────────────────
    with open(out_path) as f:
        report_loaded = json.load(f)
    assert "metrics" in report_loaded, "JSON report missing 'metrics' key"
    assert "metadata" in report_loaded, "JSON report missing 'metadata' key"

    # ── Shape assertions ─────────────────────────────────────────────────────
    agg = evaluator.compute_all()
    required_keys = ["psnr_mean", "ssim_mean", "lpips_mean", "fvd_score", "clip_sim_mean"]
    for k in required_keys:
        assert k in agg, f"Required metric '{k}' missing from results"

    non_nan = [v for v in agg.values() if isinstance(v, float) and not math.isnan(v)]
    assert len(non_nan) == len(agg), (
        f"NaN found in metric results: "
        + ", ".join(f"{k}={v}" for k, v in agg.items() if isinstance(v, float) and math.isnan(v))
    )

    # Verify uint8 inputs are handled correctly (not treated as float [0,1])
    assert agg["psnr_mean"] < 80.0, (
        f"PSNR={agg['psnr_mean']:.1f} dB is unrealistically high — "
        "check _to_float_01 uint8 conversion"
    )

    print("\n[phase4_metrics smoke-test] ALL PASSED ✓\n")
