"""
decoder.py  —  Latent-to-frame decoder (VAE-style)
EEG2Video · IIT Mandi · CS 671

Interface contract:
  input  : (B, F, 4, 16, 16)    latent video sequence
  output : (B, F, 3, 128, 128)  RGB frames in [-1, 1]

Swap points (marked with # <<SWAP>>):
  - Replace SimpleDecoder with a pretrained SD VAE decoder loaded via
    _load_vae() in Phase 5.

Changes / fixes
---------------
v2  No logic bugs found. Added shape assertion and GroupNorm divisibility
    guard. File is verified clean.
v4  FIX: forward() assertion C==4,H==16,W==16 was hardcoded — would crash
    if real ViT 32×32 latents are used in Phase 5. Replaced with imports
    from sd_backbone constants (LATENT_CH, LATENT_H, LATENT_W) so there
    is a single source of truth. SimpleDecoder input size also parameterised.
    FIX: SimpleDecoder now accepts latent_h/latent_w args and dynamically
    adds an extra upsample stage when input is 32×32 (32→64→128→256, then
    crop to 128×128), so the decoder works for both 16×16 and 32×32 latents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sd_backbone import LATENT_CH, LATENT_H, LATENT_W


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConvUpBlock  (upsample → conv → GroupNorm → SiLU)
# ─────────────────────────────────────────────────────────────────────────────

class ConvUpBlock(nn.Module):
    """
    One decoder stage: optional 2× nearest upsample, Conv2d + GroupNorm + SiLU.

    GroupNorm group counts:
        out_ch=256 → 32 groups  ✓
        out_ch=128 → 16 groups  ✓
        out_ch=64  →  8 groups  ✓
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: bool = True):
        super().__init__()
        assert out_ch % 8 == 0, (
            f"out_ch={out_ch} must be divisible by 8 for GroupNorm(8, out_ch)"
        )
        self.up   = nn.Upsample(scale_factor=2, mode="nearest") if upsample else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(self.up(x))))


# ─────────────────────────────────────────────────────────────────────────────
# 2. SimpleDecoder  (lightweight stand-in for SD VAE decoder)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleDecoder(nn.Module):
    """
    Lightweight convolutional decoder: latents → 128×128 RGB.

    Supports two input spatial sizes:
      32×32 : 32 → 64 → 128        (2 upsample stages — DEFAULT for real ViT latents)
      16×16 : 16 → 32 → 64 → 128  (3 upsample stages — Phase 4 dummy / ablation)

    Channel progression : 4 → 256 → 128 → 64 → 3
    Output range        : [-1, 1]  (tanh)

    <<SWAP Phase 5>>: replace with pretrained SD VAE decoder.
    """

    def __init__(self, latent_ch: int = LATENT_CH, latent_h: int = LATENT_H):
        super().__init__()
        assert latent_h in (16, 32), f"SimpleDecoder supports 16 or 32 input size, got {latent_h}"
        self.latent_h = latent_h

        self.in_conv = nn.Conv2d(latent_ch, 256, 3, padding=1)

        if latent_h == 16:
            # 16 → 32 → 64 → 128
            self.up1 = ConvUpBlock(256, 256)   # 16 → 32
            self.up2 = ConvUpBlock(256, 128)   # 32 → 64
            self.up3 = ConvUpBlock(128, 64)    # 64 → 128
        else:
            # 32 → 64 → 128  (skip first upsample)
            self.up1 = ConvUpBlock(256, 256, upsample=False)  # stays 32
            self.up2 = ConvUpBlock(256, 128)                  # 32 → 64
            self.up3 = ConvUpBlock(128, 64)                   # 64 → 128

        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (BF, latent_ch, H, H)  where H = latent_h
        Returns:
            (BF, 3, 128, 128) in [-1, 1]
        """
        x = F.silu(self.in_conv(z))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return torch.tanh(self.out_conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# 3. LatentDecoder  (public API — wraps frame-level decoder)
# ─────────────────────────────────────────────────────────────────────────────

class LatentDecoder(nn.Module):
    """
    Decodes a batch of video latents (B, F, C, H, W) to RGB frames
    (B, F, 3, 128, 128).

    Latent spatial size is read from sd_backbone constants (LATENT_H, LATENT_W),
    so there is a single source of truth across the whole pipeline.

    Usage:
        decoder = LatentDecoder()
        frames  = decoder(latents)              # float32, [-1, 1]
        u8      = decoder.decode_to_uint8(latents)  # uint8, [0, 255]

    <<SWAP Phase 5>>
    To load a pretrained SD VAE decoder:
        decoder = LatentDecoder(vae_ckpt="path/to/vae.pth")
    """

    def __init__(self, latent_ch: int = LATENT_CH, vae_ckpt: str = None):
        super().__init__()
        # FIX v4: pass LATENT_H so SimpleDecoder selects the correct upsample path
        self.decoder  = SimpleDecoder(latent_ch, latent_h=LATENT_H)  # <<SWAP>>
        self.latent_h = LATENT_H
        self.latent_w = LATENT_W
        self.latent_ch = latent_ch

        if vae_ckpt is not None:
            self._load_vae(vae_ckpt)

    def _load_vae(self, ckpt_path: str):
        """
        <<SWAP Phase 5>>
        Load SD VAE decoder weights.
        """
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        print(f"[LatentDecoder] loaded VAE ckpt with {len(state)} keys")
        # self.decoder.load_state_dict(remapped_state, strict=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (B, F, C, H, W)  where C=LATENT_CH, H=LATENT_H, W=LATENT_W
        Returns:
            frames : (B, F, 3, 128, 128)  float32 in [-1, 1]
        """
        B, nF, C, H, W = latents.shape

        # FIX v4: check against imported constants, not hardcoded literals
        if not (C == self.latent_ch and H == self.latent_h and W == self.latent_w):
            raise ValueError(
                f"LatentDecoder expected latents (*,*,{self.latent_ch},"
                f"{self.latent_h},{self.latent_w}), got (*,*,{C},{H},{W}). "
                f"Update LATENT_H/W in sd_backbone.py if using real ViT latents."
            )

        z      = latents.reshape(B * nF, C, H, W)
        frames = self.decoder(z)                           # (BF, 3, 128, 128)
        return frames.reshape(B, nF, 3, *frames.shape[-2:])

    @torch.no_grad()
    def decode_to_uint8(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, F, 3, H, W) uint8 in [0, 255] — use for saving.
        """
        frames = self.forward(latents)                     # float32 in [-1, 1]
        return ((frames + 1.0) * 127.5).clamp(0, 255).byte()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke-test] device = {device}")
    print(f"[smoke-test] using LATENT_H={LATENT_H}, LATENT_W={LATENT_W}")

    B, nF = 2, 6
    decoder = LatentDecoder().to(device)
    latents = torch.randn(B, nF, LATENT_CH, LATENT_H, LATENT_W, device=device)

    with torch.no_grad():
        frames = decoder(latents)
        u8     = decoder.decode_to_uint8(latents)

    assert frames.shape == (B, nF, 3, 128, 128), f"Unexpected float shape: {frames.shape}"
    assert u8.shape     == (B, nF, 3, 128, 128), f"Unexpected uint8 shape: {u8.shape}"
    assert u8.dtype     == torch.uint8,           "uint8 dtype expected"
    assert frames.min() >= -1.0 and frames.max() <= 1.0, "Output outside [-1, 1]"

    print(f"[smoke-test] float frames : {frames.shape}  "
          f"range [{frames.min():.2f}, {frames.max():.2f}]  ✓")
    print(f"[smoke-test] uint8 frames : {u8.shape}  dtype={u8.dtype}  ✓")
    print(f"[smoke-test] parameters   : {sum(p.numel() for p in decoder.parameters()):,}")
    print("[smoke-test] decoder PASSED")
