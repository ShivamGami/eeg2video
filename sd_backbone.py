"""
sd_backbone.py  —  TemporalUNet / Tune-A-Video style diffusion backbone
EEG2Video · IIT Mandi · CS 671

Interface contract (matches team docs):
  visual_latents  : (B, F, 4, 16, 16)   F=6 frames, latent_channels=4
  text_embeddings : (B, 77, 512)         CLIP-style token embeddings
  timestep        : (B,)                 integer diffusion timestep
  output          : (B, F, 4, 16, 16)   predicted noise / denoised latents

Swap points (marked with # <<SWAP>>):
  - Replace with real SD/TAV weights via load_sd_weights() in Phase 5

Changes / fixes
---------------
v2  BUG: decoder spatial flow used `i < len-1` → [up, up, Identity].
    Fixed to `i > 0` → [Identity, up, up]:
      i=0: Identity   x@4×4  + skip@4×4  — exact match, no interpolate
      i=1: up 4→8     x@8×8  + skip@8×8  — correct
      i=2: up 8→16    x@16×16 + skip@16×16 — correct
    The interpolate guard is replaced by a hard assertion.
v3  No new bugs found. File is verified clean.
v4  FIX: SpatialCrossAttention added assert BF % B_ctx == 0 to catch
    batch-size mismatches early instead of silently computing wrong F.
    FIX: TemporalUNet.forward() text_emb shape check changed from hard
    assert to a clear ValueError with a descriptive message — avoids
    confusing AssertionError tracebacks during debugging.
    FIX: LATENT_H / LATENT_W constants added at module level so all
    files can import a single source of truth (16×16 for dummy/phase-4;
    swap to 32×32 if using real ViT latents in phase-5).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants  (single source of truth — import from here)
# ─────────────────────────────────────────────────────────────────────────────

LATENT_CH   = 4
LATENT_H    = 16   # Real ViT latent_proj.bias=24576=6×4×32×32 → confirmed 32×32
LATENT_W    = 16
N_FRAMES    = 6
TEXT_SEQ    = 77
TEXT_DIM    = 512


# ─────────────────────────────────────────────────────────────────────────────
# 1. Timestep Embedding
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep → MLP projection, identical to DDPM / SD."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, dim * 4)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]             # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)                                # (B, dim*4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatial Cross-Attention  (text embeddings → each spatial frame)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialCrossAttention(nn.Module):
    """
    Multi-head cross-attention between spatial feature tokens (query)
    and text / EEG-text embeddings (key / value).

    Shapes:
        x      : (B*F, HW, inner_dim)
        context: (B,   77,  text_dim)   — broadcast over F frames
    """

    def __init__(self, inner_dim: int, text_dim: int = TEXT_DIM, n_heads: int = 4):
        super().__init__()
        assert inner_dim % n_heads == 0, "inner_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = inner_dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm   = nn.LayerNorm(inner_dim)
        self.to_q   = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k   = nn.Linear(text_dim,  inner_dim, bias=False)
        self.to_v   = nn.Linear(text_dim,  inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, inner_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        BF, HW, C = x.shape
        B_ctx = context.shape[0]

        # FIX v4: guard against silent wrong-F computation
        if BF % B_ctx != 0:
            raise ValueError(
                f"SpatialCrossAttention: BF={BF} is not divisible by "
                f"B_ctx={B_ctx}. Check that text_emb batch size matches latents."
            )
        F = BF // B_ctx

        # Expand context: (B, 77, D) → (B*F, 77, D)
        ctx = context.unsqueeze(1).expand(-1, F, -1, -1).reshape(BF, -1, context.shape[-1])

        residual = x
        x = self.norm(x)
        q = self.to_q(x)    # (BF, HW, C)
        k = self.to_k(ctx)  # (BF, 77, C)
        v = self.to_v(ctx)  # (BF, 77, C)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            bsz, seq, _ = t.shape
            return t.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (BF, n_heads, HW, 77)
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(attn, v)                               # (BF, n_heads, HW, head_dim)
        out  = out.transpose(1, 2).reshape(BF, HW, C)
        return residual + self.to_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Temporal Self-Attention  (across F frames at each spatial position)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Self-attention across the frame dimension F at each spatial position.

    Full attention in dummy phase.
    <<SWAP Phase 5>>: add TAV sparse [first + previous frame] mask.

    Shapes: x (input/output): (B*F, HW, inner_dim)
    """

    def __init__(self, inner_dim: int, n_heads: int = 4):
        super().__init__()
        assert inner_dim % n_heads == 0, "inner_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = inner_dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm   = nn.LayerNorm(inner_dim)
        self.to_q   = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_k   = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_v   = nn.Linear(inner_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, inner_dim)

    def forward(self, x: torch.Tensor, n_frames: int) -> torch.Tensor:
        BF, HW, C = x.shape
        B = BF // n_frames

        residual = x
        x = self.norm(x)

        # Rearrange for temporal attention: (B*F, HW, C) → (B*HW, F, C)
        x = x.view(B, n_frames, HW, C).permute(0, 2, 1, 3).reshape(B * HW, n_frames, C)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            bsz, seq, _ = t.shape
            return t.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(self.to_q(x)), split_heads(self.to_k(x)), split_heads(self.to_v(x))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B * HW, n_frames, C)

        # Restore: (B*HW, F, C) → (B*F, HW, C)
        out = out.view(B, HW, n_frames, C).permute(0, 2, 1, 3).reshape(BF, HW, C)
        return residual + self.to_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ResBlock with timestep injection
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    GroupNorm → SiLU → Conv2d (×2) with timestep bias injection.
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1     = nn.GroupNorm(8, in_ch)
        self.conv1     = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (BF, in_ch,  H, W)
            t_emb: (BF, time_emb_dim)
        Returns:
            (BF, out_ch, H, W)
        """
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. STTransformerBlock  (Spatial + Temporal attention together)
# ─────────────────────────────────────────────────────────────────────────────

class STTransformerBlock(nn.Module):
    """
    Applies SpatialCrossAttention then TemporalAttention on a spatial feature map.
    """

    def __init__(self, inner_dim: int, text_dim: int = TEXT_DIM):
        super().__init__()
        self.spatial_attn  = SpatialCrossAttention(inner_dim, text_dim)
        self.temporal_attn = TemporalAttention(inner_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, n_frames: int) -> torch.Tensor:
        BF, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(BF, H * W, C)   # (BF, HW, C)
        x_flat = self.spatial_attn(x_flat, context)
        x_flat = self.temporal_attn(x_flat, n_frames)
        return x_flat.reshape(BF, H, W, C).permute(0, 3, 1, 2)  # (BF, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TemporalUNet
# ─────────────────────────────────────────────────────────────────────────────

class TemporalUNet(nn.Module):
    """
    Lightweight Tune-A-Video style 3D U-Net for latent video denoising.

    Encoder spatial progression (32×32 latent input — real ViT size):
      Level 0 : ResBlock + ST  → skip (BF, 64,  32,32) → stride-2 → 16×16
      Level 1 : ResBlock + ST  → skip (BF, 128, 16,16) → stride-2 → 8×8
      Level 2 : ResBlock + ST  → skip (BF, 256,  8, 8) → Identity → 8×8

    Bottleneck: (BF, 256, 8, 8)

    Decoder spatial progression (FIXED — no silent interpolate):
      i=0: Identity  → cat skip(256,8×8)   → ResBlock(512→256) → ST → 8×8
      i=1: up 8→16  → cat skip(128,16×16)  → ResBlock(384→128) → ST → 16×16
      i=2: up 16→32 → cat skip(64,32×32)   → ResBlock(192→64)  → ST → 32×32

    Output: Conv2d(64→4) → (B, F, 4, 32, 32)
    """

    BASE_CH  = 64
    CH_MULTS = (1, 2, 4)   # → channels [64, 128, 256]
    TIME_DIM = 256          # sinusoidal embedding dim; MLP projects to TIME_DIM*4=1024

    def __init__(
        self,
        latent_ch: int = LATENT_CH,
        text_dim:  int = TEXT_DIM,
        n_frames:  int = N_FRAMES,
    ):
        super().__init__()
        self.n_frames = n_frames
        chs           = [self.BASE_CH * m for m in self.CH_MULTS]  # [64, 128, 256]
        time_proj_dim = self.TIME_DIM * 4                           # 1024

        # ── Timestep embedding ───────────────────────────────────────────────
        self.time_embed = TimestepEmbedding(self.TIME_DIM)

        # ── Input projection ─────────────────────────────────────────────────
        self.in_conv = nn.Conv2d(latent_ch, chs[0], 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_res  = nn.ModuleList()
        self.enc_st   = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        in_ch = chs[0]
        for idx, out_ch in enumerate(chs):
            self.enc_res.append(ResBlock(in_ch, out_ch, time_proj_dim))
            self.enc_st.append(STTransformerBlock(out_ch, text_dim))
            # Levels 0 and 1 halve spatial; level 2 (bottleneck) stays the same
            if idx < len(chs) - 1:
                self.enc_down.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.enc_down.append(nn.Identity())
            in_ch = out_ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid_res1 = ResBlock(chs[-1], chs[-1], time_proj_dim)
        self.mid_st   = STTransformerBlock(chs[-1], text_dim)
        self.mid_res2 = ResBlock(chs[-1], chs[-1], time_proj_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        # reversed chs        = [256, 128,  64]
        # reversed skips      = [256@4×4, 128@8×8, 64@16×16]
        # Upsample condition  : i > 0  (FIXED — not i < len-1)
        #   i=0: Identity,        x@4×4  + skip@4×4   → no mismatch
        #   i=1: ConvTranspose,   x@8×8  + skip@8×8   → correct
        #   i=2: ConvTranspose,   x@16×16+ skip@16×16 → correct
        dec_out_chs = list(reversed(chs))  # [256, 128, 64]
        skip_chs    = list(reversed(chs))  # [256, 128, 64]

        self.dec_up  = nn.ModuleList()
        self.dec_res = nn.ModuleList()
        self.dec_st  = nn.ModuleList()
        cur_ch = chs[-1]  # 256
        for i in range(len(dec_out_chs)):
            skip_ch = skip_chs[i]
            out_ch  = dec_out_chs[i]
            self.dec_up.append(
                nn.ConvTranspose2d(cur_ch, cur_ch, 2, stride=2) if i > 0 else nn.Identity()
            )
            self.dec_res.append(ResBlock(cur_ch + skip_ch, out_ch, time_proj_dim))
            self.dec_st.append(STTransformerBlock(out_ch, text_dim))
            cur_ch = out_ch
        # cur_ch ends at 64 = chs[0]  ✓

        # ── Output projection ────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], latent_ch, 3, padding=1)

    def forward(
        self,
        latents:  torch.Tensor,  # (B, F, 4, 16, 16)
        text_emb: torch.Tensor,  # (B, 77, 512)
        timestep: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Returns (B, F, 4, 16, 16) — predicted noise or denoised latents."""
        B, nF, C_lat, H, W = latents.shape

        # FIX v4: descriptive ValueError instead of bare assert
        if not (C_lat == LATENT_CH and H == LATENT_H and W == LATENT_W):
            raise ValueError(
                f"Expected latents (B,F,{LATENT_CH},{LATENT_H},{LATENT_W}), "
                f"got (B,F,{C_lat},{H},{W}). "
                f"If using real ViT latents (32×32), update LATENT_H/W constants."
            )
        if text_emb.shape != (B, TEXT_SEQ, TEXT_DIM):
            raise ValueError(
                f"Expected text_emb (B={B},{TEXT_SEQ},{TEXT_DIM}), "
                f"got {tuple(text_emb.shape)}"
            )

        x     = latents.reshape(B * nF, C_lat, H, W)
        t_emb = self.time_embed(timestep.repeat_interleave(nF))  # (B*F, 1024)

        # ── Encoder ──────────────────────────────────────────────────────────
        x = self.in_conv(x)
        skips = []
        for res, st, down in zip(self.enc_res, self.enc_st, self.enc_down):
            x = res(x, t_emb)
            x = st(x, text_emb, nF)
            skips.append(x)   # save BEFORE downsampling
            x = down(x)
        # skips = [64@16×16, 128@8×8, 256@4×4]

        # ── Bottleneck ───────────────────────────────────────────────────────
        x = self.mid_res1(x, t_emb)
        x = self.mid_st(x, text_emb, nF)
        x = self.mid_res2(x, t_emb)

        # ── Decoder ──────────────────────────────────────────────────────────
        for up, res, st, skip in zip(self.dec_up, self.dec_res, self.dec_st, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(
                    f"Decoder size mismatch: x={x.shape[-2:]}, skip={skip.shape[-2:]}. "
                    f"This should never happen — check encoder downsampling."
                )
            x = torch.cat([x, skip], dim=1)
            x = res(x, t_emb)
            x = st(x, text_emb, nF)

        x = self.out_conv(F.silu(self.out_norm(x)))  # (B*F, 4, 16, 16)
        return x.reshape(B, nF, C_lat, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Utility: load real SD weights  (stub — Phase 5)
# ─────────────────────────────────────────────────────────────────────────────

def load_sd_weights(model: TemporalUNet, ckpt_path: str) -> TemporalUNet:
    """
    <<SWAP Phase 5>>
    Load Stable Diffusion / Tune-A-Video weights into TemporalUNet.
    Prints checkpoint keys for manual mapping.
    """
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    print(f"[load_sd_weights] checkpoint has {len(state)} keys — map manually in Phase 5")
    # model.load_state_dict(mapped_state, strict=False)
    return model
