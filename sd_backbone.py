"""
sd_backbone.py
==============
Sub-team 2 – Generative Backbone | Phase 1 Architecture
CS 671 EEG2Video Reproduction | Team 22

Stable Diffusion fine-tuning backbone (Tune-A-Video style) that accepts:
  - Visual latents   : (Batch, 6, 4, 32, 32)  from Sub-team 3 (or VAE)
  - Text embeddings  : (Batch, 77, 768)        from Sub-team 4 (or SD Text Encoder)

DEPENDENCIES: zero third-party deps beyond what is already on the server.
  - einops has been deliberately removed; all tensor reshaping uses
    pure PyTorch (reshape / permute / contiguous) so this runs on the
    server pip freeze as-is.

Phase 1 goal : verify full forward pass on CPU with dummy tensors.
Phase 2+     : pass --mode train --use_gpu on the server.

Usage (local Windows CPU test):
    conda activate eeg2video_env
    python sd_backbone.py --mode phase1_check

Server (Phase 2+):
    tmux new -s yourname_backbone
    python sd_backbone.py --mode train --use_gpu --use_wandb
"""

import sys
import os
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL

import torchvision.utils as vutils
import torch
# ─────────────────────────────────────────────────────────────────────────────
#  Environment Safeguard – MUST be eeg2video_env (server & local)
# ─────────────────────────────────────────────────────────────────────────────
_REQUIRED_ENV = "eeg2video_env"
_active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
if _active_env != _REQUIRED_ENV:
    print("\n" + "!"*65)
    print(f"  [BLOCKED] Wrong conda environment detected.")
    print(f"  Active  : '{_active_env or 'none'}'")
    print(f"  Required: '{_REQUIRED_ENV}'")
    print(f"\n  Fix: conda activate {_REQUIRED_ENV}")
    print("!"*65 + "\n")
    sys.exit(1)

import argparse
import torch
import torch.nn as nn

# wandb is optional – only imported when --use_wandb flag is set
# (avoids hard crash on machines where wandb login is not configured)


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-PyTorch rearrange helpers  (replaces einops entirely)
# ─────────────────────────────────────────────────────────────────────────────

def bt_to_b_t(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """(B*T, C, H, W) -> (B, T, C, H, W)"""
    BT, C, H, W = x.shape
    return x.reshape(B, T, C, H, W)

def b_t_to_bt(x: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) -> (B*T, C, H, W)"""
    B, T, C, H, W = x.shape
    return x.reshape(B * T, C, H, W)

def bhw_t_c_to_b_t_c_h_w(x: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    """(B*H*W, T, C) -> (B, T, C, H, W)"""
    BHW, T, C = x.shape
    # x is (B*H*W, T, C) — we need (B, T, C, H, W)
    x = x.reshape(B, H, W, T, C)           # (B, H, W, T, C)
    x = x.permute(0, 3, 4, 1, 2)           # (B, T, C, H, W)
    return x.contiguous()

def b_t_c_h_w_to_bhw_t_c(x: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) -> (B*H*W, T, C)"""
    B, T, C, H, W = x.shape
    x = x.permute(0, 3, 4, 1, 2)           # (B, H, W, T, C)
    return x.reshape(B * H * W, T, C).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    # Interface contract shapes (Phase 1 – do not change)
    BATCH_SIZE      = 2
    NUM_FRAMES      = 6      # 2-sec clip at 3 FPS
    LATENT_C        = 4      # SD VAE latent channels
    LATENT_H        = 32     # spatial height of latent
    LATENT_W        = 32     # spatial width  of latent
    TEXT_SEQ        = 77     # CLIP token sequence length
    TEXT_DIM        = 768    # CLIP text embedding dim (SD 1.x)

    # UNet architecture
    MODEL_CHANNELS  = 128    # base channel count (small for CPU testing)
    DROPOUT         = 0.1

    # Training (Phase 2+)
    LEARNING_RATE   = 1e-4
    EPOCHS          = 50
    DIFFUSION_STEPS = 1000

    # W&B
    PROJECT  = "eeg2video-cs671"
    GROUP    = "Sub-team 2: Generative Backbone"
    RUN_NAME = "subteam2_run_01"   # CHANGE to your name before pushing


# ─────────────────────────────────────────────────────────────────────────────
#  Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Encodes diffusion timestep t -> fixed-dim sinusoidal embedding.
    Tells the UNet how noisy the input latent currently is.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps : (B,)  integer indices in [0, DIFFUSION_STEPS)
        Returns:
            emb       : (B, dim*4)
        """
        half  = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=timesteps.device).float()
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)           # (B, dim)
        return self.proj(emb)                                         # (B, dim*4)


class CrossAttention(nn.Module):
    """
    Cross-attention: spatial latent tokens (Q) attend to text tokens (K, V).
    Core text-conditioning mechanism — text guides what gets denoised.
    """
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = query_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.to_q   = nn.Linear(query_dim,  query_dim,  bias=False)
        self.to_k   = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v   = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, N, query_dim)    spatial tokens
            context : (B, 77, context_dim) text embeddings
        Returns:
                    : (B, N, query_dim)
        """
        B, N, C = x.shape
        H = self.num_heads

        q = self.to_q(x      ).reshape(B, N,  H, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, H, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, H, self.head_dim).transpose(1, 2)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class TemporalAttention(nn.Module):
    """
    Self-attention across the TIME axis (6 video frames).
    Operates independently per spatial position — the Tune-A-Video key idea.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=Config.DROPOUT,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C, H, W)
        Returns:
            x : (B, T, C, H, W)  — same shape, temporally attended
        """
        B, T, C, H, W = x.shape
        x_flat   = b_t_c_h_w_to_bhw_t_c(x)          # (B*H*W, T, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out + x_flat)       # residual + norm
        return bhw_t_c_to_b_t_c_h_w(attn_out, B, H, W)  # (B, T, C, H, W)


class ResBlock(nn.Module):
    """ResNet block conditioned on timestep embedding."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1     = nn.GroupNorm(8, in_channels)
        self.conv1     = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2     = nn.GroupNorm(8, out_channels)
        self.conv2     = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.act       = nn.SiLU()
        self.skip      = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal UNet  (Tune-A-Video style)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalUNet(nn.Module):
    """
    Denoising UNet for video latents.

    Inputs:
        noisy_latents   : (B, 6, 4, 32, 32)   noisy video latents at timestep t
        timesteps       : (B,)                 diffusion timestep indices
        text_embeddings : (B, 77, 768)         text conditioning from CLIP

    Output:
        noise_pred      : (B, 6, 4, 32, 32)   predicted noise at each frame

    Architecture (3-level encoder–decoder):
        Encoder  32x32 -> 16x16 -> 8x8
        Bottleneck 8x8
        Decoder  8x8  -> 16x16 -> 32x32
        At every level: ResBlock + CrossAttention (text) + TemporalAttention (frames)
    """

    def __init__(self, cfg: Config = None):
        super().__init__()
        cfg = cfg or Config()
        C         = cfg.MODEL_CHANNELS
        T_dim     = cfg.TEXT_DIM
        t_emb_dim = C * 4

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(C)

        # Input projection: LATENT_C(4) -> C
        self.input_proj = nn.Conv2d(cfg.LATENT_C, C, 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        # Level 0: spatial 32x32
        self.enc0_res   = ResBlock(C,   C,   t_emb_dim)
        self.enc0_xattn = CrossAttention(C, T_dim)
        self.enc0_tattn = TemporalAttention(C)
        self.down0      = nn.Conv2d(C, C*2, 4, stride=2, padding=1)   # 32->16

        # Level 1: spatial 16x16
        self.enc1_res   = ResBlock(C*2, C*2, t_emb_dim)
        self.enc1_xattn = CrossAttention(C*2, T_dim)
        self.enc1_tattn = TemporalAttention(C*2)
        self.down1      = nn.Conv2d(C*2, C*4, 4, stride=2, padding=1) # 16->8

        # ── Bottleneck: spatial 8x8 ───────────────────────────────────────────
        self.mid_res    = ResBlock(C*4, C*4, t_emb_dim)
        self.mid_xattn  = CrossAttention(C*4, T_dim)
        self.mid_tattn  = TemporalAttention(C*4)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Level 1: 8->16, receives skip from enc1
        self.up1        = nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1)
        self.dec1_res   = ResBlock(C*4, C*2, t_emb_dim)   # C*4 after skip concat
        self.dec1_xattn = CrossAttention(C*2, T_dim)
        self.dec1_tattn = TemporalAttention(C*2)

        # Level 0: 16->32, receives skip from enc0
        self.up0        = nn.ConvTranspose2d(C*2, C, 4, stride=2, padding=1)
        self.dec0_res   = ResBlock(C*2, C, t_emb_dim)     # C*2 after skip concat
        self.dec0_xattn = CrossAttention(C, T_dim)
        self.dec0_tattn = TemporalAttention(C)

        # Output projection: C -> LATENT_C(4)
        self.out_norm = nn.GroupNorm(8, C)
        self.out_proj = nn.Conv2d(C, cfg.LATENT_C, 3, padding=1)

        self.cfg = cfg

    # ── Helper: apply cross-attn + temporal-attn on a (B*T, C, H, W) tensor ─
    def _apply_attn(
        self,
        x:      torch.Tensor,   # (B*T, C, H, W)
        txt:    torch.Tensor,   # (B*T, 77, 768)
        xattn:  CrossAttention,
        tattn:  TemporalAttention,
        B:      int,
        T:      int,
    ) -> torch.Tensor:
        BT, C, H, W = x.shape
        # Cross-attention (text conditioning): flatten spatial, attend, reshape back
        x_seq = x.view(BT, C, -1).transpose(1, 2)    # (B*T, H*W, C)
        x_seq = xattn(x_seq, txt)
        x = x_seq.transpose(1, 2).view(BT, C, H, W)
        # Temporal attention (across frames)
        x_vol = bt_to_b_t(x, B, T)                    # (B, T, C, H, W)
        x_vol = tattn(x_vol)
        return b_t_to_bt(x_vol)                        # (B*T, C, H, W)

    def forward(
        self,
        noisy_latents:   torch.Tensor,   # (B, 6, 4, 32, 32)
        timesteps:       torch.Tensor,   # (B,)
        text_embeddings: torch.Tensor,   # (B, 77, 768)
    ) -> torch.Tensor:
        """Returns predicted noise of shape (B, 6, 4, 32, 32)."""
        B, T, C_lat, H, W = noisy_latents.shape

        # ── Timestep embedding ───────────────────────────────────────────────
        t_emb = self.time_embed(timesteps)               # (B, C*4)

        # ── Flatten frames into batch dim for all 2-D spatial ops ───────────
        x       = b_t_to_bt(noisy_latents)               # (B*T, 4, 32, 32)

        # Broadcast t_emb and text over frames
        t_rep   = t_emb.unsqueeze(1).expand(-1, T, -1).reshape(B*T, -1)        # (B*T, C*4)
        txt_rep = text_embeddings.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, 77, -1)  # (B*T,77,768)

        # ── Input projection ─────────────────────────────────────────────────
        x = self.input_proj(x)                           # (B*T, C, 32, 32)

        # ── Encoder Level 0 ──────────────────────────────────────────────────
        x = self.enc0_res(x, t_rep)
        x = self._apply_attn(x, txt_rep, self.enc0_xattn, self.enc0_tattn, B, T)
        skip0 = x
        x = self.down0(x)                                # (B*T, 2C, 16, 16)

        # ── Encoder Level 1 ──────────────────────────────────────────────────
        x = self.enc1_res(x, t_rep)
        x = self._apply_attn(x, txt_rep, self.enc1_xattn, self.enc1_tattn, B, T)
        skip1 = x
        x = self.down1(x)                                # (B*T, 4C, 8, 8)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.mid_res(x, t_rep)
        x = self._apply_attn(x, txt_rep, self.mid_xattn, self.mid_tattn, B, T)

        # ── Decoder Level 1 ───────────────────────────────────────────────────
        x = self.up1(x)                                  # (B*T, 2C, 16, 16)
        x = torch.cat([x, skip1], dim=1)                 # (B*T, 4C, 16, 16)
        x = self.dec1_res(x, t_rep)
        x = self._apply_attn(x, txt_rep, self.dec1_xattn, self.dec1_tattn, B, T)

        # ── Decoder Level 0 ───────────────────────────────────────────────────
        x = self.up0(x)                                  # (B*T, C, 32, 32)
        x = torch.cat([x, skip0], dim=1)                 # (B*T, 2C, 32, 32)
        x = self.dec0_res(x, t_rep)
        x = self._apply_attn(x, txt_rep, self.dec0_xattn, self.dec0_tattn, B, T)

        # ── Output ────────────────────────────────────────────────────────────
        x = self.out_proj(nn.functional.silu(self.out_norm(x)))  # (B*T, 4, 32, 32)
        return bt_to_b_t(x, B, T)                                 # (B, 6, 4, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    model:           TemporalUNet,
    noisy_latents:   torch.Tensor,   # (B, 6, 4, 32, 32)
    noise_target:    torch.Tensor,   # (B, 6, 4, 32, 32)
    timesteps:       torch.Tensor,   # (B,)
    text_embeddings: torch.Tensor,   # (B, 77, 768)
) -> torch.Tensor:
    """MSE diffusion loss: ||noise_target - model(noisy, t, text)||^2"""
    noise_pred = model(noisy_latents, timesteps, text_embeddings)
    return nn.functional.mse_loss(noise_pred, noise_target)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 Sanity Check
# ─────────────────────────────────────────────────────────────────────────────


def run_phase1_check():
    """CPU-only forward pass check with dummy tensors. No GPU or W&B needed."""
    from dummy_data import get_dummy_batch, get_dummy_noisy_latents

    print("\n" + "="*65)
    print("  Sub-team 2 – Phase 1 Forward Pass Check")
    print("  Running on: CPU  |  No real data needed")
    print("="*65)

    cfg    = Config()
    device = torch.device("cpu")

    print("\n  [1/4] Building TemporalUNet...")
    model  = TemporalUNet(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"        Total parameters : {params:,}")

    print("\n  [2/4] Generating dummy tensors (server shapes)...")
    batch = get_dummy_batch(batch_size=cfg.BATCH_SIZE, device=str(device))
    noisy = get_dummy_noisy_latents(batch["visual_latents"], batch["timesteps"])
    for k, v in batch.items():
        print(f"        {k:<26} : {tuple(v.shape)}")

    print("\n  [3/4] Forward pass...")
    model.eval()
    with torch.no_grad():
        noise_pred = model(noisy, batch["timesteps"], batch["text_embeddings"])
    print(f"        Input  noisy_latents  : {tuple(noisy.shape)}")
    print(f"        Output noise_pred     : {tuple(noise_pred.shape)}")
    expected = (cfg.BATCH_SIZE, cfg.NUM_FRAMES, cfg.LATENT_C, cfg.LATENT_H, cfg.LATENT_W)
    assert tuple(noise_pred.shape) == expected, \
        f"Shape mismatch! Got {tuple(noise_pred.shape)}, expected {expected}"

    print("\n  [4/4] Loss computation...")
    model.train()
    loss = compute_loss(model, noisy, batch["noise_target"],
                        batch["timesteps"], batch["text_embeddings"])
    print(f"        Loss : {loss.item():.6f}")

    print("\n" + "="*65)
    print("  RESULT: Phase 1 PASSED.")
    print("  Output shape: (B, 6, 4, 32, 32)  matches interface contract.")
    print("  Next: commit to feature/generative-backbone, push to server.")
    print("="*65 + "\n")

def run_inference(model, vae, device):

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    model.eval()
    vae.eval()

    B, T, C, H, W = 1, 6, 4, 32, 32

    # start from pure noise
    latents = torch.randn((B, T, C, H, W)).to(device)

    # dummy text embeddings
    text_embeddings = torch.randn((B, 77, 768)).to(device)

    # 🔥 reverse diffusion
    for t in reversed(range(1000)):
        timestep = torch.tensor([t], device=device).long()

        with torch.no_grad():
            noise_pred = model(latents, timestep, text_embeddings)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    print("Inference complete ✅")

    # 🔥 decode latents → images
    latents = latents / 0.18215

    latents = latents.view(B*T, C, H, W)

    with torch.no_grad():
        images = vae.decode(latents).sample

    # reshape back
    images = images.view(B, T, 3, 256, 256)

    # normalize to [0,1]
    images = (images.clamp(-1, 1) + 1) / 2

    return images
# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG2Video – Sub-team 2 Backbone")
    parser.add_argument(
        "--mode", default="phase1_check",
        choices=["phase1_check", "train"],
        help="phase1_check = CPU dummy test (today) | train = server training (Phase 2+)",
    )
    parser.add_argument("--use_gpu",   action="store_true", help="Enable GPU (server only)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging (Phase 2+)")
    args = parser.parse_args()

    if args.mode == "phase1_check":
        run_phase1_check()



# ─────────────────────────────────────────────────────────────
# TRAIN MODE
# ─────────────────────────────────────────────────────────────

    elif args.mode == "train":

        from dummy_data import get_dummy_batch
        import torchvision.utils as vutils

        cfg = Config()

        if args.use_gpu and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.2, device=0)
            device = torch.device("cuda")
            print(f"  GPU : {torch.cuda.get_device_name(0)} (20% VRAM cap)")
        else:
            device = torch.device("cpu")
            print("  Device: CPU")

        model = TemporalUNet(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(device)
        vae.eval()

        scheduler = DDPMScheduler(num_train_timesteps=cfg.DIFFUSION_STEPS)

        print(f"\n  Training for {cfg.EPOCHS} epochs...")

        for epoch in range(cfg.EPOCHS):
            model.train()

            batch = get_dummy_batch(batch_size=cfg.BATCH_SIZE, device=str(device))

            latents = batch["visual_latents"].to(device).float()
            text    = batch["text_embeddings"].to(device)
            t       = batch["timesteps"].to(device)

            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, t)

            optimizer.zero_grad()

            noise_pred = model(noisy_latents, t, text)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()

            print(f"  Epoch [{epoch+1:>3}/{cfg.EPOCHS}]  loss: {loss.item():.6f}")

        print("\n  Training complete ✅")

        # 🔥 INFERENCE HERE (IMPORTANT)
        images = run_inference(model, vae, device)

        vutils.save_image(images[0, 0], "output_frame.png")

        print("Image saved as output_frame.png ✅")



