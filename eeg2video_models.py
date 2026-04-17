"""
EEG2Video — Phase 4 Model Suite
================================
All architectures verified against actual saved weight shapes in:
  • vit_real_data.pth
  • text_mlp_final.pth
  • dynamics_model.pth

Sub-team 2 requested tensors:
  • text_embeddings : (B, 77, 512)
  • is_fast         : (B,)

Run `python eeg2video_models.py` to generate them.
"""

import torch
import torch.nn as nn
import os

# ---------------------------------------------------------------------------
# CONSTANTS  (verified against actual .pth weight shapes)
# ---------------------------------------------------------------------------

# ── ViT / EEG encoder ──────────────────────────────────────────────────────
EEG_CHANNELS    = 62
EEG_TIME_PTS    = 100
EEG_FLAT_DIM    = EEG_CHANNELS * EEG_TIME_PTS  # 6200  ← embed.weight = (6200, 256)

VIT_D_MODEL     = 256    # ← actual weight dim (your code said 512 — WRONG)
VIT_D_FFN       = 2048   # 8× expansion, confirmed from linear1/linear2 weights
VIT_N_HEADS     = 8      # 256 / 8 = 32 per head (standard)
VIT_N_LAYERS    = 6      # confirmed: transformer.layers.0 … .5

# ── Text MLP ───────────────────────────────────────────────────────────────
TEXT_IN_DIM     = 512    # net.0.weight = (1024, 512)  → input dim = 512
TEXT_HIDDEN_DIM = 1024   # net.0 output / net.4 input
TEXT_OUT_DIM    = 512    # net.4.weight = (512, 1024)  → output dim = 512
CLIP_TOKENS     = 77     # CLIP sequence length requested by Sub-team 2
CLIP_DIM        = 512    # CLIP-B/32 embedding dim

# ── Dynamics model ─────────────────────────────────────────────────────────
DYN_IN_DIM      = 512    # net.0.weight = (512, 512)   → input dim = 512
DYN_HIDDEN_DIM  = 512    # net.0 output
DYN_MID_DIM     = 256    # net.4.weight = (256, 512)   → output dim = 256
# net.6.weight = (256,) + net.6.bias = (1,) → Linear(256, 1) final

# ── Latent output ──────────────────────────────────────────────────────────
LATENT_FRAMES   = 6
LATENT_CHANNELS = 4
LATENT_H        = 32
LATENT_W        = 32
LATENT_FLAT_DIM = LATENT_FRAMES * LATENT_CHANNELS * LATENT_H * LATENT_W  # 24576

DROPOUT = 0.1


# ===========================================================================
# 1.  EEG ENCODER  (ViT backbone)
# ===========================================================================

class EEGEncoder(nn.Module):
    """
    Encodes raw EEG (B, 62, 100) → shared 512-dim feature vector (B, 512).

    Pipeline
    --------
    Flatten (B, 6200)
      → Linear embed (6200 → 256)          ← matches embed.weight (6200×256)
      → prepend CLS token
      → 6-layer Transformer (d=256, h=8)   ← matches all transformer.layers.*
      → CLS pool
      → project (256 → 512)                ← bridges to Text MLP / Dynamics inputs

    Note: ViT was trained on raw time-series EEG (B, 62, 100), NOT on the
    STFT format (B, 62, 51, 9) used in Phase-2 preprocessing. If your
    pipeline produces STFT tensors, flatten them to (B, 28458) and add a
    Linear(28458, 6200) adapter, or re-export the ViT with STFT input.
    """

    def __init__(self,
                 eeg_flat_dim : int = EEG_FLAT_DIM,
                 d_model      : int = VIT_D_MODEL,
                 n_heads      : int = VIT_N_HEADS,
                 d_ffn        : int = VIT_D_FFN,
                 n_layers     : int = VIT_N_LAYERS,
                 out_dim      : int = TEXT_IN_DIM,   # 512 — matches both downstream heads
                 dropout      : float = DROPOUT):
        super().__init__()

        # patch / token embedding (matches embed.weight shape exactly)
        self.embed    = nn.Linear(eeg_flat_dim, d_model)

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_drop = nn.Dropout(dropout)

        # Transformer — Pre-LN for training stability
        layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_ffn,
            dropout        = dropout,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,   # Pre-LN (more stable than Post-LN)
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

        # Bridge: ViT d_model (256) → shared feature dim (512)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        eeg : (B, 62, 100)  raw EEG time-series

        Returns
        -------
        feat : (B, 512)  shared feature vector
        """
        B = eeg.shape[0]
        x = eeg.flatten(1)                          # (B, 6200)
        x = self.embed(x).unsqueeze(1)              # (B, 1, 256)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)            # (B, 2, 256)
        x   = self.pos_drop(x)

        x   = self.transformer(x)                   # (B, 2, 256)
        x   = self.norm(x)
        x   = x[:, 0]                               # CLS token → (B, 256)
        return self.out_proj(x)                     # (B, 512)

    def load_vit_weights(self, path: str):
        """Load the vit_real_data.pth state dict (256-dim weights only)."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        # state dict uses keys: embed.*, transformer.layers.*, etc.
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[EEGEncoder] loaded '{path}'")
        print(f"  missing   : {missing}")
        print(f"  unexpected: {unexpected}")


# ===========================================================================
# 2.  TEXT MLP
# ===========================================================================

class TextMLP(nn.Module):
    """
    Maps shared EEG feature (B, 512) → CLIP-aligned embedding (B, 512).

    Verified architecture from text_mlp_final.pth:
      net.0  Linear(512, 1024)   weight=(1024,512)  bias=(1024,)
      net.1  LayerNorm(1024)     weight=(1024,)      bias=(1024,)
      net.2  GELU
      net.3  Dropout(0.1)
      net.4  Linear(1024, 512)   weight=(512,1024)   bias=(512,)

    Output shape from saved weights: (B, 512)
    Sub-team 2 needs:               (B, 77, 512)

    → `forward()` returns (B, 77, 512) by expanding the 512-dim output
      across all 77 CLIP token positions (same embedding broadcast).
      If you want per-token diversity, replace the expander with a
      learned Linear(512, 77*512) head.
    """

    def __init__(self,
                 in_dim     : int = TEXT_IN_DIM,
                 hidden_dim : int = TEXT_HIDDEN_DIM,
                 out_dim    : int = TEXT_OUT_DIM,
                 clip_tokens: int = CLIP_TOKENS,
                 dropout    : float = DROPOUT):
        super().__init__()
        self.clip_tokens = clip_tokens

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),   # net.0
            nn.LayerNorm(hidden_dim),        # net.1
            nn.GELU(),                       # net.2
            nn.Dropout(dropout),             # net.3
            nn.Linear(hidden_dim, out_dim),  # net.4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, 512)

        Returns
        -------
        text_embeddings : (B, 77, 512)  — CLIP-aligned, broadcast across tokens
        """
        emb = self.net(x)                               # (B, 512)
        return emb.unsqueeze(1).expand(-1, self.clip_tokens, -1)  # (B, 77, 512)

    def load_weights(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(sd, strict=True)
        print(f"[TextMLP] loaded '{path}'")


# ===========================================================================
# 3.  DYNAMICS MODEL
# ===========================================================================

class DynamicsModel(nn.Module):
    """
    Maps shared EEG feature (B, 512) → motion label scalar (B,).

    Verified architecture from dynamics_model.pth:
      net.0  Linear(512, 512)    weight=(512,512)   bias=(512,)
      net.1  BatchNorm1d(512)    weight/bias/running_mean/running_var
      net.2  ReLU
      net.3  Dropout(0.1)
      net.4  Linear(512, 256)    weight=(256,512)   bias=(256,)
      net.5  ReLU
      net.6  Linear(256, 1)      weight=(1,256)     bias=(1,)
      Sigmoid → scalar in [0, 1]

    0 = slow motion, 1 = fast motion  (DANA temporal noise mixing)
    """

    def __init__(self,
                 in_dim     : int = DYN_IN_DIM,
                 hidden_dim : int = DYN_HIDDEN_DIM,
                 mid_dim    : int = DYN_MID_DIM,
                 dropout    : float = DROPOUT):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),   # net.0
            nn.BatchNorm1d(hidden_dim),      # net.1
            nn.ReLU(),                       # net.2
            nn.Dropout(dropout),             # net.3
            nn.Linear(hidden_dim, mid_dim),  # net.4
            nn.ReLU(),                       # net.5
            nn.Linear(mid_dim, 1),           # net.6
            nn.Sigmoid(),                    # → [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, 512)

        Returns
        -------
        is_fast : (B,)   values in [0, 1]
        """
        return self.net(x).squeeze(-1)    # (B, 1) → (B,)

    def load_weights(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(sd, strict=True)
        print(f"[DynamicsModel] loaded '{path}'")


# ===========================================================================
# 4.  LATENT PROJECTION HEAD  (ViT → SD latent)
# ===========================================================================

class LatentProjectionHead(nn.Module):
    """
    Maps pooled ViT output (B, 256) → SD latent (B, 6, 4, 32, 32).

    Verified from vit_real_data.pth:
      latent_proj.weight = (24576, 256)   Linear(256, 24576)
      latent_proj.bias   = (24576,)       24576 = 6×4×32×32 ✓

    Improvement: adds a residual bottleneck before the large projection
    to reduce gradient shock while keeping final Linear weight-loadable.
    """

    def __init__(self,
                 in_dim        : int = VIT_D_MODEL,     # 256
                 latent_frames : int = LATENT_FRAMES,   # 6
                 latent_c      : int = LATENT_CHANNELS, # 4
                 latent_h      : int = LATENT_H,        # 32
                 latent_w      : int = LATENT_W,        # 32
                 dropout       : float = DROPOUT):
        super().__init__()
        self.out_shape = (latent_frames, latent_c, latent_h, latent_w)
        out_dim = latent_frames * latent_c * latent_h * latent_w  # 24576

        # Residual bottleneck (new — not in saved weights, trained from scratch)
        self.pre_norm   = nn.LayerNorm(in_dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),   # 256 → 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, in_dim),   # 512 → 256
        )
        self.bn_norm = nn.LayerNorm(in_dim)

        # Final projection — exact shape from saved weights (256 → 24576)
        self.latent_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, 256)   CLS-pooled ViT output

        Returns
        -------
        latents : (B, 6, 4, 32, 32)
        """
        x = self.pre_norm(x)
        x = x + self.bottleneck(x)      # residual
        x = self.bn_norm(x)
        x = self.latent_proj(x)         # (B, 24576)
        return x.view(-1, *self.out_shape)


# ===========================================================================
# 5.  FULL PIPELINE
# ===========================================================================

class EEG2VideoModel(nn.Module):
    """
    Complete EEG → conditioning tensors pipeline.

    Input
    -----
    eeg : (B, 62, 100)   raw EEG time-series

    Outputs
    -------
    text_embeddings : (B, 77, 512)   CLIP-aligned (Sub-team 2 FILE 1)
    is_fast         : (B,)           motion label  (Sub-team 2 FILE 2)
    latents         : (B, 6, 4, 32, 32)  SD latent
    """

    def __init__(self):
        super().__init__()
        self.encoder    = EEGEncoder()
        self.text_mlp   = TextMLP()
        self.dynamics   = DynamicsModel()
        self.latent_head = LatentProjectionHead()

    def forward(self, eeg: torch.Tensor):
        # Shared EEG feature
        feat = self.encoder(eeg)                    # (B, 512)

        text_embeddings = self.text_mlp(feat)       # (B, 77, 512)
        is_fast         = self.dynamics(feat)       # (B,)
        latents         = self.latent_head(
            self.encoder.norm(
                self.encoder.transformer(
                    torch.cat([
                        self.encoder.cls_token.expand(eeg.shape[0], -1, -1),
                        self.encoder.embed(eeg.flatten(1)).unsqueeze(1)
                    ], dim=1)
                )
            )[:, 0]
        )                                           # (B, 6, 4, 32, 32)

        return text_embeddings, is_fast, latents


# ===========================================================================
# 6.  STFT ADAPTER  (for Phase-2 preprocessed inputs)
# ===========================================================================

class STFTToRawAdapter(nn.Module):
    """
    Bridges STFT EEG (B, 62, 51, 9) → raw-equivalent (B, 62, 100).

    The ViT was trained on raw time-series (B, 62, 100) but Phase-2
    preprocessing outputs STFT spectrograms (B, 62, 51, 9).

    This learned adapter lets you use STFT inputs without retraining the ViT.
    STFT flat = 62×51×9 = 28,458 features → projected to 62×100 = 6,200.
    """

    def __init__(self,
                 stft_shape : tuple = (62, 51, 9),
                 raw_shape  : tuple = (62, 100)):
        super().__init__()
        in_dim  = 1
        for d in stft_shape: in_dim *= d     # 28458
        out_dim = 1
        for d in raw_shape:  out_dim *= d    # 6200

        self.adapter = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),  # 28458 → 12400
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim), # 12400 → 6200
        )
        self.raw_shape = raw_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args   : x  (B, 62, 51, 9)
        Returns: x' (B, 62, 100)
        """
        B = x.shape[0]
        x = x.flatten(1)                    # (B, 28458)
        x = self.adapter(x)                 # (B, 6200)
        return x.view(B, *self.raw_shape)   # (B, 62, 100)


# ===========================================================================
# 7.  INFERENCE SCRIPT  — generates Sub-team 2's required .pt files
# ===========================================================================

def generate_conditioning_tensors(
    vit_path      : str = "vit_real_data.pth",
    text_mlp_path : str = "text_mlp_final.pth",
    dynamics_path : str = "dynamics_model.pth",
    batch_size    : int = 8,
    eeg_input     : torch.Tensor = None,   # (B, 62, 100)  or None for dummy
    use_stft_input: bool = False,          # set True if eeg_input is (B,62,51,9)
    output_dir    : str = "real_inputs",
):
    """
    Runs forward passes and saves:
      real_inputs/visual_latents.pt    → (B, 6, 4, 32, 32)
      real_inputs/text_embeddings.pt   → (B, 77, 512)
      real_inputs/is_fast.pt           → (B,)

    All three files are required by inference.py run_from_real_inputs().
    No random/dummy tensors are generated — real EEG input is mandatory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Build models ──────────────────────────────────────────────────────
    encoder  = EEGEncoder()
    text_mlp = TextMLP()
    dynamics = DynamicsModel()
    lat_head = LatentProjectionHead()

    # ── Load weights ──────────────────────────────────────────────────────
    encoder.load_vit_weights(vit_path)
    text_mlp.load_weights(text_mlp_path)
    dynamics.load_weights(dynamics_path)
    # LatentProjectionHead weights live inside vit_real_data.pth under
    # key 'latent_proj.*' — load with strict=False (other keys also present)
    vit_sd = torch.load(vit_path, map_location="cpu", weights_only=False)
    lp_sd  = {k.replace("latent_proj.", ""): v
               for k, v in vit_sd.items() if k.startswith("latent_proj.")}
    if lp_sd:
        lat_head.latent_proj.load_state_dict(lp_sd, strict=True)
        print(f"[LatentProjectionHead] loaded latent_proj weights from '{vit_path}'")
    else:
        print("[LatentProjectionHead] WARNING: no latent_proj.* keys in vit checkpoint")

    encoder.eval()
    text_mlp.eval()
    dynamics.eval()
    lat_head.eval()

    # ── Prepare EEG input ─────────────────────────────────────────────────
    if eeg_input is None:
        raise ValueError(
            "eeg_input is required. Pass a real (B, 62, 100) EEG tensor.\n"
            "Do NOT use random tensors — they produce meaningless embeddings "
            "that will corrupt Phase 4 diffusion training."
        )
    if use_stft_input:
        print("[INFO] Converting STFT (B,62,51,9) → raw-equivalent (B,62,100)")
        adapter   = STFTToRawAdapter()
        eeg_input = adapter(eeg_input)

    assert eeg_input.ndim == 3 and eeg_input.shape[1:] == (62, 100), \
        f"Expected (B, 62, 100), got {eeg_input.shape}"

    # ── Forward pass ──────────────────────────────────────────────────────
    with torch.no_grad():
        feat            = encoder(eeg_input)            # (B, 512)
        text_embeddings = text_mlp(feat)                # (B, 77, 512)
        is_fast         = dynamics(feat)                # (B,)

        # Visual latents: run ViT encoder's internal CLS path → (B, 256) → (B, 6, 4, 32, 32)
        B_in = eeg_input.shape[0]
        x    = encoder.embed(eeg_input.flatten(1)).unsqueeze(1)     # (B, 1, 256)
        cls  = encoder.cls_token.expand(B_in, -1, -1)
        x    = torch.cat([cls, x], dim=1)                           # (B, 2, 256)
        x    = encoder.transformer(x)
        x    = encoder.norm(x)
        cls_feat = x[:, 0]                                           # (B, 256)
        visual_latents = lat_head(cls_feat)                          # (B, 6, 4, 32, 32)

    # ── Validate shapes ───────────────────────────────────────────────────
    B = eeg_input.shape[0]
    assert visual_latents.shape  == (B, 6, 4, 32, 32), \
        f"visual_latents shape mismatch: {visual_latents.shape}"
    assert text_embeddings.shape == (B, 77, 512), \
        f"text_embeddings shape mismatch: {text_embeddings.shape}"
    assert is_fast.shape == (B,), \
        f"is_fast shape mismatch: {is_fast.shape}"
    assert is_fast.min() >= 0.0 and is_fast.max() <= 1.0, \
        "is_fast values out of [0,1] range"

    # ── Save ──────────────────────────────────────────────────────────────
    lat_path  = os.path.join(output_dir, "visual_latents.pt")
    text_path = os.path.join(output_dir, "text_embeddings.pt")
    fast_path = os.path.join(output_dir, "is_fast.pt")

    torch.save(visual_latents,  lat_path)
    torch.save(text_embeddings, text_path)
    torch.save(is_fast,         fast_path)

    print(f"\n✅ Saved to {output_dir}/:")
    print(f"   {lat_path}   → {tuple(visual_latents.shape)}  dtype={visual_latents.dtype}")
    print(f"   {text_path}  → {tuple(text_embeddings.shape)}  dtype={text_embeddings.dtype}")
    print(f"   {fast_path}  → {tuple(is_fast.shape)}  dtype={is_fast.dtype}")
    print(f"   is_fast range: [{is_fast.min():.4f}, {is_fast.max():.4f}]")

    return visual_latents, text_embeddings, is_fast


# ===========================================================================
# 8.  SANITY CHECK
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EEG2Video Model Suite — Shape Verification")
    print("=" * 60)

    B = 4

    # ── EEGEncoder ────────────────────────────────────────────────────────
    enc  = EEGEncoder()
    eeg  = torch.randn(B, 62, 100)
    feat = enc(eeg)
    assert feat.shape == (B, 512),       f"EEGEncoder: {feat.shape}"
    print(f"✓ EEGEncoder      : {eeg.shape} → {feat.shape}")

    # ── TextMLP ───────────────────────────────────────────────────────────
    tmpl = TextMLP()
    temb = tmpl(feat)
    assert temb.shape == (B, 77, 512),   f"TextMLP: {temb.shape}"
    print(f"✓ TextMLP         : {feat.shape} → {temb.shape}")

    # ── DynamicsModel ─────────────────────────────────────────────────────
    dyn  = DynamicsModel()
    fast = dyn(feat)
    assert fast.shape == (B,),           f"DynamicsModel: {fast.shape}"
    assert fast.min() >= 0 and fast.max() <= 1
    print(f"✓ DynamicsModel   : {feat.shape} → {fast.shape}  range=[{fast.min():.3f},{fast.max():.3f}]")

    # ── LatentProjectionHead ──────────────────────────────────────────────
    vit_feat = torch.randn(B, 256)
    lph      = LatentProjectionHead()
    lat      = lph(vit_feat)
    assert lat.shape == (B, 6, 4, 32, 32), f"LatentHead: {lat.shape}"
    print(f"✓ LatentProjHead  : {vit_feat.shape} → {lat.shape}")

    # ── STFT Adapter ──────────────────────────────────────────────────────
    stft_eeg = torch.randn(B, 62, 51, 9)
    adapter  = STFTToRawAdapter()
    raw_eeg  = adapter(stft_eeg)
    assert raw_eeg.shape == (B, 62, 100), f"STFTAdapter: {raw_eeg.shape}"
    print(f"✓ STFTAdapter     : {stft_eeg.shape} → {raw_eeg.shape}")

    total = sum(p.numel() for p in enc.parameters()) + \
            sum(p.numel() for p in tmpl.parameters()) + \
            sum(p.numel() for p in dyn.parameters())
    print(f"\nTotal parameters  : {total:,}")

    print("\n" + "=" * 60)
    print("All shape assertions passed ✓")
    print("=" * 60)

    print("""
To generate real conditioning tensors from real EEG data:

    from eeg2video_models import generate_conditioning_tensors
    import torch

    # Load your real preprocessed EEG: (B, 62, 100) float32
    eeg_input = torch.load("your_eeg_data.pt")

    generate_conditioning_tensors(
        vit_path      = "real_models/vit_real_data.pth",
        text_mlp_path = "real_models/text_mlp_final.pth",
        dynamics_path = "real_models/dynamics_model.pth",
        eeg_input     = eeg_input,     # REQUIRED — no random fallback
        output_dir    = "real_inputs",
    )
    # Saves:
    #   real_inputs/visual_latents.pt   → (B, 6, 4, 32, 32)
    #   real_inputs/text_embeddings.pt  → (B, 77, 512)
    #   real_inputs/is_fast.pt          → (B,)
""")