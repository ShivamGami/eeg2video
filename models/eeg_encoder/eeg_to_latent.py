import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# CONSTANTS (Stable Diffusion latent shape)
# ---------------------------------------------------------------------------
LATENT_FRAMES    = 6
LATENT_CHANNELS  = 4
LATENT_H         = 32
LATENT_W         = 32
LATENT_FLAT_DIM  = LATENT_FRAMES * LATENT_CHANNELS * LATENT_H * LATENT_W  # 24576

TRANSFORMER_DIM  = 512
TRANSFORMER_TOKENS = 700


#---------------------------------------------------------------------------
# PROJECTION HEAD
# ---------------------------------------------------------------------------
class LatentProjectionHead(nn.Module):
    """
    Converts transformer output (B, 700, 512)
    → (B, 6, 4, 32, 32) using temporal grouping
    """

    def __init__(self, in_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 4 * 32 * 32),  # 4096 per frame
        )

    def forward(self, x):
        """
        x: (B, 700, 512)
        """

        B, T, D = x.shape  # (B, 700, 512)

        # --- Step 1: Ensure divisible by 6 ---
        tokens_per_frame = T // 6           # 116
        usable_tokens = tokens_per_frame * 6  # 696

        x = x[:, :usable_tokens, :]         # (B, 696, 512)

        # --- Step 2: Split into 6 temporal chunks ---
        x = x.view(B, 6, tokens_per_frame, D)  # (B, 6, 116, 512)

        # --- Step 3: Pool within each chunk ---
        x = x.mean(dim=2)                   # (B, 6, 512)

        # --- Step 4: Project each frame ---
        x = self.net(x)                     # (B, 6, 4096)

        # --- Step 5: Reshape to latent format ---
        x = x.view(B, 6, 4, 32, 32)

        return x       

# ---------------------------------------------------------------------------
# FULL MODEL
# ---------------------------------------------------------------------------
class EEGToLatentModel(nn.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.projection = LatentProjectionHead()

    def forward(self, eeg):
        # ✅ EVERYTHING inside must be indented

        assert eeg.shape[1:] == (7, eeg.shape[2], 100), "EEG input shape mismatch"

        tokens = self.transformer(eeg)
        assert tokens.shape[1:] == (700, 512), "Transformer output mismatch"

        latents = self.projection(tokens)
        assert latents.shape[1:] == (6, 4, 32, 32), "Latent output mismatch"

        return latents
