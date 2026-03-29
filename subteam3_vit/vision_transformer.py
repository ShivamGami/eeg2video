# vision_transformer.py
# Sub-team 3 | EEG2Video CS671
# Task: Map EEG signals → VAE visual latents

import torch
import torch.nn as nn
import wandb

# ─────────────────────────────────────────────
# PROTOCOL RULE: Limit VRAM when testing
# Comment this out only for full training runs
# ─────────────────────────────────────────────
torch.cuda.set_per_process_memory_fraction(0.2, device=0)

# ─────────────────────────────────────────────
# W&B Initialization (exactly as per protocol)
# ─────────────────────────────────────────────
wandb.init(
    project="eeg2video-cs671",
    group="Sub-team 3: Vision Transformer",
    name="your_name_run_01",           # change to your name
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 16,
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
    }
)

# ─────────────────────────────────────────────
# Interface Contract Shapes (from Phase 1 doc)
# ─────────────────────────────────────────────
BATCH       = 4         # use small batch locally
EEG_SEGS    = 7         # time segments
EEG_CHAN    = 62        # EEG channels (SEED-DV)
EEG_TIME    = 100       # time points per segment
VIS_FRAMES  = 6         # video frames
VIS_CHAN    = 4         # VAE latent channels
VIS_H       = 32        # latent height
VIS_W       = 32        # latent width

# ─────────────────────────────────────────────
# Dummy Data (use this until Phase 1 data ready)
# Shape: (Batch, 7, Channels, 100)
# ─────────────────────────────────────────────
eeg_dummy = torch.randn(BATCH, EEG_SEGS, EEG_CHAN, EEG_TIME)
target_latents = torch.randn(BATCH, VIS_FRAMES, VIS_CHAN, VIS_H, VIS_W)

print(f"EEG input shape:      {eeg_dummy.shape}")
print(f"Target latent shape:  {target_latents.shape}")


# ─────────────────────────────────────────────
# Model: EEG → Visual Latent Transformer
# ─────────────────────────────────────────────
class EEGPatchEmbedding(nn.Module):
    """Flatten each EEG segment into a token."""
    def __init__(self, eeg_chan, eeg_time, d_model):
        super().__init__()
        self.proj = nn.Linear(eeg_chan * eeg_time, d_model)

    def forward(self, x):
        # x: (B, 7, C, T) → (B, 7, d_model)
        B, S, C, T = x.shape
        x = x.reshape(B, S, C * T)
        return self.proj(x)


class EEGToLatentTransformer(nn.Module):
    """
    Vanilla Transformer: maps EEG tokens → visual latent tokens.
    Input:  (B, 7, EEG_CHAN, EEG_TIME)
    Output: (B, 6, VIS_CHAN, VIS_H, VIS_W)
    """
    def __init__(self, eeg_chan, eeg_time, d_model=256,
                 num_heads=8, num_layers=4,
                 vis_frames=6, vis_chan=4, vis_h=32, vis_w=32):
        super().__init__()

        self.embed = EEGPatchEmbedding(eeg_chan, eeg_time, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True       # (B, S, D) convention
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Project each output token to one video frame latent
        self.vis_frames  = vis_frames
        self.vis_chan    = vis_chan
        self.vis_h       = vis_h
        self.vis_w       = vis_w
        latent_dim = vis_chan * vis_h * vis_w

        self.frame_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        # x: (B, 7, EEG_CHAN, EEG_TIME)
        tokens = self.embed(x)                      # (B, 7, d_model)
        tokens = self.transformer(tokens)            # (B, 7, d_model)

        # Use first 6 tokens → one per video frame
        frame_tokens = tokens[:, :self.vis_frames]   # (B, 6, d_model)
        latents = self.frame_proj(frame_tokens)      # (B, 6, C*H*W)

        B = latents.shape[0]
        latents = latents.reshape(
            B, self.vis_frames, self.vis_chan, self.vis_h, self.vis_w
        )
        return latents


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EEGToLatentTransformer(
    eeg_chan=EEG_CHAN,
    eeg_time=EEG_TIME,
    d_model=wandb.config.d_model,
    num_heads=wandb.config.num_heads,
    num_layers=wandb.config.num_layers,
    vis_frames=VIS_FRAMES,
    vis_chan=VIS_CHAN,
    vis_h=VIS_H,
    vis_w=VIS_W,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion = nn.MSELoss()

eeg_dummy     = eeg_dummy.to(device)
target_latents = target_latents.to(device)

for epoch in range(wandb.config.epochs):
    model.train()
    optimizer.zero_grad()

    pred_latents = model(eeg_dummy)             # (B, 6, 4, 32, 32)
    loss = criterion(pred_latents, target_latents)

    loss.backward()
    optimizer.step()

    # ── W&B Logging (required by protocol) ──
    wandb.log({"train_loss": loss.item(), "epoch": epoch})

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# Save weights — locally only, never push to GitHub
torch.save(model.state_dict(), "vision_transformer.pth")
print("Model saved to vision_transformer.pth")
wandb.finish()
