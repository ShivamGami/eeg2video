import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import os

# ─────────────────────────────────────────────
# 1. HYBRID CONFIGURATION
# ─────────────────────────────────────────────
# This path only exists on the DSLAB server
SERVER_PATH = "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/EEG/"
IS_SERVER = os.path.exists(SERVER_PATH)

# Device & Protocol Logic
if IS_SERVER:
    device = torch.device("cuda")
    # PROTOCOL RULE: VRAM fencing for shared server
    torch.cuda.set_per_process_memory_fraction(0.2, device=0)
    print("🚀 Running on SERVER (GPU Mode)")
else:
    device = torch.device("cpu")
    print("💻 Running on MAC (Local CPU Mode)")

# ─────────────────────────────────────────────
# 2. W&B INITIALIZATION
# ─────────────────────────────────────────────
wandb.init(
    project="eeg2video-cs671",
    group="Sub-team 3: Vision Transformer",
    name="manan_test_run",
    config={
        "learning_rate": 0.001,
        "epochs": 5,           # Keep small for local testing
        "batch_size": 2,       # Small batch to avoid OOM
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "eeg_chan": 62,
        "eeg_time": 100,       # Points after windowing
        "vis_frames": 6
    }
)

# ─────────────────────────────────────────────
# 3. HYBRID DATA LOADER
# ─────────────────────────────────────────────
class SEEDDVDataset(Dataset):
    def __init__(self, root_dir, target_len=100):
        self.root_dir = root_dir
        self.target_len = target_len
        if IS_SERVER:
            self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        else:
            self.file_list = ["local_test_1", "local_test_2"] # Fake list for Mac

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if IS_SERVER:
            # REAL DATA LOADING
            file_path = os.path.join(self.root_dir, self.file_list[idx])
            data = np.load(file_path, allow_pickle=True)
            # Slice middle 100 points from 104,000
            start = (data.shape[-1] // 2) - (self.target_len // 2)
            sliced_data = data[:, :, start:start + self.target_len]
            x = torch.from_numpy(sliced_data).float()
        else:
            # LOCAL DUMMY DATA
            # Shape: (Segments, Channels, Time) -> (7, 62, 100)
            x = torch.randn(7, 62, 100)
        
        # Target visual latents (Phase 1 dummy)
        # Shape: (Frames, Channels, H, W) -> (6, 4, 32, 32)
        y = torch.randn(6, 4, 32, 32)
        return x, y

# ─────────────────────────────────────────────
# 4. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class EEGToLatentTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Flattened EEG input: 62 channels * 100 time points = 6200
        self.embed = nn.Linear(cfg.eeg_chan * cfg.eeg_time, cfg.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        
        # Project tokens to Visual Latent space (4*32*32 = 4096)
        self.latent_proj = nn.Linear(cfg.d_model, 4 * 32 * 32)

    def forward(self, x):
        # x: (B, 7, 62, 100) -> (B, 7, 6200)
        B, S, C, T = x.shape
        x = x.view(B, S, -1)
        
        tokens = self.embed(x)             # (B, 7, 256)
        feat = self.transformer(tokens)    # (B, 7, 256)
        
        # Predict 6 video frames from the first 6 EEG segments
        out = self.latent_proj(feat[:, :6, :]) # (B, 6, 4096)
        return out.view(B, 6, 4, 32, 32)

# ─────────────────────────────────────────────
# 5. TEST RUN EXECUTION
# ─────────────────────────────────────────────
dataset = SEEDDVDataset(SERVER_PATH if IS_SERVER else "")
loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

model = EEGToLatentTransformer(wandb.config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion = nn.MSELoss()

print(f"✅ Setup complete. Starting {wandb.config.epochs} epochs...")

for epoch in range(wandb.config.epochs):
    model.train()
    for batch_idx, (eeg, target) in enumerate(loader):
        eeg, target = eeg.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(eeg)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 5 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item()})

print("🎉 Test run finished successfully!")
wandb.finish()
