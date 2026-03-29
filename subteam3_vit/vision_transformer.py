import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import os

# ─────────────────────────────────────────────
# 1. SERVER PATHS & CONFIG
# ─────────────────────────────────────────────
EEG_DIR = "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/EEG/"
LATENT_DIR = "/home/teaching/vishal_workspace/eeg2video-cs671/processed_features/"

# Check if we are on server
IS_SERVER = os.path.exists(EEG_DIR)
device = torch.device("cuda" if (IS_SERVER and torch.cuda.is_available()) else "cpu")

if IS_SERVER:
    torch.cuda.set_per_process_memory_fraction(0.2, device=0)
    print("🚀 PHASE 2: REAL DATA TRAINING STARTING...")

# ─────────────────────────────────────────────
# 2. W&B INITIALIZATION
# ─────────────────────────────────────────────
wandb.init(
    project="eeg2video-cs671",
    group="Sub-team 3: Vision Transformer",
    name="manan_real_data_run_01",
    config={
        "learning_rate": 0.0001, # Lowered for real data
        "epochs": 50,
        "batch_size": 1,         # Start small as real latents are large
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,         # Slightly deeper for real data
        "eeg_chan": 62,
        "eeg_time": 100,
        "vis_frames": 6
    }
)

# ─────────────────────────────────────────────
# 3. REAL DATA LOADER (EEG + VISHAL'S LATENTS)
# ─────────────────────────────────────────────
class SEEDDVRealDataset(Dataset):
    def __init__(self, eeg_dir, latent_dir, target_len=100):
        self.eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith('.npy')]
        self.eeg_dir = eeg_dir
        self.latent_dir = latent_dir
        self.target_len = target_len
        # The 7 latent files provided by Sub-team 2
        self.latent_filenames = [
            "1st_10min_latents.pt", "2nd_10min_latents.pt", "3rd_10min_latents.pt",
            "4th_10min_latents.pt", "5th_10min_latents.pt", "6th_10min_latents.pt",
            "7th_10min_latents.pt"
        ]

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        # 1. Load EEG Subject File (Shape: 7, 62, 104000)
        eeg_path = os.path.join(self.eeg_dir, self.eeg_files[idx])
        eeg_data = np.load(eeg_path, allow_pickle=True) # (7, 62, 104000)
        
        all_x = []
        all_y = []
        
        # 2. Process all 7 segments for this subject
        for i in range(7):
            # EEG Slicing (same as Phase 1)
            start = (eeg_data.shape[-1] // 2) - (self.target_len // 2)
            eeg_slice = eeg_data[i, :, start:start+self.target_len]
            all_x.append(torch.from_numpy(eeg_slice).float())
            
            # 3. Load Corresponding Real Latent from Vishal's work
            latent_path = os.path.join(self.latent_dir, self.latent_filenames[i])
            real_latents = torch.load(latent_path) # Shape: [Num_Clips, 6, 4, 32, 32]
            
            # For now, we take the first clip from the latent file 
            # (Matches the single slice we take from EEG)
            target_latent = real_latents[0] # (6, 4, 32, 32)
            all_y.append(target_latent)

        return torch.stack(all_x), torch.stack(all_y)

# ─────────────────────────────────────────────
# 4. MODEL (Unchanged Architecture)
# ─────────────────────────────────────────────
class EEGToLatentTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(cfg.eeg_chan * cfg.eeg_time, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.latent_proj = nn.Linear(cfg.d_model,6 * 4 * 32 * 32)

    def forward(self, x):
        # x: (B, 7, 62, 100)
        B, S, C, T = x.shape
        x = x.view(B * S, -1)              # Treat segments as part of batch for tokenization
        tokens = self.embed(x)             # (B*7, d_model)
        feat = self.transformer(tokens.unsqueeze(0)).squeeze(0)
        out = self.latent_proj(feat)       # (B*7, 4096)
        return out.view(B, S, 6, 4, 32, 32) # (B, 7 segments, 6 frames, ...)

# ─────────────────────────────────────────────
# 5. EXECUTION
# ─────────────────────────────────────────────
dataset = SEEDDVRealDataset(EEG_DIR, LATENT_DIR)
loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

model = EEGToLatentTransformer(wandb.config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion = nn.MSELoss()

for epoch in range(wandb.config.epochs):
    model.train()
    for eeg, latents in loader:
        eeg, latents = eeg.to(device), latents.to(device)
        
        optimizer.zero_grad()
        output = model(eeg)
        loss = criterion(output, latents)
        loss.backward()
        optimizer.step()
        
        wandb.log({"real_data_loss": loss.item()})
    
    print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "subteam3_vit/checkpoints/vit_real_data.pth")
wandb.finish()
