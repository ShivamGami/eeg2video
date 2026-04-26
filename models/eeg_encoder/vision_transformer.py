import torch
import torch.nn as nn

class STFT_FeatureExtractor(nn.Module):
    def __init__(self, in_channels=62, freq_bins=51, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=(freq_bins, 3), padding=(0, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.conv(x) 
        x = x.squeeze(2)          
        x = x.permute(0, 2, 1)    
        return x

class EEGVideoTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.extractor = STFT_FeatureExtractor(d_model=d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head A: Semantic Meaning (CLIP)
        self.clip_mlp = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512)
        )

        # Head B: Visual Latents (VQ-VAE)
        self.latent_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 6 * 4 * 16 * 16)
        )

    def forward(self, x):
        features = self.extractor(x)
        T = features.size(1)
        features = features + self.pos_encoder[:, :T, :]

        t_out = self.transformer(features)
        pooled_out = t_out.mean(dim=1)

        clip_emb = self.clip_mlp(pooled_out)
        latents = self.latent_head(pooled_out).view(-1, 6, 4, 16, 16)

        return latents, clip_emb


import os
import torch.optim as optim
import torch.nn.functional as F

import os
import torch
from torch.utils.data import Dataset, DataLoader

class BlueprintDataset(Dataset):
    def __init__(self, stft_dir, target_dir):
        self.stft_dir = stft_dir
        self.target_dir = target_dir
        
        # Grab all the newly generated STFT files
        self.stft_files = sorted([f for f in os.listdir(stft_dir) if f.startswith('eeg_sample') and f.endswith('.pt')])
        self.stft_files.sort() # Keep things ordered

    def __len__(self):
        return len(self.stft_files)

    def __getitem__(self, idx):
        stft_filename = self.stft_files[idx]
        
        # Strip '_stft.pt' to find the base name (e.g., 'video_01')
        base_name = stft_filename.split('_')[-1].replace('.pt', '')
        
        # 1. Load Input (EEG)
        eeg_path = os.path.join(self.stft_dir, stft_filename)
        eeg_tensor = torch.load(eeg_path)
        
        # 2. Load Target A (Visuals)
        latent_path = os.path.join(self.target_dir, f"video_sample_{base_name}.pt")
        latent_tensor = torch.load(latent_path)
        
        # 3. Load Target B (Semantics)
        clip_path = os.path.join(self.target_dir, f"text_sample_{base_name}.pt")
        clip_tensor = torch.load(clip_path)
        
        return {
            'eeg': eeg_tensor,
            'latents': latent_tensor,
            'clip_emb': clip_tensor
        }

# --- TRAINING LOOP ---
if __name__ == "__main__":
    print("🚀 Initializing Blueprint Phase 1 Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the new Model
    model = EEGVideoTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 2. Define Data Paths
    TARGET_DIR = "/home/teaching/TEAM_22_DATASET/processed/processed/"
    INPUT_DIR = "/home/teaching/manan_workspace/eeg2video-cs671/data/stft_features"
    
    # 3. Initialize DataLoader
    print("Loading datasets into memory...")
    dataset = BlueprintDataset(INPUT_DIR, TARGET_DIR)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # ... then your model.train() and epoch loop starts here!
    
    
    model.train()
    for epoch in range(50):
        total_epoch_loss = 0
        for batch in dataloader:
            # Inputs must be (B, 62, 51, T) for STFT
            eeg_stft = batch['eeg'].to(device)       
            target_latents = batch['latents'].to(device)   
            target_clip = batch['clip_emb'].to(device)     

            optimizer.zero_grad()

            # Forward Pass: Get both targets
            pred_latents, pred_clip = model(eeg_stft)

            # Loss A: Visual Structure (MSE for VQ-VAE)
            loss_visual = F.mse_loss(pred_latents, target_latents)

            # Loss B: Semantic Meaning (Cosine Similarity for CLIP)
            target_ones = torch.ones(eeg_stft.size(0)).to(device)
            loss_semantic = F.cosine_embedding_loss(pred_clip, target_clip, target_ones)

            # Combine and Backprop
            loss = loss_visual + loss_semantic
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
        print(f"Epoch {epoch} | Total Loss: {total_epoch_loss:.4f}")
    print("✅ Architecture Ready. Waiting for DataLoader to begin training.")
