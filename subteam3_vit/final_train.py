import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb

# --- THE WINNING HYPERPARAMETERS ---
BEST_BATCH_SIZE = 16
BEST_LR = 0.00005        
BEST_LOSS_WEIGHT = 0.0001
EPOCHS = 50             

# --- ARCHITECTURE ---
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
        return self.conv(x).squeeze(2).permute(0, 2, 1)    

class EEGVideoTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.extractor = STFT_FeatureExtractor(d_model=d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.clip_mlp = nn.Sequential(nn.Linear(d_model, 1024), nn.GELU(), nn.Linear(1024, 512))
        self.latent_head = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 6 * 4 * 16 * 16))

    def forward(self, x):
        features = self.extractor(x)
        features = features + self.pos_encoder[:, :features.size(1), :]
        t_out = self.transformer(features).mean(dim=1)
        return self.latent_head(t_out).view(-1, 6, 4, 16, 16), self.clip_mlp(t_out)

# --- DATASET ---
class BlueprintDataset(Dataset):
    def __init__(self, stft_dir, target_dir):
        self.stft_dir = stft_dir
        self.target_dir = target_dir
        self.stft_files = sorted([f for f in os.listdir(stft_dir) if f.startswith('eeg_sample') and f.endswith('.pt')])

    def __len__(self):
        return len(self.stft_files)

    def __getitem__(self, idx):
        eeg_filename = self.stft_files[idx]
        sample_id = eeg_filename.replace('eeg_sample_', '').replace('.pt', '')
        
        eeg_tensor = torch.load(os.path.join(self.stft_dir, eeg_filename))
        latent_tensor = torch.load(os.path.join(self.target_dir, f"video_sample_{sample_id}.pt"))
        clip_tensor = torch.load(os.path.join(self.target_dir, f"text_sample_{sample_id}.pt"))
        
        return {'eeg': eeg_tensor, 'latents': latent_tensor, 'clip_emb': clip_tensor}

# --- FINAL TRAINING EXECUTION ---
if __name__ == "__main__":
    wandb.init(project="eeg2video-cs671-subteam3_vit", name="FINAL_MODEL_RUN") 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGVideoTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
    scaler = GradScaler() 
    
    INPUT_DIR = "/home/teaching/manan_workspace/eeg2video-cs671/data/stft_features"
    TARGET_DIR = "/home/teaching/TEAM_22_DATASET/processed/processed/"
    
    dataset = BlueprintDataset(INPUT_DIR, TARGET_DIR)
    # Added num_workers to load data faster
    dataloader = DataLoader(dataset, batch_size=BEST_BATCH_SIZE, shuffle=True, num_workers=4) 
    
    print("Starting Final Training Run...")
    model.train()
    for epoch in range(EPOCHS): 
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            with autocast():
                p_lat, p_clip = model(batch['eeg'].to(device))
                l_vis = F.mse_loss(p_lat, batch['latents'].to(device))
                l_sem = F.cosine_embedding_loss(p_clip, batch['clip_emb'].to(device), torch.ones(batch['eeg'].size(0)).to(device))
                total_loss = (l_vis * BEST_LOSS_WEIGHT) + l_sem
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            
        print(f"Epoch {epoch} | Loss: {epoch_loss}")
        wandb.log({"Total_Loss": epoch_loss, "Epoch": epoch})
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"vit_checkpoint_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "FINAL_VIT_MODEL.pth")
    print("Training Complete. Model Saved.")
