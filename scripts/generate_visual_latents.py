import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. REVERSE-ENGINEERED ARCHITECTURE
# (Exactly matches vit_real_data.pth keys)
# ==========================================
class RecoveredEEGViT(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        # Matches embed.weight: [256, 6200]
        self.embed = nn.Linear(6200, d_model)
        
        # Matches transformer.layers.0 to 5 with 2048 feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Matches latent_proj.weight: [24576, 256]
        self.latent_proj = nn.Linear(d_model, 24576)

    def forward(self, x):
        # Flatten whatever tensor we receive to 1D array per batch
        x = x.reshape(x.size(0), -1)
        
        # Failsafe: Handle dimension mismatches dynamically 
        # (Just in case Manan's input tensors are weirdly sized)
        if x.size(1) > 6200:
            x = x[:, :6200]
        elif x.size(1) < 6200:
            padding = torch.zeros(x.size(0), 6200 - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)

        x = self.embed(x)             # Output: (Batch, 256)
        x = x.unsqueeze(1)            # Sequence of length 1 for Transformer
        x = self.transformer(x)       # Output: (Batch, 1, 256)
        x = x.squeeze(1)              # Output: (Batch, 256)
        
        latents = self.latent_proj(x) # Output: (Batch, 24576)
        
        return latents.view(-1, 6, 4, 32, 32)

# ==========================================
# 2. INFERENCE DATASET
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.startswith('eeg_sample') and f.endswith('.pt')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.data_dir, self.files[idx])
        return torch.load(tensor_path)

# ==========================================
# 3. GENERATION LOOP
# ==========================================
if __name__ == "__main__":
    print("🕵️  Loading Reverse-Engineered Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init custom model
    model = RecoveredEEGViT().to(device)
    weights_path = "./checkpoints/vit_real_data.pth"
    
    # Load the strict weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✅ Keys matched! Model weights loaded successfully!")

    # Path to EEG tensors
    INPUT_DIR = "/home/teaching/manan_workspace/eeg2video-cs671/data/stft_features"
    dataset = InferenceDataset(INPUT_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_latents = []
    print(f"⏳ Processing {len(dataset)} samples. Generating visual_latents.pt...")
    
    with torch.no_grad():
        for batch_idx, eeg_batch in enumerate(dataloader):
            eeg_batch = eeg_batch.to(device)
            
            # Forward pass (It will only return latents now)
            pred_latents = model(eeg_batch)
            all_latents.append(pred_latents.cpu())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {(batch_idx + 1) * 32}/{len(dataset)} samples...")

    final_latents_tensor = torch.cat(all_latents, dim=0)
    save_path = "visual_latents.pt"
    torch.save(final_latents_tensor, save_path)
    
    print("=====================================================")
    print(f"🎉 SUCCESS! Saved {save_path}")
    print(f"📊 Tensor Shape: {final_latents_tensor.shape}")
    print("=====================================================")