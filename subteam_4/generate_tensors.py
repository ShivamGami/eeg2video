"""
generate_tensors.py — ALIGNED Inference Script
=============================================================================
Loads trained weights and exports conditioning tensors in STRICT ALPHABETICAL 
ORDER to guarantee 1-to-1 match with Subteam 3's visual_latents.pt.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from subteam4_models import TextProjectorMLP, EEGAdapter, DynamicsClassifier

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR      = "/home/teaching/TEAM_22_DATASET/processed/processed"
BATCH_SIZE    = 256

TEXT_MLP_PATH = "text_mlp_final.pth"
ADAPTER_PATH  = "eeg_adapter.pth"
DYNAMICS_PATH = "dynamics_model.pth"

OUT_TEXT      = "text_embeddings.pt"
OUT_DYNAMICS  = "is_fast.pt"
# ─────────────────────────────────────────────────────────────────────────────

# 🚀 CUSTOM DATASET: Guarantees exact same alphabetical sorting as Manan's script
import re

# 🚀 CUSTOM DATASET: (UPDATED TO FIX STORAGE & 873K FILES ISSUE)
class AlignedInferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        all_files = os.listdir(data_dir)
        
        # 🔥 PROBLEM 1 FIX: Filter out non-EEG files!
        # We only want files with numbers (samples) and ignore text/video/image modalities
        valid_files = []
        for f in all_files:
            if f.endswith(('.pt', '.npy')) and re.search(r'\d+', f):
                # Reject files that clearly belong to other modalities
                if not any(x in f.lower() for x in ['video', 'text', 'image', 'label']):
                    valid_files.append(f)
                    
        self.files = sorted(valid_files)
        print(f"🔥 Filtered out noise! Now working with EXACTLY {len(self.files)} EEG files.")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        
        if file_path.endswith('.pt'):
            # 🔥 PROBLEM 2 FIX: .clone().detach() is the magic cure for "not resizable" errors
            eeg = torch.load(file_path).clone().detach().float()
        else:
            import numpy as np
            eeg = torch.tensor(np.load(file_path), dtype=torch.float32)
            
        return eeg, 0, 0, 0
    
def load_models():
    adapter  = EEGAdapter().to(DEVICE)
    text_mlp = TextProjectorMLP(input_dim=512, hidden_dim=1024, output_dim=512).to(DEVICE)
    dynamics = DynamicsClassifier().to(DEVICE)

    adapter.load_state_dict(torch.load(ADAPTER_PATH,  map_location=DEVICE))
    text_mlp.load_state_dict(torch.load(TEXT_MLP_PATH, map_location=DEVICE))
    dynamics.load_state_dict(torch.load(DYNAMICS_PATH, map_location=DEVICE))

    adapter.eval(); text_mlp.eval(); dynamics.eval()
    print(f"Loaded weights successfully!")
    return adapter, text_mlp, dynamics


def generate_aligned_tensors():
    adapter, text_mlp, dynamics = load_models()

    # Use our custom strictly-sorted dataset instead of splits
    dataset = AlignedInferenceDataset(DATA_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"\nGenerating ALIGNED tensors | Total Files Found: {len(dataset)}")

    all_text_embeds = []
    all_is_fast     = []

    with torch.no_grad():
        for batch_idx, (eeg, _, _, _) in enumerate(loader):
            eeg = eeg.to(DEVICE)   # Expected: (B, 62, 51, 9)

            # ── Text embedding ─────────────────────────────────────────────
            adapted    = adapter(eeg)             
            text_embed = text_mlp(adapted)        
            text_embed = F.normalize(text_embed, p=2, dim=1)

            # ── Dynamics prediction ────────────────────────────────────────
            logits  = dynamics(eeg)               
            is_fast = (torch.sigmoid(logits) > 0.5).float()  

            all_text_embeds.append(text_embed.cpu())
            all_is_fast.append(is_fast.cpu())

            if (batch_idx + 1) % 50 == 0:
                done = (batch_idx + 1) * BATCH_SIZE
                print(f"  Processed {min(done, len(dataset))}/{len(dataset)} samples...")

    # ── Concatenate and save ───────────────────────────────────────────────
    text_tensor = torch.cat(all_text_embeds, dim=0)  # (N, 512)
    fast_tensor = torch.cat(all_is_fast,     dim=0)  # (N, 1)

    torch.save(text_tensor, OUT_TEXT)
    torch.save(fast_tensor, OUT_DYNAMICS)

    print(f"\n=====================================================")
    print(f"🎉 ALIGNED FINAL TENSORS READY")
    print(f"Saved: {OUT_TEXT}    shape={tuple(text_tensor.shape)}")
    print(f"Saved: {OUT_DYNAMICS} shape={tuple(fast_tensor.shape)}")
    print(f"=====================================================")


if __name__ == "__main__":
    generate_aligned_tensors()