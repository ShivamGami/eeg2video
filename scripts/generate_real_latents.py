import os
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Import YOUR actual architecture from your training script
from final_train import EEGVideoTransformer 

# 2. Recreate the Dataset loader
class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.startswith('eeg_sample') and f.endswith('.pt')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_dir, self.files[idx]))

if __name__ == "__main__":
    print("🚀 Loading the REAL 10-Hour Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize YOUR model and load YOUR weights
    model = EEGVideoTransformer().to(device)
    model.load_state_dict(torch.load("FINAL_VIT_MODEL.pth", map_location=device))
    model.eval()
    print("✅ Weights loaded successfully!")

    # 4. Run the dataset through the model
    INPUT_DIR = "/home/teaching/TEAM_22_DATASET/processed"
    dataset = InferenceDataset(INPUT_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_latents = []
    print(f"⏳ Processing {len(dataset)} samples. Generating new visual_latents.pt...")

    with torch.no_grad():
        for batch_idx, eeg_batch in enumerate(dataloader):
            if batch_idx >= 10:  # Only do 10 batches to save memory
                break
            eeg_batch = eeg_batch.to(device)
            pred_latents, _ = model(eeg_batch)
            all_latents.append(pred_latents.cpu())
            if (batch_idx + 1) % 5 == 0:
                print(f"Processed batch {batch_idx + 1}")

    # --- SAVE SECTION ---
    print("Finalizing tensor...")
    final_latents_tensor = torch.cat(all_latents, dim=0)
    
    # Standardize to make the signal "loud"
    mean = final_latents_tensor.mean()
    std = final_latents_tensor.std()
    final_latents_tensor = (final_latents_tensor - mean) / (std + 1e-6)
    
    torch.save(final_latents_tensor, "visual_latents.pt")
    print("✅ SUCCESS! Saved visual_latents.pt")
