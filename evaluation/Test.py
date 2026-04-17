import torch
# Load one sample to check
latent = torch.load('/home/teaching/TEAM_22_DATASET/Phase1_Tensors/sample_0_latent.pt')
print(f"Latent Shape: {latent.shape}") 

eeg = torch.load('/home/teaching/TEAM_22_DATASET/Phase1_Tensors/sample_0_eeg.pt')
print(f"EEG Shape: {eeg.shape}")