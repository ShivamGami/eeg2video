import torch
from models.eeg_transformer import EEGTransformer
from models.eeg_to_latent import EEGToLatentModel

# Fix CUDA library issue
import os
os.environ["LD_LIBRARY_PATH"] = os.environ.get("CONDA_PREFIX", "") + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy EEG input
x = torch.randn(2, 7, 32, 100).to(device)

# Build transformer (must output B, 700, 512)
transformer = EEGTransformer(C=32, latent_dim=512)

# Wrap into full model
model = EEGToLatentModel(transformer).to(device)

# Forward pass
out = model(x)

print("Final Output shape:", out.shape)
