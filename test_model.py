import torch
from models.eeg_transformer import EEGTransformer
from models.eeg_to_latent import EEGToLatentModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy input
x = torch.randn(4, 7, 32, 100).to(device)

transformer = EEGTransformer(C=32, latent_dim=512)
model = EEGToLatentModel(transformer).to(device)

out = model(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)
