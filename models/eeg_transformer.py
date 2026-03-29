import torch
import torch.nn as nn

class EEGTransformer(nn.Module):
    def __init__(self, C, latent_dim):
        super().__init__()

        self.input_proj = nn.Linear(C, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.output_proj = nn.Linear(256, latent_dim)

    def forward(self, x):
        B, T, C, S = x.shape

        x = x.permute(0, 1, 3, 2)   # (B, 7, 100, C)
        x = x.reshape(B, T*S, C)    # (B, 700, C)

        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)

        return x
