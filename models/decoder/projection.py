import torch
import torch.nn as nn

class EEGProjection(nn.Module):
    def __init__(
        self,
        input_dim:  int = 512,
        output_dim: int = 768,
        seq_len:    int = 77,
        num_heads:  int = 12, # Increased for more detail
        dropout:    float = 0.05, # Lower dropout for sharper features
    ):
        super().__init__()
        self.seq_len = seq_len

        # Step 1: Deep MLP to extract EEG features
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Step 2: Learnable Latent Queries (The "Secret Sauce")
        # Instead of repeating EEG, we start with 77 unique "slots"
        self.latent_queries = nn.Parameter(torch.randn(1, seq_len, output_dim) * 0.02)

        # Step 3: Multi-Layer Transformer (More brain power)
        # We'll use 4 layers instead of 1 to really process the strokes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Step 4: Final Stroke Refiner
        self.out_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Extract deep EEG features
        eeg_features = self.input_proj(x).unsqueeze(1) # (B, 1, 768)

        # Combine EEG with 77 learnable queries
        # This allows the EEG to modulate 77 unique tokens
        queries = self.latent_queries.expand(B, -1, -1) # (B, 77, 768)
        h = queries + eeg_features # Inject EEG context into every token

        # Let the 4-layer Transformer refine the calligraphy structure
        h = self.transformer(h) 

        return self.out_proj(h)
