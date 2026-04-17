import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.2, device=0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# FREQUENCY BAND INDICES  (51 freq bins, 0-50 Hz)
# ============================================================
FREQ_BANDS = [
    (0,  4),   # Delta
    (4,  8),   # Theta
    (8,  14),  # Alpha
    (14, 31),  # Beta
    (31, 51),  # Gamma
]
N_BANDS  = len(FREQ_BANDS)
N_CHAN   = 62
POOL_DIM = N_CHAN * N_BANDS  # 310
FLAT_DIM = 62 * 51 * 9      # 28458


def band_pool(x):
    """(B, 62, 51, 9) -> (B, 310)"""
    B, C, F, T = x.shape
    x_flat = x.view(B, C, -1)
    mean   = x_flat.mean(dim=2, keepdim=True)
    std    = x_flat.std(dim=2,  keepdim=True) + 1e-6
    x_norm = ((x_flat - mean) / std).view(B, C, F, T)
    feats  = []
    for (f_lo, f_hi) in FREQ_BANDS:
        feats.append(x_norm[:, :, f_lo:f_hi, :].mean(dim=[2, 3]))
    return torch.cat(feats, dim=1)   # (B, 310)


# ==========================================
# 1. EEG ADAPTER — Dual-path (flat + band)
#
# Flat path: keeps ALL 28458 raw time-freq values
#            (this is what made text-mlp-full-pipeline
#             achieve val_loss 0.675 — our best result)
# Band path: keeps spectral structure (delta/theta/
#            alpha/beta/gamma per channel)
# Fusion:    concat 256+256 -> 512
#
# Output: (B, 512)  <- contract UNCHANGED
# ==========================================
class EEGAdapter(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        self.flat_path = nn.Sequential(
            nn.BatchNorm1d(FLAT_DIM),
            nn.Linear(FLAT_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.band_path = nn.Sequential(
            nn.BatchNorm1d(POOL_DIM),
            nn.Linear(POOL_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.fusion = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        B = x.shape[0]
        # Flat path — global z-score then flatten
        x_flat      = x.view(B, -1)
        mean_f      = x_flat.mean(dim=1, keepdim=True)
        std_f       = x_flat.std(dim=1,  keepdim=True) + 1e-6
        x_flat_norm = (x_flat - mean_f) / std_f
        path_a      = self.flat_path(x_flat_norm)    # (B, 256)
        path_b      = self.band_path(band_pool(x))   # (B, 256)
        return self.fusion(torch.cat([path_a, path_b], dim=1))  # (B, 512)


# ==========================================
# 2. TEXT EMBEDDING MLP — UNCHANGED
#    Input:  (B, 512)
#    Output: (B, 512)  <- contract UNCHANGED
# ==========================================
class TextProjectorMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. DYNAMICS CLASSIFIER
#    Uses band mean+std -> 620-dim features.
#    Input:  raw EEG (B, 62, 51, 9)
#    Output: (B, 1)  <- contract UNCHANGED
#
#    Key change for stability vs v4:
#    - Added third hidden layer (128 -> 64) so the
#      decision boundary has more capacity to separate
#      the noisy 620-dim feature space.
#    - Label smoothing is applied in the training loop
#      (not here) to reduce overconfident predictions.
# ==========================================
class DynamicsClassifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(POOL_DIM * 2)  # 620
        self.net = nn.Sequential(
            nn.Linear(POOL_DIM * 2, hidden_dim),       # 620 -> 256
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, hidden_dim // 2),    # 256 -> 128
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 128 -> 64
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),             # 64 -> 1
        )

    def forward(self, x):
        B, C, F, T = x.shape
        x_flat = x.view(B, C, -1)
        mean_c = x_flat.mean(dim=2, keepdim=True)
        std_c  = x_flat.std(dim=2,  keepdim=True) + 1e-6
        x_norm = ((x_flat - mean_c) / std_c).view(B, C, F, T)

        band_means, band_stds = [], []
        for (f_lo, f_hi) in FREQ_BANDS:
            band  = x_norm[:, :, f_lo:f_hi, :]
            bflat = band.reshape(B, C, -1)
            band_means.append(bflat.mean(dim=2))
            band_stds.append(bflat.std(dim=2))

        feat = torch.cat(band_means + band_stds, dim=1)  # (B, 620)
        feat = self.bn_input(feat)
        return self.net(feat)                             # (B, 1)


# ==========================================
# VERIFICATION
# ==========================================
def verify_models():
    print(f"Verifying on {DEVICE}...")
    dummy = torch.randn(8, 62, 51, 9).to(DEVICE)

    adapter = EEGAdapter().to(DEVICE)
    a_out   = adapter(dummy)
    assert a_out.shape == (8, 512)
    print(f"EEGAdapter         : {a_out.shape}  <- (B,512) OK")

    t_model = TextProjectorMLP().to(DEVICE)
    t_out   = t_model(a_out)
    assert t_out.shape == (8, 512)
    print(f"TextProjectorMLP   : {t_out.shape}  <- (B,512) OK")

    d_model = DynamicsClassifier().to(DEVICE)
    d_out   = d_model(dummy)
    assert d_out.shape == (8, 1)
    print(f"DynamicsClassifier : {d_out.shape}   <- (B,1)   OK")

    print("\nAll pipeline contracts verified.")

if __name__ == "__main__":
    verify_models()