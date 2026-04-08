"""
Central configuration for EEG2Video pipeline.
All hyperparameters in one place.
"""

import torch

# ═══════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════
PATHS = {
    "data_dir"    : "/home/teaching/TEAM_22_DATASET/processed/processed",
    "checkpoint_dir" : "./checkpoints",
    "log_dir"        : "./logs",
}

# ═══════════════════════════════════════════
# DATA CONTRACT (VERIFIED SHAPES)
# ═══════════════════════════════════════════
DATA = {
    "eeg_channels"  : 62,
    "eeg_freq_bins" : 51,
    "eeg_time_bins" : 9,
    "text_dim"      : 512,
    "video_frames"  : 6,
    "video_latent_ch": 4,
    "video_latent_h" : 16,
    "video_latent_w" : 16,
}

# ═══════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════
MODEL = {
    # EEGNet Transformer
    "patch_size"      : 9,        # patch along time axis
    "embed_dim"       : 256,      # transformer hidden dim
    "num_heads"       : 8,        # attention heads
    "num_layers"      : 4,        # transformer layers
    "dropout"         : 0.1,

    # CLIP Projection MLP
    "clip_dim"        : 512,      # target CLIP space dim
    "mlp_hidden_dim"  : 512,      # MLP hidden size
}

# ═══════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════
TRAIN = {
    "batch_size"      : 32,
    "num_epochs"      : 50,
    "learning_rate"   : 1e-4,
    "weight_decay"    : 1e-5,
    "num_workers"     : 4,
    "device"          : "cuda" if torch.cuda.is_available() else "cpu",

    # Loss weights
    "lambda_cosine"   : 1.0,
    "lambda_recon"    : 0.5,

    # Scheduler
    "scheduler"       : "cosine",
    "warmup_epochs"   : 5,

    # Checkpointing
    "save_every"      : 5,        # save every N epochs
    "early_stop"      : 10,       # stop if no improvement
}

# ═══════════════════════════════════════════
# TARGET METRICS
# ═══════════════════════════════════════════
TARGETS = {
    "clip_cosine_sim" : 0.55,     # minimum target from rubric
    "fid_score"       : 10.0,     # maximum FID target
}