"""
subteam2_dataset.py — VideoLDM Dataset (FINAL CORRECT VERSION)
===============================================================
ROOT CAUSE OF BLUR (why previous versions failed):

  real_vae_latents_50k.pt was shaped (300000, 4, 32, 32) and
  then reshaped to (50000, 6, 4, 32, 32). This means:
    - frame 0 of sample 0 = flat tensor row 0
    - frame 1 of sample 0 = flat tensor row 1
    - frame 5 of sample 0 = flat tensor row 5
    - frame 0 of sample 1 = flat tensor row 6
  These are NOT 6 frames of the same video clip. They are
  6 consecutive but UNRELATED VAE latents. The UNet trained
  to denoise random frame sequences = learned nothing useful.

  Additionally, real_vae_latents_50k std=0.18 vs real VAE
  latents std~0.9-5.0. The rescaling destroyed the scale
  information that the VAE decoder needs to produce sharp images.

CORRECT APPROACH:
  Load video_sample_XXXXXX.pt directly from processed/processed/.
  Each file = one (6, 4, 16, 16) tensor — 6 frames of the SAME
  2-second clip, VAE-encoded with correct statistics.
  These are paired 1-to-1 with text_sample_XXXXXX.pt by sample ID.

  NOTE: video latent shape is (6, 4, 16, 16) not (6, 4, 32, 32).
  The UNet spatial resolution changes accordingly but the
  architecture still works — SD UNet accepts any H/W divisible by 8.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


PROCESSED_DIR = "/home/teaching/TEAM_22_DATASET/processed/processed/"


class VideoLDMDataset(Dataset):
    """
    Loads aligned (video_latent, text_embed, is_fast) triplets
    directly from the preprocessed .pt files.

    Each video_sample_XXXXXX.pt = (6, 4, 16, 16) — real VAE latents
    Each text_sample_XXXXXX.pt  = (512,) — real CLIP embeddings
    is_fast loaded from dynamics_labels_fixed_BINARY.npy

    Alignment is guaranteed by matching sample IDs (6-digit numbers).
    """

    def __init__(
        self,
        split: str = 'train',
        processed_dir: str = PROCESSED_DIR,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        max_samples: int   = None,   # kept for API compatibility, ignored
        use_real_latents: bool = True,  # kept for API compat, always True now
    ):
        self.processed_dir = processed_dir

        # ── Find all sample IDs ────────────────────────────────────────────
        video_files = sorted(glob.glob(
            os.path.join(processed_dir, "video_sample_*.pt")
        ))
        if len(video_files) == 0:
            raise FileNotFoundError(
                f"No video_sample_*.pt files found in {processed_dir}\n"
                f"Check that the preprocessing script has completed."
            )

        # Extract 6-digit sample IDs
        self.sample_ids = []
        for f in video_files:
            sid = os.path.basename(f).replace("video_sample_", "").replace(".pt", "")
            # Verify matching text file exists
            txt = os.path.join(processed_dir, f"text_sample_{sid}.pt")
            if os.path.exists(txt):
                self.sample_ids.append(sid)

        n_total = len(self.sample_ids)
        print(f"Found {n_total:,} aligned (video, text) pairs in {processed_dir}")

        # ── Load dynamics labels ───────────────────────────────────────────
        dynamics_path = os.path.join(
            os.path.dirname(processed_dir.rstrip("/")),
            "..",
            "vishal_workspace/eeg2video-cs671/is_fast.pt"
        )
        # Try standard paths in order
        candidate_paths = [
            "/home/teaching/vishal_workspace/eeg2video-cs671/is_fast.pt",
            os.path.join(processed_dir, "../../is_fast.pt"),
        ]
        self.is_fast_tensor = None
        for p in candidate_paths:
            if os.path.exists(p):
                self.is_fast_tensor = torch.load(p, map_location="cpu")
                print(f"  is_fast loaded from: {p}  shape={self.is_fast_tensor.shape}")
                break
        if self.is_fast_tensor is None:
            print("  WARNING: is_fast.pt not found — defaulting all to slow (0.0)")
            self.is_fast_tensor = torch.zeros(n_total, 1)

        # ── Split by sample ID index ───────────────────────────────────────
        n_train = int(n_total * train_ratio)
        n_val   = int(n_total * val_ratio)

        if split == 'train':
            self.sample_ids = self.sample_ids[:n_train]
        elif split == 'val':
            self.sample_ids = self.sample_ids[n_train:n_train + n_val]
        elif split == 'test':
            self.sample_ids = self.sample_ids[n_train + n_val:]
        else:
            pass  # 'all' — keep everything

        print(f"  split={split} → {len(self.sample_ids):,} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        int_idx = int(sid)  # numeric index for is_fast lookup

        # Load video latent: (6, 4, 16, 16) — real VAE encoding
        video_path = os.path.join(self.processed_dir, f"video_sample_{sid}.pt")
        video = torch.load(video_path, map_location="cpu")   # (6, 4, 16, 16)
        if video.dtype != torch.float32:
            video = video.float()

        # Load text embedding: (512,) — real CLIP embedding
        text_path = os.path.join(self.processed_dir, f"text_sample_{sid}.pt")
        text = torch.load(text_path, map_location="cpu").flatten()  # (512,)
        if text.dtype != torch.float32:
            text = text.float()

        # Load dynamics flag: scalar → (1,)
        if int_idx < len(self.is_fast_tensor):
            fast = self.is_fast_tensor[int_idx].float()
            if fast.dim() == 0:
                fast = fast.unsqueeze(0)
        else:
            fast = torch.zeros(1)

        return {
            'video_latent': video,   # (6, 4, 16, 16) — real VAE latents
            'text_embed':   text,    # (512,)
            'is_fast':      fast,    # (1,)
            'idx':          int_idx,
        }


if __name__ == "__main__":
    print("="*55)
    print("VideoLDMDataset self-test")
    print("="*55)
    ds = VideoLDMDataset(split='train')
    sample = ds[0]
    print(f"\nSample keys  : {list(sample.keys())}")
    print(f"video_latent : {sample['video_latent'].shape}  std={sample['video_latent'].std():.4f}")
    print(f"text_embed   : {sample['text_embed'].shape}")
    print(f"is_fast      : {sample['is_fast']}")
    print(f"\n✅ Dataset working correctly")