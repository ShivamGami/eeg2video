# dataset.py
"""
Loads the preprocessed SEED-DV triplets from:
/home/teaching/TEAM_22_DATASET/processed/processed/

Each sample:
    eeg_sample_XXXXXX.pt   → (62, 51, 9)
    text_sample_XXXXXX.pt  → (512,)
    video_sample_XXXXXX.pt → (6, 4, 16, 16)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

PROCESSED_DIR = "/home/teaching/TEAM_22_DATASET/processed/processed"


class SEEDDV_Dataset(Dataset):

    def __init__(self, split: str = "train"):
        """
        split: "train", "val", or "test"
        """
        split_file = os.path.join(PROCESSED_DIR, f"{split}_split.txt")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Make sure preprocessing has finished."
            )

        with open(split_file, "r") as f:
            # Each line is a sample ID like "000001"
            self.sample_ids = [line.strip() for line in f if line.strip()]

        print(f"[SEEDDV_Dataset] {split} split: {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        eeg_path   = os.path.join(PROCESSED_DIR, f"eeg_sample_{sid}.pt")
        text_path  = os.path.join(PROCESSED_DIR, f"text_sample_{sid}.pt")
        video_path = os.path.join(PROCESSED_DIR, f"video_sample_{sid}.pt")

        # Check all files exist
        for path in (eeg_path, text_path, video_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing file: {path}\n"
                    f"Sample {sid} is incomplete."
                )

        eeg   = torch.load(eeg_path,   map_location="cpu").float()  # (62, 51, 9)
        text  = torch.load(text_path,  map_location="cpu").float()  # (512,)
        video = torch.load(video_path, map_location="cpu").float()  # (6, 4, 16, 16)

        # Expand text: (512,) → (77, 512) by broadcasting across 77 CLIP tokens
        # This matches the (B, 77, 512) contract expected by TemporalUNet
        text_expanded = text.unsqueeze(0).expand(77, -1)  # (77, 512)

        return {
            "eeg"   : eeg,            # (62, 51, 9)
            "text"  : text_expanded,  # (77, 512)
            "video" : video,          # (6, 4, 16, 16)
            "id"    : sid,
        }


def get_dataloader(split: str = "train", batch_size: int = 4,
                   shuffle: bool = True, num_workers: int = 2):
    dataset = SEEDDV_Dataset(split=split)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = True,
    )


# Quick test
if __name__ == "__main__":
    print("Testing dataset loader...")
    loader = get_dataloader(split="train", batch_size=4)
    batch  = next(iter(loader))

    print(f"  EEG shape  : {batch['eeg'].shape}")    # (4, 62, 51, 9)
    print(f"  Text shape : {batch['text'].shape}")   # (4, 77, 512)
    print(f"  Video shape: {batch['video'].shape}")  # (4, 6, 4, 16, 16)
    print(f"  Sample IDs : {batch['id']}")
    print("Dataset loader working correctly ✓")