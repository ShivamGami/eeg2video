"""
╔══════════════════════════════════════════════════════════════╗
║  EEG2VIDEO DATASET — CS671 TEAM 22                           ║
║                                                              ║
║  Reads preprocessed .pt triplets from disk                   ║
║  Returns matched (eeg, text, video) per sample               ║
║                                                              ║
║  CONFIRMED INPUT SHAPES:                                     ║
║    eeg_sample_XXXXXX.pt   → (62, 51, 9)                      ║
║    text_sample_XXXXXX.pt  → (512,)                           ║
║    video_sample_XXXXXX.pt → (6, 4, 16, 16)                   ║
║                                                              ║
║  CONFIRMED OUTPUT BATCH SHAPES:                              ║
║    eeg   → (B, 62, 51, 9)                                    ║
║    text  → (B, 512)                                          ║
║    video → (B, 6, 4, 16, 16)                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class EEGVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for EEG-to-Video generation.

    Reads preprocessed .pt files from disk.
    Each sample is a matched triplet of:
        - EEG STFT spectrogram
        - CLIP text embedding
        - VQ-VAE video latent

    Args:
        data_dir : str  path to processed/ folder
        split    : str  one of 'train', 'val', 'test'

    Returns per item (dict):
        eeg       : FloatTensor (62, 51, 9)
        text      : FloatTensor (512,)
        video     : FloatTensor (6, 4, 16, 16)
        sample_id : str e.g '000001'

    Example:
        ds   = EEGVideoDataset('/path/to/processed', split='train')
        item = ds[0]
        eeg  = item['eeg']    # (62, 51, 9)
        text = item['text']   # (512,)
    """

    # ── Expected shapes (from verified preprocessing) ──────────
    EEG_SHAPE   = (62, 51, 9)
    TEXT_SHAPE  = (512,)
    VIDEO_SHAPE = (6, 4, 16, 16)

    def __init__(self, data_dir, split="train"):
        super().__init__()

        # Validate split argument
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val' or 'test'. Got: {split}"

        self.data_dir = data_dir
        self.split    = split

        # ── Step 1: Load split file ─────────────────────────────
        split_file = os.path.join(data_dir, f"{split}_split.txt")

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"\n❌ Split file not found: {split_file}"
                f"\n   Make sure preprocess_full.py has been run first!"
                f"\n   Expected location: {data_dir}/{split}_split.txt"
            )

        with open(split_file) as f:
            self.sample_ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(self.sample_ids) == 0:
            raise ValueError(
                f"❌ Split file is empty: {split_file}"
            )

        # ── Step 2: Verify sample files exist ──────────────────
        self._verify_samples()

        print(
            f"✅ EEGVideoDataset [{split:5s}]: "
            f"{len(self.sample_ids):>8,} samples loaded"
        )

    def _verify_samples(self):
        """
        Check first, middle and last sample exist on disk.
        Catches missing files early before training starts.
        """
        # Check 3 samples: first, middle, last
        n = len(self.sample_ids)
        check_ids = [
            self.sample_ids[0],
            self.sample_ids[n // 2],
            self.sample_ids[-1],
        ]

        for sid in check_ids:
            for prefix in ["eeg", "text", "video"]:
                path = os.path.join(
                    self.data_dir,
                    f"{prefix}_sample_{sid}.pt"
                )
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"\n❌ Missing file: {path}"
                        f"\n   Sample ID {sid} is in split file"
                        f"   but .pt file does not exist!"
                        f"\n   Re-run preprocess_full.py"
                    )

    def __len__(self):
        """Total number of samples in this split."""
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Load and return one matched triplet.

        Args:
            idx : int  index into sample list

        Returns:
            dict with keys:
                eeg       : FloatTensor (62, 51, 9)
                text      : FloatTensor (512,)
                video     : FloatTensor (6, 4, 16, 16)
                sample_id : str
        """
        sid = self.sample_ids[idx]

        # ── Build file paths ────────────────────────────────────
        eeg_path   = os.path.join(
            self.data_dir, f"eeg_sample_{sid}.pt"
        )
        text_path  = os.path.join(
            self.data_dir, f"text_sample_{sid}.pt"
        )
        video_path = os.path.join(
            self.data_dir, f"video_sample_{sid}.pt"
        )

        # ── Load tensors from disk ──────────────────────────────
        try:
            eeg = torch.load(
                eeg_path,
                map_location = "cpu",
                weights_only = True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load EEG {sid}: {e}")

        try:
            text = torch.load(
                text_path,
                map_location = "cpu",
                weights_only = True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load text {sid}: {e}")

        try:
            video = torch.load(
                video_path,
                map_location = "cpu",
                weights_only = True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load video {sid}: {e}")

        # ── Shape verification ──────────────────────────────────
        if eeg.shape != torch.Size(list(self.EEG_SHAPE)):
            raise ValueError(
                f"EEG shape mismatch for {sid}: "
                f"got {tuple(eeg.shape)}, "
                f"expected {self.EEG_SHAPE}"
            )

        if text.shape != torch.Size(list(self.TEXT_SHAPE)):
            raise ValueError(
                f"Text shape mismatch for {sid}: "
                f"got {tuple(text.shape)}, "
                f"expected {self.TEXT_SHAPE}"
            )

        if video.shape != torch.Size(list(self.VIDEO_SHAPE)):
            raise ValueError(
                f"Video shape mismatch for {sid}: "
                f"got {tuple(video.shape)}, "
                f"expected {self.VIDEO_SHAPE}"
            )

        # ── Ensure correct dtype ────────────────────────────────
        eeg   = eeg.float()
        text  = text.float()
        video = video.float()

        return {
            "eeg"       : eeg,       # (62, 51, 9)
            "text"      : text,      # (512,)
            "video"     : video,     # (6, 4, 16, 16)
            "sample_id" : sid,       # "000001"
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 2: DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════════

def get_dataloaders(
    data_dir,
    batch_size  = 32,
    num_workers = 4,
):
    """
    Build train, val and test DataLoaders.

    Args:
        data_dir    : str  path to processed/ folder
        batch_size  : int  samples per batch
        num_workers : int  parallel loading workers

    Returns:
        train_loader : DataLoader
        val_loader   : DataLoader
        test_loader  : DataLoader

    Batch tensor shapes:
        eeg   : (B, 62, 51, 9)
        text  : (B, 512)
        video : (B, 6, 4, 16, 16)

    Example:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir   = '/path/to/processed',
            batch_size = 32,
        )
        for batch in train_loader:
            eeg   = batch['eeg']    # (32, 62, 51, 9)
            text  = batch['text']   # (32, 512)
            video = batch['video']  # (32, 6, 4, 16, 16)
    """

    print("\n" + "="*60)
    print("📦 BUILDING DATALOADERS")
    print("="*60)

    # ── Build datasets ──────────────────────────────────────────
    train_ds = EEGVideoDataset(data_dir, split="train")
    val_ds   = EEGVideoDataset(data_dir, split="val")
    test_ds  = EEGVideoDataset(data_dir, split="test")

    # ── Build loaders ───────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,          # shuffle every epoch
        num_workers = num_workers,
        pin_memory  = True,          # faster GPU transfer
        drop_last   = True,          # avoid partial batches
        persistent_workers = (num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,         # no shuffle for val/test
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        persistent_workers = (num_workers > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        persistent_workers = (num_workers > 0),
    )

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n   Batch size      : {batch_size}")
    print(f"   Num workers     : {num_workers}")
    print(f"\n   Train batches   : {len(train_loader):,}")
    print(f"   Val   batches   : {len(val_loader):,}")
    print(f"   Test  batches   : {len(test_loader):,}")
    print(f"\n   Batch shapes:")
    print(f"     eeg   → (B={batch_size}, 62, 51, 9)")
    print(f"     text  → (B={batch_size}, 512)")
    print(f"     video → (B={batch_size}, 6, 4, 16, 16)")

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════
# SECTION 3: SELF TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DATA_DIR = "/home/teaching/TEAM_22_DATASET/processed/processed"

    print("="*60)
    print("🧪 TESTING dataset.py")
    print("="*60)
    print(f"Data dir: {DATA_DIR}")

    # ── Test 1: Dataset creation ────────────────────────────────
    print("\n📋 TEST 1: Dataset Creation")
    print("─"*40)

    train_ds = EEGVideoDataset(DATA_DIR, split="train")
    val_ds   = EEGVideoDataset(DATA_DIR, split="val")
    test_ds  = EEGVideoDataset(DATA_DIR, split="test")

    total = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"\n   Total samples: {total:,}")

    # ── Test 2: Single item loading ─────────────────────────────
    print("\n📋 TEST 2: Single Item Loading")
    print("─"*40)

    item = train_ds[0]

    print(f"\n   sample_id  : {item['sample_id']}")
    print(f"   eeg  shape : {item['eeg'].shape}")
    print(f"   text shape : {item['text'].shape}")
    print(f"   video shape: {item['video'].shape}")
    print(f"   eeg  dtype : {item['eeg'].dtype}")
    print(f"   text dtype : {item['text'].dtype}")
    print(f"   video dtype: {item['video'].dtype}")
    print(f"   eeg  range : [{item['eeg'].min():.3f}, {item['eeg'].max():.3f}]")
    print(f"   text range : [{item['text'].min():.3f}, {item['text'].max():.3f}]")
    print(f"   video range: [{item['video'].min():.3f}, {item['video'].max():.3f}]")

    # Verify shapes
    assert item["eeg"].shape   == torch.Size([62, 51, 9]),    "❌ EEG shape wrong"
    assert item["text"].shape  == torch.Size([512]),           "❌ Text shape wrong"
    assert item["video"].shape == torch.Size([6, 4, 16, 16]), "❌ Video shape wrong"
    assert item["eeg"].dtype   == torch.float32,               "❌ EEG dtype wrong"
    assert item["text"].dtype  == torch.float32,               "❌ Text dtype wrong"
    assert item["video"].dtype == torch.float32,               "❌ Video dtype wrong"

    print("\n   ✅ All shapes and dtypes correct!")

    # ── Test 3: DataLoader batch ────────────────────────────────
    print("\n📋 TEST 3: DataLoader Batch")
    print("─"*40)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir    = DATA_DIR,
        batch_size  = 8,
        num_workers = 0,       # 0 for safe testing
    )

    batch = next(iter(train_loader))

    print(f"\n   Batch eeg   shape : {batch['eeg'].shape}")
    print(f"   Batch text  shape : {batch['text'].shape}")
    print(f"   Batch video shape : {batch['video'].shape}")
    print(f"   Sample IDs        : {batch['sample_id'][:3]}...")

    assert batch["eeg"].shape   == torch.Size([8, 62, 51, 9])
    assert batch["text"].shape  == torch.Size([8, 512])
    assert batch["video"].shape == torch.Size([8, 6, 4, 16, 16])

    print("\n   ✅ Batch shapes correct!")

    # ── Test 4: Multiple batches ────────────────────────────────
    print("\n📋 TEST 4: Iterating Multiple Batches")
    print("─"*40)

    count = 0
    for batch in train_loader:
        assert batch["eeg"].shape[1:]   == torch.Size([62, 51, 9])
        assert batch["text"].shape[1:]  == torch.Size([512])
        assert batch["video"].shape[1:] == torch.Size([6, 4, 16, 16])
        count += 1
        if count == 5:
            break

    print(f"   Iterated {count} batches successfully")
    print(f"   ✅ All batches have correct shapes!")

    # ── Final Summary ───────────────────────────────────────────
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED")
    print("="*60)
    print(f"   Train : {len(train_ds):,} samples")
    print(f"   Val   : {len(val_ds):,} samples")
    print(f"   Test  : {len(test_ds):,} samples")
    print(f"\n   EEG   shape : (B, 62, 51, 9)")
    print(f"   Text  shape : (B, 512)")
    print(f"   Video shape : (B, 6, 4, 16, 16)")
    print(f"\n   Import with:")
    print(f"   from dataset import get_dataloaders")
    print("="*60)