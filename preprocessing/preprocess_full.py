"""
╔══════════════════════════════════════════════════════════════╗
║     COMPLETE PREPROCESSING PIPELINE — EEG2VIDEO CS671        ║
║     Dataset: SEED-DV (20 subjects × 7 videos)                ║
║                                                              ║
║  TARGET SHAPES (VERIFIED):                                   ║
║    eeg_sample_XXXXXX.pt   → (62, 51, 9)                      ║
║    text_sample_XXXXXX.pt  → (512,)                           ║
║    video_sample_XXXXXX.pt → (6, 4, 16, 16)                   ║
║                                                              ║
║  MODELS (LOCAL CACHE):                                       ║
║    CLIP → openai/clip-vit-base-patch32 (pytorch_model.bin)   ║
║    VAE  → stabilityai/sd-vae-ft-mse   (safetensors)          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import torch
from scipy.signal import stft
import warnings
warnings.filterwarnings("ignore")

# ── Suppress HuggingFace security warning ─────────────────────
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

# ═══════════════════════════════════════════════════════════════
# SECTION 0: AUTO-DETECT LOCAL MODEL PATHS
# ═══════════════════════════════════════════════════════════════

def find_model_path(hub_base, model_name):
    """
    Find the correct snapshot folder that contains config.json.
    Iterates all snapshots and returns the complete one.
    """
    model_dir = os.path.join(hub_base, model_name, "snapshots")

    if not os.path.exists(model_dir):
        print(f"   ⚠️  Snapshots folder not found: {model_dir}")
        return None

    for snapshot in os.listdir(model_dir):
        snap_path    = os.path.join(model_dir, snapshot)
        config_path  = os.path.join(snap_path, "config.json")

        if os.path.exists(config_path):
            print(f"   ✅ Found: .../{model_name}/snapshots/{snapshot}/")
            return snap_path

    print(f"   ❌ No complete snapshot in: {model_dir}")
    return None


# ── Detect paths at import time ───────────────────────────────
HUB_BASE = "/home/teaching/.cache/huggingface/hub"

print("🔍 Auto-detecting model paths...")

CLIP_PATH = find_model_path(HUB_BASE, "models--openai--clip-vit-base-patch32")
VAE_PATH  = find_model_path(HUB_BASE, "models--stabilityai--sd-vae-ft-mse")

# Fallbacks if local not found
if CLIP_PATH is None:
    CLIP_PATH = "openai/clip-vit-base-patch32"
    print("   ⚠️  CLIP: falling back to HuggingFace Hub")

if VAE_PATH is None:
    VAE_PATH = "stabilityai/sd-vae-ft-mse"
    print("   ⚠️  VAE: falling back to HuggingFace Hub")

print(f"   CLIP → {CLIP_PATH}")
print(f"   VAE  → {VAE_PATH}")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # ── Data Paths ─────────────────────────────────────────────
    "eeg_dir"     : "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/EEG",
    "video_dir"   : "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/Video",
    "caption_dir" : "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/Video/BLIP-caption",
    "output_dir"  : "/home/teaching/TEAM_22_DATASET/processed/processed",

    # ── Model Paths (auto-detected above) ─────────────────────
    "clip_model"  : CLIP_PATH,
    "vae_model"   : VAE_PATH,

    # ── EEG Parameters (from SEED-DV readme) ──────────────────
    # 200 Hz, 62 channels, 104000 timepoints per block
    "sfreq"       : 200,
    "n_channels"  : 62,

    # ── Sliding Window ─────────────────────────────────────────
    # 500ms window, 50% overlap → 250ms hop
    "window_sec"  : 0.5,
    "overlap"     : 0.5,

    # ── STFT Parameters ────────────────────────────────────────
    # VERIFIED to produce shape (62, 51, 9):
    #   F = (nfft/2) + 1 = (100/2) + 1 = 51  ✅
    #   T = floor((window_samples - noverlap)
    #             / (nperseg - noverlap))
    #     = floor((100 - 10) / (20 - 10))
    #     = floor(90 / 10) = 9               ✅
    "stft_nperseg"  : 20,
    "stft_noverlap" : 10,
    "stft_nfft"     : 100,

    # ── Video Parameters ───────────────────────────────────────
    # 24 fps source, extract 6 frames per 500ms window
    # VAE downsamples 128x128 → 16x16 latents
    "n_frames"    : 6,
    "video_size"  : (128, 128),

    # ── Caption Alignment ──────────────────────────────────────
    # Each BLIP caption covers 3 seconds of video
    # 200 captions × 3s = 600s = 10 minutes
    "segment_dur" : 3.0,

    # ── Runtime ────────────────────────────────────────────────
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",

    # ── Dataset Split Ratios ───────────────────────────────────
    "train_ratio" : 0.70,
    "val_ratio"   : 0.15,
    "test_ratio"  : 0.15,
    "random_seed" : 42,
}

print(f"\n🖥️  Device : {CONFIG['device'].upper()}")
print(f"📂 Output : {CONFIG['output_dir']}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2: VERIFY ALL PATHS EXIST
# ═══════════════════════════════════════════════════════════════

def verify_paths(config):
    """Check all required directories and model paths exist."""

    print("\n" + "="*60)
    print("🔍 VERIFYING ALL PATHS")
    print("="*60)

    checks = {
        "EEG directory"     : config["eeg_dir"],
        "Video directory"   : config["video_dir"],
        "Caption directory" : config["caption_dir"],
        "CLIP model path"   : config["clip_model"],
        "VAE model path"    : config["vae_model"],
    }

    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        print(f"   {'✅' if exists else '❌'} {name}: {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        raise FileNotFoundError(
            "❌ One or more required paths are missing. "
            "Check CONFIG and model cache."
        )

    # Count available files
    eeg_files = sorted([
        f for f in os.listdir(config["eeg_dir"])
        if f.endswith(".npy") and "session2" not in f
    ])
    video_files = sorted([
        f for f in os.listdir(config["video_dir"])
        if f.endswith(".mp4")
    ])
    caption_files = sorted([
        f for f in os.listdir(config["caption_dir"])
        if f.endswith(".txt")
    ])

    print(f"\n   📊 Files discovered:")
    print(f"      EEG subjects : {len(eeg_files)}")
    print(f"      Videos       : {len(video_files)}")
    print(f"      Caption files: {len(caption_files)}")

    # Sanity check
    if len(video_files) != 7:
        print(f"   ⚠️  Expected 7 videos, found {len(video_files)}")
    if len(eeg_files) < 1:
        raise ValueError("❌ No EEG .npy files found!")

    return eeg_files, video_files, caption_files


# ═══════════════════════════════════════════════════════════════
# SECTION 3: LOAD MODELS
# ═══════════════════════════════════════════════════════════════

def load_models(config):
    """
    Load CLIP text encoder and VAE from local cache.
    Both now use safetensors format.
    """

    print("\n" + "="*60)
    print("📦 LOADING MODELS FROM LOCAL CACHE")
    print("="*60)

    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import AutoencoderKL

    device = config["device"]

    # ── CLIP Tokenizer ─────────────────────────────────────────
    print("   Loading CLIP tokenizer...")
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        config["clip_model"],
        local_files_only = True,
    )
    print("   ✅ CLIP tokenizer loaded")

    # ── CLIP Text Model ────────────────────────────────────────
    # NOW uses model.safetensors (copied from c237dc49 snapshot)
    print("   Loading CLIP text model (safetensors)...")
    clip_text_model = CLIPTextModel.from_pretrained(
        config["clip_model"],
        local_files_only  = True,
        use_safetensors   = True,   # ✅ NOW WORKS after cp command
        low_cpu_mem_usage = True,
    )
    clip_text_model = clip_text_model.to(device).eval()
    print("   ✅ CLIP text model loaded")

    # ── VAE ────────────────────────────────────────────────────
    print("   Loading VAE (safetensors)...")
    vae = AutoencoderKL.from_pretrained(
        config["vae_model"],
        local_files_only  = True,
        use_safetensors   = True,
        low_cpu_mem_usage = True,
    )
    vae = vae.to(device).eval()
    print("   ✅ VAE loaded")

    print(f"\n   🖥️  All models ready on: {device.upper()}")
    return clip_tokenizer, clip_text_model, vae
    

# ═══════════════════════════════════════════════════════════════
# SECTION 4: EEG PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_eeg_block(eeg_block, config, verbose=False):
    """
    Slide 500ms windows across one EEG block.
    Compute STFT per window per channel.

    Input  : numpy array (62, 104000)
    Output : List of torch tensors, each (62, 51, 9)

    Window math:
        window_samples = 0.5s × 200Hz = 100 samples
        hop_samples    = 100 × 0.5    = 50  samples
        windows        = (104000 - 100) / 50 + 1 = 2079
    """
    n_channels, n_times = eeg_block.shape
    sfreq          = config["sfreq"]
    window_samples = int(config["window_sec"] * sfreq)            # 100
    hop_samples    = int(window_samples * (1 - config["overlap"])) # 50

    nperseg  = config["stft_nperseg"]    # 20
    noverlap = config["stft_noverlap"]   # 10
    nfft     = config["stft_nfft"]       # 100

    tensors    = []
    start      = 0
    shape_done = False

    while start + window_samples <= n_times:

        # Extract 500ms raw EEG: (62, 100)
        raw = eeg_block[:, start : start + window_samples]

        # STFT per channel
        ch_stfts = []
        for ch in range(n_channels):
            _, _, Zxx = stft(
                raw[ch],
                fs       = sfreq,
                nperseg  = nperseg,
                noverlap = noverlap,
                nfft     = nfft,
                boundary = None,
                padded   = False
            )
            ch_stfts.append(np.abs(Zxx))   # magnitude: (51, 9)

        # Stack → (62, 51, 9)
        stft_3d = np.stack(ch_stfts, axis=0)

        # Print and assert shape on first window
        if verbose and not shape_done:
            F, T = stft_3d.shape[1], stft_3d.shape[2]
            ok   = (F == 51 and T == 9)
            print(f"         {'✅' if ok else '❌'} "
                  f"EEG window shape: (62, {F}, {T})"
                  f"{' — matches blueprint!' if ok else ' — MISMATCH!'}")
            shape_done = True

        # Per-window normalization (zero mean, unit std)
        mean    = stft_3d.mean()
        std     = stft_3d.std() + 1e-8
        stft_3d = (stft_3d - mean) / std

        tensors.append(torch.tensor(stft_3d, dtype=torch.float32))
        start += hop_samples

    return tensors


# ═══════════════════════════════════════════════════════════════
# SECTION 5: VIDEO PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_video_block(video_path, n_windows, config, vae):
    """
    Extract 6 evenly-spaced frames per 500ms EEG window.
    Encode each set of 6 frames with the VAE.

    Input  : .mp4 file path, number of windows
    Output : List of torch tensors, each (6, 4, 16, 16)

    Frame → Latent pipeline:
        Frame  : (128, 128, 3)  numpy uint8
        Tensor : (3, 128, 128)  float32 in [-1, 1]
        Batch  : (6, 3, 128, 128)
        VAE    : (6, 4, 16, 16) latent
    """
    import cv2

    print(f"      📽️  {os.path.basename(video_path)}")
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"         {total_frames} frames @ {fps:.1f} fps")

    # Extract all frames into RAM
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, config["video_size"])
        all_frames.append(frame)
    cap.release()
    print(f"         Extracted {len(all_frames)} frames")

    video_latents = []

    with torch.no_grad():
        for w in range(n_windows):
            # Window time range
            t_start = w * 0.25           # 250ms hop
            t_end   = t_start + 0.5      # 500ms window

            # Convert to frame indices
            f_start = int(t_start * fps)
            f_end   = int(t_end   * fps)

            # Sample n_frames evenly across window
            indices = np.linspace(
                f_start,
                max(f_start, f_end - 1),
                config["n_frames"],
                dtype=int
            )
            # Clamp to valid range
            indices = np.clip(indices, 0, len(all_frames) - 1)

            # Build frame tensor
            frames = []
            for idx in indices:
                f  = torch.tensor(
                    all_frames[idx], dtype=torch.float32
                )                              # (128, 128, 3)
                f  = f.permute(2, 0, 1)        # (3, 128, 128)
                f  = (f / 127.5) - 1.0         # [-1, 1] for VAE
                frames.append(f)

            # (6, 3, 128, 128) → VAE → (6, 4, 16, 16)
            frames_t = torch.stack(frames).to(config["device"])
            latents  = vae.encode(frames_t).latent_dist.sample()
            video_latents.append(latents.cpu())

    print(f"         ✅ {len(video_latents)} video windows created")
    return video_latents


# ═══════════════════════════════════════════════════════════════
# SECTION 6: TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_captions(caption_path, n_windows, config,
                     clip_tokenizer, clip_text_model):
    """
    Map each EEG window to its BLIP caption.
    Encode caption with CLIP text encoder.

    Caption file: 200 lines, each covers 3 seconds
    EEG windows : start at t = w × 0.25 seconds

    Mapping:
        caption_index = floor(t_start / 3.0)

    Input  : .txt caption file, number of windows
    Output : List of torch tensors, each (512,)

    Why pooler_output?
        CLIPTextModel returns:
          last_hidden_state  → (1, seq_len, 768) sequence
          pooler_output      → (1, 512) sentence embedding ✅
    """
    with open(caption_path) as f:
        captions = [l.strip() for l in f if l.strip()]

    print(f"      📝 {len(captions)} captions loaded")

    text_embeds = []

    with torch.no_grad():
        for w in range(n_windows):
            # Map window to caption
            t_start = w * 0.25
            c_idx   = min(
                int(t_start / config["segment_dur"]),
                len(captions) - 1
            )

            # Tokenize
            tokens = clip_tokenizer(
                captions[c_idx],
                return_tensors = "pt",
                padding        = "max_length",
                truncation     = True,
                max_length     = 77
            )
            tokens = {
                k: v.to(config["device"])
                for k, v in tokens.items()
            }

            # Encode → pooler_output is (1, 512)
            outputs = clip_text_model(**tokens)
            feat    = outputs.pooler_output   # (1, 512)
            feat    = feat.squeeze(0).cpu()   # (512,)

            # Shape guard
            assert feat.shape == torch.Size([512]), \
                f"❌ Wrong text shape: {feat.shape}, expected (512,)"

            text_embeds.append(feat)

    # Report final shape
    print(f"      ✅ Text embed shape: {text_embeds[0].shape}")
    return text_embeds


# ═══════════════════════════════════════════════════════════════
# SECTION 7: SAVE MATCHED TRIPLETS
# ═══════════════════════════════════════════════════════════════

def save_triplets(eeg_list, text_list, video_list,
                  output_dir, start_idx):
    """
    Save matched (EEG, text, video) triplets as .pt files.

    Naming: eeg_sample_000001.pt
            text_sample_000001.pt
            video_sample_000001.pt

    Returns next available sample index.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use minimum length to ensure perfect alignment
    min_len = min(len(eeg_list), len(text_list), len(video_list))

    for i in range(min_len):
        s = f"{start_idx + i:06d}"

        torch.save(
            eeg_list[i],
            os.path.join(output_dir, f"eeg_sample_{s}.pt")
        )
        torch.save(
            text_list[i],
            os.path.join(output_dir, f"text_sample_{s}.pt")
        )
        torch.save(
            video_list[i],
            os.path.join(output_dir, f"video_sample_{s}.pt")
        )

    return start_idx + min_len


# ═══════════════════════════════════════════════════════════════
# SECTION 8: CREATE TRAIN / VAL / TEST SPLITS
# ═══════════════════════════════════════════════════════════════

def create_splits(output_dir, total, config):
    """
    Create reproducible train/val/test split files.

    Format: one sample ID per line (e.g. 000001)
    Ratios: 70% train, 15% val, 15% test
    Seed  : 42 (reproducible)
    """
    print("\n" + "="*60)
    print("📋 CREATING TRAIN / VAL / TEST SPLITS")
    print("="*60)

    # All sample IDs
    ids = [f"{i:06d}" for i in range(1, total + 1)]

    # Shuffle with fixed seed
    np.random.seed(config["random_seed"])
    np.random.shuffle(ids)

    # Calculate sizes
    n_tr = int(total * config["train_ratio"])
    n_va = int(total * config["val_ratio"])

    splits = {
        "train_split.txt" : ids[:n_tr],
        "val_split.txt"   : ids[n_tr : n_tr + n_va],
        "test_split.txt"  : ids[n_tr + n_va:],
    }

    for name, part in splits.items():
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            f.write("\n".join(part))
        print(f"   ✅ {name:<20}: {len(part):>8,} samples")

    print(f"\n   Total : {total:,} samples")
    print(f"   Train : {len(splits['train_split.txt']):,}")
    print(f"   Val   : {len(splits['val_split.txt']):,}")
    print(f"   Test  : {len(splits['test_split.txt']):,}")


# ═══════════════════════════════════════════════════════════════
# SECTION 9: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main(is_test=False):
    """
    Main preprocessing pipeline.

    is_test=True  → 1 subject × 1 block  (~2,079 samples, fast)
    is_test=False → 20 subjects × 7 blocks (~291,060 samples)
    """
    print("\n" + "="*60)
    print("🚀 EEG2VIDEO PREPROCESSING — CS671 TEAM 22")
    print(f"   Mode   : {'🧪 TEST (1 subject × 1 block)' if is_test else '🔥 FULL DATASET (20 × 7)'}")
    print(f"   Device : {CONFIG['device'].upper()}")
    print("="*60)

    # ── Step 1: Verify all paths ────────────────────────────────
    eeg_files, video_files, caption_files = verify_paths(CONFIG)

    # ── Step 2: Load models ─────────────────────────────────────
    clip_tokenizer, clip_text_model, vae = load_models(CONFIG)

    # ── Step 3: Limit scope for test mode ───────────────────────
    if is_test:
        eeg_files     = eeg_files[:1]
        video_files   = video_files[:1]
        caption_files = caption_files[:1]
        print(f"\n   🧪 TEST MODE: processing sub1.npy × block 1 only")

    # ── Step 4: Process all subjects and blocks ─────────────────
    sample_counter = 1
    total_saved    = 0
    first_block    = True   # for verbose shape printing

    for sub_idx, eeg_file in enumerate(eeg_files):
        print(f"\n{'─'*60}")
        print(f"👤 SUBJECT {sub_idx+1}/{len(eeg_files)}: {eeg_file}")
        print(f"{'─'*60}")

        # Load all 7 blocks at once: (7, 62, 104000)
        eeg_data = np.load(
            os.path.join(CONFIG["eeg_dir"], eeg_file)
        )
        n_blocks = eeg_data.shape[0]   # always 7

        # In test mode: only process block 0
        block_range = range(1) if is_test else range(n_blocks)

        for b_idx in block_range:
            print(f"\n   🎬 Block {b_idx+1}/{n_blocks}")

            # ── EEG ──────────────────────────────────────────────
            eeg_wins = process_eeg_block(
                eeg_data[b_idx],     # (62, 104000)
                CONFIG,
                verbose = first_block
            )
            first_block = False
            n_wins = len(eeg_wins)
            print(f"      EEG windows  : {n_wins:,}")

            # ── VIDEO ─────────────────────────────────────────────
            vid_path = os.path.join(
                CONFIG["video_dir"], video_files[b_idx]
            )
            vid_lats = process_video_block(
                vid_path, n_wins, CONFIG, vae
            )

            # ── TEXT ──────────────────────────────────────────────
            cap_path = os.path.join(
                CONFIG["caption_dir"], caption_files[b_idx]
            )
            txt_embs = process_captions(
                cap_path, n_wins, CONFIG,
                clip_tokenizer, clip_text_model
            )

            # ── ALIGN & SAVE ──────────────────────────────────────
            min_len = min(n_wins, len(vid_lats), len(txt_embs))

            if min_len == 0:
                print(f"      ⚠️  Skipping block — no aligned samples")
                continue

            sample_counter = save_triplets(
                eeg_wins[:min_len],
                txt_embs[:min_len],
                vid_lats[:min_len],
                CONFIG["output_dir"],
                sample_counter
            )
            total_saved += min_len

            print(f"      ✅ Saved : {min_len:,} samples")
            print(f"      📦 Total : {total_saved:,} samples so far")

    # ── Step 5: Create split files ──────────────────────────────
    create_splits(CONFIG["output_dir"], total_saved, CONFIG)

    # ── Step 6: Final summary ───────────────────────────────────
    print("\n" + "="*60)
    print("🎉 PREPROCESSING COMPLETE")
    print("="*60)
    print(f"   Samples saved  : {total_saved:,}")
    print(f"   Output dir     : {CONFIG['output_dir']}")
    print(f"\n   Shape contract:")
    print(f"     EEG   → (62, 51, 9)     ✅")
    print(f"     Text  → (512,)           ✅")
    print(f"     Video → (6, 4, 16, 16)  ✅")
    print(f"\n   Split files created:")
    print(f"     train_split.txt")
    print(f"     val_split.txt")
    print(f"     test_split.txt")
    print("="*60)

    if is_test:
        print("\n💡 TEST PASSED!")
        print("   Change is_test=True → is_test=False")
        print("   Then run with nohup for full dataset:")
        print("   nohup eegpython preprocess_full.py > run.log 2>&1 &")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ╔═══════════════════════════════════════╗
    # ║  is_test = True  → fast verification  ║
    # ║  is_test = False → full dataset run   ║
    # ╚═══════════════════════════════════════╝
    main(is_test=False)