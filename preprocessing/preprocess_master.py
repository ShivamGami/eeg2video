"""
╔══════════════════════════════════════════════════════════════╗
║    COMPLETE PREPROCESSING PIPELINE — EEG2VIDEO CS671         ║
║    Dataset: SEED-DV (20 subjects × 7 videos)                 ║
║                                                              ║
║  TARGET SHAPES:                                              ║
║    • eeg_sample_XXXXXX.pt   → (62, 51, 9)                    ║
║    • text_sample_XXXXXX.pt  → (512,)                         ║
║    • video_sample_XXXXXX.pt → (6, 4, 16, 16)                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import torch
from scipy.signal import stft
import warnings
warnings.filterwarnings("ignore")

# ── Fix HuggingFace torch.load vulnerability error ────────────
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION (Corrected Paths & Splits)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────
    "eeg_dir"       : "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/EEG",
    "video_dir"     : "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/video",
    "caption_dir"   : "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/video/BLIP-caption",
    "output_dir"    : "/home/teaching/TEAM_22_DATASET/processed",

    # ── EEG Parameters (from readme: 200Hz, 62ch) ─────────────
    "sfreq"         : 200,
    "n_channels"    : 62,

    # ── Sliding Window ─────────────────────────────────────────
    "window_sec"    : 0.5,          # 500ms
    "overlap"       : 0.5,          # 50% overlap

    # ── STFT (VERIFIED: gives exactly (51, 9)) ─────────────────
    "stft_nperseg"  : 20,
    "stft_noverlap" : 10,
    "stft_nfft"     : 100,

    # ── Video ──────────────────────────────────────────────────
    "n_frames"      : 6,
    "video_size"    : (128, 128),
    "video_fps"     : 24,

    # ── Caption Alignment ──────────────────────────────────────
    "segment_dur"   : 3.0,        # Each BLIP caption = 3 seconds

    # ── Models ─────────────────────────────────────────────────
    "clip_model"    : "openai/clip-vit-base-patch32",
    "vae_model"     : "stabilityai/sd-vae-ft-mse",
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",

    # ── Split Ratios ───────────────────────────────────────────
    "train_ratio"   : 0.80,
    "val_ratio"     : 0.10,
    "test_ratio"    : 0.10,
    "random_seed"   : 42,
}


# ═══════════════════════════════════════════════════════════════
# SECTION 1: LOAD MODELS SAFELY
# ═══════════════════════════════════════════════════════════════

def load_models(config):
    print("\n" + "="*60)
    print("📦 LOADING MODELS (safetensors mode)")
    print("="*60)

    from transformers import CLIPProcessor, CLIPModel
    from diffusers import AutoencoderKL

    device = config["device"]

    # ── Load CLIP ──────────────────────────────────────────────
    print("   Loading CLIP-B/32...")
    try:
        clip_model = CLIPModel.from_pretrained(
            config["clip_model"],
            use_safetensors = True,
            low_cpu_mem_usage = True,
        )
        print("   ✅ CLIP loaded via safetensors")

    except Exception as e1:
        print(f"   ⚠️  safetensors failed: {e1}")
        try:
            import transformers
            transformers.utils.import_utils._torch_version = "2.6.0"
            clip_model = CLIPModel.from_pretrained(
                config["clip_model"],
                low_cpu_mem_usage = True,
            )
            print("   ✅ CLIP loaded via fallback")
        except Exception as e2:
            print(f"   ❌ CLIP load failed: {e2}")
            raise

    clip_proc = CLIPProcessor.from_pretrained(config["clip_model"])
    clip_model = clip_model.to(device).eval()

    # ── Load VAE ───────────────────────────────────────────────
    print("   Loading VAE (sd-vae-ft-mse)...")
    try:
        vae = AutoencoderKL.from_pretrained(
            config["vae_model"],
            use_safetensors = True,
            low_cpu_mem_usage = True,
        )
        print("   ✅ VAE loaded via safetensors")

    except Exception as e1:
        print(f"   ⚠️  VAE safetensors failed: {e1}")
        try:
            vae = AutoencoderKL.from_pretrained(
                config["vae_model"],
                low_cpu_mem_usage = True,
            )
            print("   ✅ VAE loaded via fallback")
        except Exception as e2:
            print(f"   ❌ VAE load failed: {e2}")
            raise

    vae = vae.to(device).eval()

    print(f"\n   ✅ ALL MODELS LOADED on {device.upper()}")
    return clip_model, clip_proc, vae


# ═══════════════════════════════════════════════════════════════
# SECTION 2: EEG PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_eeg_block(eeg_block, config, verbose=False):
    n_channels, n_times = eeg_block.shape
    sfreq          = config["sfreq"]
    window_samples = int(config["window_sec"] * sfreq)  # 100
    hop_samples    = int(window_samples * (1 - config["overlap"]))  # 50

    nperseg  = config["stft_nperseg"]   # 20
    noverlap = config["stft_noverlap"]  # 10
    nfft     = config["stft_nfft"]      # 100

    window_tensors = []
    start          = 0
    first_window   = True

    while start + window_samples <= n_times:
        raw_window = eeg_block[:, start : start + window_samples]

        channel_stfts = []
        for ch in range(n_channels):
            _, _, Zxx = stft(
                raw_window[ch],
                fs       = sfreq,
                nperseg  = nperseg,
                noverlap = noverlap,
                nfft     = nfft,
                boundary = None,
                padded   = False
            )
            channel_stfts.append(np.abs(Zxx))   # (F=51, T=9)

        stft_3d = np.stack(channel_stfts, axis=0)

        if first_window and verbose:
            print(f"        ✅ EEG window shape: {stft_3d.shape}")
            assert stft_3d.shape == (62, 51, 9), \
                f"Shape mismatch! Got {stft_3d.shape}, expected (62, 51, 9)"
            first_window = False

        mean = stft_3d.mean()
        std  = stft_3d.std() + 1e-8
        stft_3d = (stft_3d - mean) / std

        window_tensors.append(torch.tensor(stft_3d, dtype=torch.float32))
        start += hop_samples

    return window_tensors


# ═══════════════════════════════════════════════════════════════
# SECTION 3: VIDEO PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_video_block(video_path, n_windows, config, vae):
    import cv2

    print(f"      📽️  Loading video: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"      ❌ Cannot open video!")
        return []

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"         FPS={fps:.1f}, Total frames={total_frm}")

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
            t_start = w * 0.25      # 250ms hop
            t_end   = t_start + 0.5 # 500ms window

            f_start = int(t_start * fps)
            f_end   = int(t_end   * fps)
            indices = np.linspace(f_start, max(f_start, f_end - 1),
                                  config["n_frames"], dtype=int)

            indices = np.clip(indices, 0, len(all_frames) - 1)

            frames_list = []
            for idx in indices:
                f  = all_frames[idx]                              
                ft = torch.tensor(f, dtype=torch.float32)
                ft = ft.permute(2, 0, 1)                          
                ft = (ft / 127.5) - 1.0                           
                frames_list.append(ft)

            frames_tensor = torch.stack(frames_list).to(config["device"])
            latents = vae.encode(frames_tensor).latent_dist.sample()
            video_latents.append(latents.cpu())

    print(f"         ✅ {len(video_latents)} video windows created")
    return video_latents


# ═══════════════════════════════════════════════════════════════
# SECTION 4: TEXT (CLIP) PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_captions(caption_path, n_windows, config, clip_model, clip_proc):
    with open(caption_path) as f:
        captions = [l.strip() for l in f if l.strip()]

    print(f"      Loaded {len(captions)} captions")
    text_embeds = []

    with torch.no_grad():
        for w in range(n_windows):
            t     = w * 0.25
            c_idx = min(int(t / config["segment_dur"]), len(captions) - 1)

            inputs = clip_proc(
                text           = [captions[c_idx]],
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = 77
            )
            inputs = {k: v.to(config["device"]) for k, v in inputs.items()}

            feat = clip_model.get_text_features(**inputs)  
            feat = feat.squeeze(0).cpu()                   

            assert feat.shape == torch.Size([512]), \
                f"Wrong shape: {feat.shape}, expected (512,)"

            text_embeds.append(feat)

    print(f"      ✅ {len(text_embeds)} embeddings, shape: {text_embeds[0].shape}")
    return text_embeds
    
# ═══════════════════════════════════════════════════════════════
# SECTION 5: SAVE SAMPLES
# ═══════════════════════════════════════════════════════════════

def save_triplet_batch(eeg_list, text_list, video_list, output_dir, start_idx):
    min_len = min(len(eeg_list), len(text_list), len(video_list))
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min_len):
        idx_str = f"{start_idx + i:06d}"
        torch.save(eeg_list[i],   os.path.join(output_dir, f"eeg_sample_{idx_str}.pt"))
        torch.save(text_list[i],  os.path.join(output_dir, f"text_sample_{idx_str}.pt"))
        torch.save(video_list[i], os.path.join(output_dir, f"video_sample_{idx_str}.pt"))

    return start_idx + min_len  


# ═══════════════════════════════════════════════════════════════
# SECTION 6: CREATE TRAIN / VAL / TEST SPLITS
# ═══════════════════════════════════════════════════════════════

def create_splits(output_dir, total_samples, config):
    print("\n" + "="*60)
    print("📋 CREATING TRAIN / VAL / TEST SPLITS")
    print("="*60)

    all_ids = [f"sample_{i:06d}" for i in range(1, total_samples + 1)]

    np.random.seed(config["random_seed"])
    np.random.shuffle(all_ids)

    n_train = int(len(all_ids) * config["train_ratio"])
    n_val   = int(len(all_ids) * config["val_ratio"])

    splits = {
        "train_split.txt" : all_ids[:n_train],
        "val_split.txt"   : all_ids[n_train : n_train + n_val],
        "test_split.txt"  : all_ids[n_train + n_val:],
    }

    for name, ids in splits.items():
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            f.write("\n".join(ids))
        print(f"   ✅ {name}: {len(ids):,} samples")

    print(f"\n   Total: {total_samples:,} samples")


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main(is_test=False):
    print("\n" + "="*60)
    print("🚀 EEG2VIDEO PREPROCESSING — CS671 TEAM 22")
    print(f"   Mode: {'TEST (1 subject, 1 block)' if is_test else 'FULL DATASET'}")
    print(f"   Device: {CONFIG['device'].upper()}")
    print("="*60)

    clip_model, clip_proc, vae = load_models(CONFIG)

    eeg_files = sorted([
        f for f in os.listdir(CONFIG["eeg_dir"])
        if f.endswith(".npy") and "session2" not in f
    ])
    video_files = sorted([
        f for f in os.listdir(CONFIG["video_dir"])
        if f.endswith(".mp4")
    ])
    caption_files = sorted([
        f for f in os.listdir(CONFIG["caption_dir"])
        if f.endswith(".txt")
    ])

    print(f"\n   EEG subjects : {len(eeg_files)}")
    print(f"   Videos       : {len(video_files)}")
    print(f"   Caption files: {len(caption_files)}")

    if is_test:
        eeg_files     = eeg_files[:1]
        video_files   = video_files[:1]
        caption_files = caption_files[:1]

    sample_counter = 1
    total_saved    = 0

    for sub_idx, eeg_file in enumerate(eeg_files):
        print(f"\n{'─'*60}")
        print(f"👤 SUBJECT {sub_idx+1}/{len(eeg_files)}: {eeg_file}")
        print(f"{'─'*60}")

        eeg_data = np.load(os.path.join(CONFIG["eeg_dir"], eeg_file))

        n_blocks = eeg_data.shape[0]
        blocks_to_process = range(1 if is_test else n_blocks)

        for b_idx in blocks_to_process:
            print(f"\n   🎬 BLOCK {b_idx+1}/{n_blocks}")

            video_path   = os.path.join(CONFIG["video_dir"],   video_files[b_idx])
            caption_path = os.path.join(CONFIG["caption_dir"], caption_files[b_idx])

            eeg_windows = process_eeg_block(
                eeg_data[b_idx], CONFIG, verbose=(total_saved == 0)
            )
            n_windows = len(eeg_windows)
            print(f"      EEG windows: {n_windows}")

            vid_latents = process_video_block(
                video_path, n_windows, CONFIG, vae
            )

            txt_embeds = process_captions(
                caption_path, n_windows, CONFIG, clip_model, clip_proc
            )

            min_len = min(n_windows, len(vid_latents), len(txt_embeds))
            print(f"\n      Saving {min_len} aligned triplets...")

            sample_counter = save_triplet_batch(
                eeg_windows[:min_len],
                txt_embeds[:min_len],
                vid_latents[:min_len],
                CONFIG["output_dir"],
                sample_counter
            )
            total_saved += min_len
            print(f"      ✅ Block done. Cumulative total: {total_saved:,}")

    create_splits(CONFIG["output_dir"], total_saved, CONFIG)

    print("\n" + "="*60)
    print("🎉 PREPROCESSING COMPLETE")
    print("="*60)
    print(f"   Total samples: {total_saved:,}")
    print(f"   Output dir:    {CONFIG['output_dir']}")
    print(f"   EEG shape:     (62, 51, 9)")
    print(f"   Text shape:    (512,)")
    print(f"   Video shape:   (6, 4, 16, 16)")
    print("="*60)


if __name__ == "__main__":
    # Test mode is ON: Processes only 1 subject & 1 block. 
    # Change to False when you want to process all data!
    main(is_test=True)