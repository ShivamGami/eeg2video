"""
train_videoldm.py — Sub-team 2 VideoLDM Fine-tuning (FINAL v5)
===============================================================
WHY THIS VERSION IS CORRECT:

  Previous runs used real_vae_latents_50k.pt which had two bugs:
    1. Wrong frame grouping — (300000,4,32,32) reshaped to
       (50000,6,4,32,32) puts UNRELATED frames together per sample.
       UNet trained on random frame sequences = learned to ignore content.
    2. Wrong scale — std=0.18 vs real VAE std ~0.9-5.0.
       VAE decoder needs correct scale to produce sharp images.

  This version loads video_sample_XXXXXX.pt directly from
  processed/processed/. Each file is one (6, 4, 16, 16) tensor —
  6 frames of the SAME 2-second clip, correctly VAE-encoded.
  text_sample_XXXXXX.pt is paired 1-to-1 by sample ID.

  NOTE: video shape changed from (6,4,32,32) to (6,4,16,16).
  The UNet spatial input changes but architecture still works —
  SD UNet accepts any (H,W) divisible by 8. EEGProjection
  is unchanged (still 512→77×768).

EXPECTED TRAINING (RTX A5000, 291k samples, 203k train):
  Per-file loading is slower than mmap tensors.
  TRAIN_SUBSET=20000, num_workers=4 → ~40 min/epoch
  5 epochs = ~3.5 hrs total
"""

import os
os.environ["WANDB_API_KEY"] = "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"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random
import wandb
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.utils import logging as diffusers_logging

from subteam2_dataset import VideoLDMDataset
from projection import EEGProjection

diffusers_logging.set_verbosity_error()
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ── Paths ──────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = "./modelscope_weights"
SAVE_DIR    = "./checkpoints"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 2       # can use batch>1 now that data loads correctly
GRAD_ACCUM    = 4       # effective batch = 8 samples
LR            = 1e-4
NUM_EPOCHS    = 10
SAVE_EVERY    = 5       # save every epoch
MAX_GRAD_NORM = 1.0
NUM_FRAMES    = 6

# 20,000 samples × 5 epochs = 100,000 gradient steps
# ~40 min/epoch on A5000 with num_workers=4
TRAIN_SUBSET = 20_0000
VAL_SUBSET   =    500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ── Pre-flight verification ────────────────────────────────────────────────────

def preflight_check():
    """Verify data quality before wasting GPU time."""
    print("="*60)
    print("PRE-FLIGHT CHECKS")
    print("="*60)

    import glob
    files = glob.glob("/home/teaching/TEAM_22_DATASET/processed/processed/video_sample_*.pt")
    print(f"  video_sample files : {len(files):,}")
    if len(files) == 0:
        print("  [FATAL] No video files found! Check PROCESSED_DIR.")
        return False

    # Check first file shape and stats
    v = torch.load(files[0], map_location="cpu")
    print(f"  video shape        : {v.shape}  (expected (6,4,16,16))")
    print(f"  video std          : {v.std():.4f}  (healthy if > 0.5)")
    print(f"  video range        : [{v.min():.2f}, {v.max():.2f}]")

    if v.std() < 0.1:
        print("  [FATAL] Video latents collapsed (std < 0.1)!")
        return False

    # Check text file
    sid = os.path.basename(files[0]).replace("video_sample_","").replace(".pt","")
    t = torch.load(f"/home/teaching/TEAM_22_DATASET/processed/processed/text_sample_{sid}.pt",
                   map_location="cpu").flatten()
    print(f"  text shape         : {t.shape}  (expected (512,))")

    print("  [OK] Data quality verified\n")
    return True


# ── Model loading ──────────────────────────────────────────────────────────────

def load_unet_from_modelscope(weights_dir: str) -> UNet2DConditionModel:
    for root, dirs, files in os.walk(weights_dir):
        if "config.json" in files and "unet" in root:
            print(f"  Found UNet at: {root}")
            return UNet2DConditionModel.from_pretrained(root, torch_dtype=torch.float16)
    print("  ModelScope UNet not found, falling back to SD v1.5...")
    return UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=torch.float16,
    )


def freeze_unet_except_cross_attention(unet: nn.Module) -> int:
    for param in unet.parameters():
        param.requires_grad = False
    trainable = 0
    for name, module in unet.named_modules():
        if "attn2" in name:
            for param in module.parameters():
                param.requires_grad = True
                trainable += param.numel()
    print(f"  UNet cross-attention trainable params: {trainable:,}")
    return trainable


def get_trainable_params(unet, projection):
    params = [p for p in projection.parameters() if p.requires_grad]
    params += [p for p in unet.parameters() if p.requires_grad]
    return params


# ── Forward pass ───────────────────────────────────────────────────────────────

def run_forward(unet, projection, scheduler, visual, text):
    """
    All fp32, no autocast.

    visual : (B, 6, 4, 16, 16) — REAL VAE latents, correct scale
    text   : (B, 512)          — REAL CLIP embeddings, diverse
    """
    B, F, C, H, W = visual.shape   # H=W=16 now

    # (B, 512) → (B, 77, 768)
    encoder_hidden_states = projection(text)

    # (B*F, 77, 768)
    enc_hs = encoder_hidden_states.repeat_interleave(F, dim=0)

    # (B*F, 4, 16, 16)
    frame_latents = visual.view(B * F, C, H, W)

    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (B,), device=DEVICE
    ).long().repeat_interleave(F)

    noise  = torch.randn_like(frame_latents)
    noisy  = scheduler.add_noise(frame_latents, noise, timesteps)

    noise_pred = unet(noisy, timesteps, encoder_hidden_states=enc_hs).sample

    return torch.nn.functional.mse_loss(noise_pred, noise)


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    if not preflight_check():
        print("[ABORTED] Fix data issues first.")
        return

    wandb.init(project="eeg2video-subteam2",
               name=f"videoldm-v5-CORRECT-processed-subset{TRAIN_SUBSET}")

    # ── Datasets ───────────────────────────────────────────────────────────
    print("Loading datasets from processed/processed/ ...")
    train_ds_full = VideoLDMDataset(split='train')
    val_ds_full   = VideoLDMDataset(split='val')

    n_train = len(train_ds_full)
    n_val   = len(val_ds_full)

    actual_train = min(TRAIN_SUBSET, n_train)
    actual_val   = min(VAL_SUBSET,   n_val)

    train_indices = random.sample(range(n_train), actual_train)
    val_indices   = random.sample(range(n_val),   actual_val)

    train_ds = Subset(train_ds_full, train_indices)
    val_ds   = Subset(val_ds_full,   val_indices)

    # num_workers=8 for parallel file loading (many small .pt files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4,
                              pin_memory=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4,
                              pin_memory=True)

    steps_per_epoch = len(train_loader)
    est_min = steps_per_epoch / 5.0 / 60   # ~5 it/s with workers
    print(f"\nTrain subset : {actual_train:,}  batches: {steps_per_epoch:,}")
    print(f"Val subset   : {actual_val:,}   batches: {len(val_loader):,}")
    print(f"Estimated    : ~{est_min:.0f} min/epoch × {NUM_EPOCHS} = ~{est_min*NUM_EPOCHS/60:.1f} hrs\n")

    # ── Models ─────────────────────────────────────────────────────────────
    print("Loading UNet...")
    unet = load_unet_from_modelscope(WEIGHTS_DIR)
    freeze_unet_except_cross_attention(unet)
    unet = unet.float().to(DEVICE)

    print("Loading noise scheduler...")
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )

    print("Building EEGProjection...")
    projection = EEGProjection(input_dim=512, output_dim=768, seq_len=77).to(DEVICE)
    print(f"  EEGProjection params: {sum(p.numel() for p in projection.parameters()):,}")

    trainable_params = get_trainable_params(unet, projection)
    print(f"Total trainable : {sum(p.numel() for p in trainable_params):,}\n")

    optimizer    = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):

        unet.train()
        projection.train()
        t_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{NUM_EPOCHS} [train]")
        for step, batch in enumerate(pbar):
            visual = batch['video_latent'].to(DEVICE)   # (B, 6, 4, 16, 16)
            text   = batch['text_embed'].to(DEVICE)     # (B, 512)

            loss = run_forward(unet, projection, scheduler, visual, text) / GRAD_ACCUM

            if torch.isnan(loss):
                print(f"\n[FATAL] NaN at epoch {epoch} step {step}. Aborting.")
                return

            loss.backward()
            t_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                optimizer.step()
                if (step + 1) % 5000 == 0:
                    torch.save(projection.state_dict(), f"checkpoints/projection_step_{step+1}.pth")
                    print(f"\n[HOT-FIX] Saved immediate checkpoint at step {step+1}")
                optimizer.zero_grad()

            if step % 200 == 0:
                pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")

        scheduler_lr.step()
        avg_t = t_loss / len(train_loader)

        unet.eval()
        projection.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch}/{NUM_EPOCHS} [val]", leave=False):
                visual = batch['video_latent'].to(DEVICE)
                text   = batch['text_embed'].to(DEVICE)
                v_loss += run_forward(unet, projection, scheduler, visual, text).item()

        avg_v      = v_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({"epoch": epoch, "train_loss": avg_t,
                   "val_loss": avg_v, "lr": current_lr})
        print(f"Epoch {epoch:>2}/{NUM_EPOCHS} | "
              f"Train: {avg_t:.5f} | Val: {avg_v:.5f} | LR: {current_lr:.2e}")

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            torch.save(projection.state_dict(), f"{SAVE_DIR}/projection_best.pth")
            unet.save_pretrained(f"{SAVE_DIR}/unet_best")
            print(f"  [OK] Best saved (val={best_val_loss:.5f})")

        torch.save(projection.state_dict(), f"{SAVE_DIR}/projection_epoch{epoch}.pth")
       # unet.save_pretrained(f"{SAVE_DIR}/unet_epoch{epoch}")
        print(f"  [OK] Epoch {epoch} checkpoint saved")

    wandb.finish()
    print(f"\nTraining done. Best val_loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    train()
