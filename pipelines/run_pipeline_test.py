"""
run_pipeline_test.py — Sub-team 5 Pipeline Sanity Test
=======================================================
Run this BEFORE inference.py to verify every component loads
and produces the right tensor shapes.

Usage:
    python run_pipeline_test.py

Expected output (if everything is correct):
    [PASS] visual_latents.pt   shape (291060, 6, 4, 32, 32)
    [PASS] text_embeddings.pt  shape (291060, 512)
    [PASS] is_fast.pt          shape (291060, 1)
    [PASS] UNet loaded
    [PASS] EEGProjection loaded
    [PASS] DANA output shape (1, 6, 4, 32, 32) beta=0.35 (slow)
    [PASS] Projection output (1, 77, 768)
    [PASS] UNet forward pass output (6, 4, 32, 32)
    [PASS] VAE decode output (6, 256, 256, 3)
    All checks passed — pipeline is ready for inference.py
"""

import torch
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
VISUAL_PATH    = "/home/teaching/manan_workspace/eeg2video-cs671/subteam3_vit/visual_latents.pt"
TEXT_PATH      = "/home/teaching/vishal_workspace/eeg2video-cs671/text_embeddings.pt"
FAST_PATH      = "/home/teaching/vishal_workspace/eeg2video-cs671/is_fast.pt"
CHECKPOINT_DIR = "/home/teaching/vishal_workspace/eeg2video-cs671/subteam2_videoldm/checkpoints"
VAE_MODEL_ID   = "runwayml/stable-diffusion-v1-5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS   = "[PASS]"
FAIL   = "[FAIL]"

errors = []

def check(cond, msg):
    if cond:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        errors.append(msg)


print("=" * 55)
print(" Sub-team 5 Pipeline Sanity Check")
print("=" * 55)
print(f"Device: {DEVICE}\n")

# ── 1. Tensor files ───────────────────────────────────────────────────────────
print("[1] Checking pre-computed tensors...")
try:
    visual = torch.load(VISUAL_PATH, map_location="cpu", mmap=True)
    check(visual.shape == torch.Size([291060, 6, 4, 32, 32]),
          f"visual_latents shape {tuple(visual.shape)}")
except Exception as e:
    check(False, f"visual_latents.pt — {e}")

try:
    text = torch.load(TEXT_PATH, map_location="cpu")
    check(text.shape == torch.Size([291060, 512]),
          f"text_embeddings shape {tuple(text.shape)}")
    check(not text.isnan().any(), "text_embeddings — no NaN")
except Exception as e:
    check(False, f"text_embeddings.pt — {e}")

try:
    fast = torch.load(FAST_PATH, map_location="cpu")
    check(fast.shape[0] == 291060, f"is_fast shape {tuple(fast.shape)}")
    check(set(fast.unique().tolist()).issubset({0.0, 1.0}),
          f"is_fast values are {{0.0, 1.0}}")
except Exception as e:
    check(False, f"is_fast.pt — {e}")

# ── 2. Sub-team 2 checkpoint ──────────────────────────────────────────────────
print("\n[2] Checking Sub-team 2 checkpoint...")
unet_path = os.path.join(CHECKPOINT_DIR, "unet_best")
proj_path = os.path.join(CHECKPOINT_DIR, "projection_best.pth")
check(os.path.exists(unet_path),     f"unet_best/ exists at {unet_path}")
check(os.path.exists(proj_path),     f"projection_best.pth exists")

# ── 3. Model loading ──────────────────────────────────────────────────────────
print("\n[3] Loading models (this takes ~30 seconds)...")
unet = proj = vae = None
try:
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    check(True, "UNet loaded")
except Exception as e:
    check(False, f"UNet load failed: {e}")

try:
    from projection import EEGProjection
    proj = EEGProjection(512, 768, 77).to(DEVICE)
    proj.load_state_dict(torch.load(proj_path, map_location=DEVICE))
    proj.eval()
    check(True, "EEGProjection loaded")
except Exception as e:
    check(False, f"EEGProjection load failed: {e}")

try:
    from dana import DANAModule
    dana = DANAModule().to(DEVICE)
    check(True, "DANA module initialised")
except Exception as e:
    check(False, f"DANA init failed: {e}")

try:
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    ).to(DEVICE).eval()
    check(True, "VAE loaded")
except Exception as e:
    check(False, f"VAE load failed: {e}")

# ── 4. Forward pass shapes ────────────────────────────────────────────────────
print("\n[4] Checking forward pass shapes with sample idx=0...")

if unet and proj and vae:
    with torch.no_grad():
        try:
            # Sample data
            vis  = visual[0].unsqueeze(0).to(DEVICE)   # (1, 6, 4, 32, 32)
            txt  = text[0].unsqueeze(0).to(DEVICE)     # (1, 512)
            f    = fast[0].unsqueeze(0).to(DEVICE)     # (1, 1)

            # DANA
            noised, betas = dana(vis, f)
            check(noised.shape == torch.Size([1, 6, 4, 32, 32]),
                  f"DANA output {tuple(noised.shape)} beta={betas[0]:.2f}")

            # Projection
            enc_hs = proj(txt)
            check(enc_hs.shape == torch.Size([1, 77, 768]),
                  f"Projection output {tuple(enc_hs.shape)}")

            # UNet (1 timestep, all 6 frames batched)
            latents   = noised.squeeze(0)                        # (6, 4, 32, 32)
            t_test    = torch.tensor([500]*6, device=DEVICE)
            cond_test = enc_hs.repeat_interleave(6, dim=0)      # (6, 77, 768)
            out       = unet(latents, t_test,
                             encoder_hidden_states=cond_test).sample
            check(out.shape == torch.Size([6, 4, 32, 32]),
                  f"UNet output {tuple(out.shape)}")

            # VAE decode
            frames = vae.decode(latents / 0.18215).sample       # (6, 3, H, W)
            check(frames.shape[0] == 6 and frames.shape[1] == 3,
                  f"VAE decode {tuple(frames.shape)}")

        except Exception as e:
            check(False, f"Forward pass error: {e}")
            import traceback; traceback.print_exc()
else:
    print("  Skipping forward pass — model load failed above")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if not errors:
    print(" All checks passed!")
    print(" Pipeline is ready. Run:")
    print("   python inference.py --idx 0 --compare --output test.mp4")
else:
    print(f" {len(errors)} check(s) FAILED:")
    for e in errors:
        print(f"   - {e}")
    print("\n Fix the above before running inference.py")
print("=" * 55)