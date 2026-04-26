"""
evaluate.py — Sub-team 5 Quantitative Evaluation (Final)
========================================================
Synchronized with SEED-DV 50% Overlap Logic:
- Step Size: 250ms (0.25s)
- Video FPS: 24
- Frame Jump per Sample: 6 frames
"""

import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from diffusers.utils import logging as diffusers_logging
from projection import EEGProjection
diffusers_logging.set_verbosity_error()

# ── Updated Paths ──────────────────────────────────────────────────────────────
TEXT_PATH      = "/home/teaching/vishal_workspace/eeg2video-cs671/text_embeddings.pt"
# FOLDER containing your individual .pt files
LATENT_DIR     = "/home/teaching/TEAM_22_DATASET/processed/processed" 
CHECKPOINT_DIR = "/home/teaching/checkpoints"
VAE_MODEL_ID   = "runwayml/stable-diffusion-v1-5"
REAL_VIDEO_DIR = "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/video"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 6

# ── Mapping Constants ──────────────────────────────────────────────────────────
SAMPLES_PER_VIDEO_BLOCK = 41580 
SAMPLES_PER_SUB_VIDEO   = 2080  
STEP_IN_FRAMES          = 6

def check_packages(mode):
    if mode == "latent": return
    missing = []
    try: import cv2
    except ImportError: missing.append("opencv-python-headless")
    try: import skimage
    except ImportError: missing.append("scikit-image")
    try: import open_clip
    except ImportError: missing.append("open-clip-torch")
    if missing:
        print(f"Missing: {missing}. Install with: pip install opencv-python-headless scikit-image open-clip-torch")
        exit(1)

def load_models():
    print("Loading UNet and Projection Layer...")
    unet = UNet2DConditionModel.from_pretrained(f"{CHECKPOINT_DIR}/unet_best", torch_dtype=torch.float32).to(DEVICE).eval()
    proj = EEGProjection(512, 768, 77).to(DEVICE)
    proj.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/projection_best.pth", map_location=DEVICE))
    proj.eval()
    return unet, proj

@torch.no_grad()
def generate_latents(idx, unet, proj, scheduler, num_steps=20, guidance_scale=7.5):
    text_embeds = torch.load(TEXT_PATH, map_location="cpu")
    text_embed  = text_embeds[idx].unsqueeze(0).to(DEVICE)
    enc_hs  = proj(text_embed)
    uncond  = torch.zeros_like(enc_hs)
    cond_in = torch.cat([uncond, enc_hs])
    scheduler.set_timesteps(num_steps)
    
    # Start from random noise (Pure Generation)
    latents = torch.randn(1, NUM_FRAMES, 4, 32, 32, device=DEVICE)
    for t in scheduler.timesteps:
        lat_in     = torch.cat([latents] * 2)
        lat_in_flat = lat_in.view(-1, *lat_in.shape[2:])
        t_batch    = t.unsqueeze(0).expand(lat_in_flat.shape[0]).to(DEVICE)
        cond_tiled = cond_in.repeat_interleave(NUM_FRAMES, dim=0)
        noise_pred = unet(lat_in_flat, t_batch, encoder_hidden_states=cond_tiled).sample
        
        noise_u = noise_pred[:NUM_FRAMES]
        noise_c = noise_pred[NUM_FRAMES:]
        noise_g = noise_u + guidance_scale * (noise_c - noise_u)
        
        # Reshape noise_g to (1, 6, 4, 32, 32)
        noise_g = noise_g.unsqueeze(0)
        latents = scheduler.step(noise_g, t, latents).prev_sample
    return latents.squeeze(0) # (F, 4, 32, 32)

def get_real_video_frames(sample_idx, target_size=(128, 128)):
    import cv2
    video_files = ["1st_10min.mp4", "2nd_10min.mp4", "3rd_10min.mp4", "4th_10min.mp4", 
                   "5th_10min.mp4", "6th_10min.mp4", "7th_10min.mp4"]
    
    # Corrected Temporal Mapping
    video_num = min(sample_idx // SAMPLES_PER_VIDEO_BLOCK, 6)
    local_idx = sample_idx % SAMPLES_PER_VIDEO_BLOCK
    subject_relative_idx = local_idx % SAMPLES_PER_SUB_VIDEO
    start_frame = subject_relative_idx * STEP_IN_FRAMES
    
    video_path = os.path.join(REAL_VIDEO_DIR, video_files[video_num])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(NUM_FRAMES):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames) if len(frames) == NUM_FRAMES else None

# ── Metric Calculators ────────────────────────────────────────────────────────

def compute_ssim_psnr(gen, gt):
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    s, p = [], []
    for g, r in zip(gen, gt):
        s.append(ssim(g, r, channel_axis=2, data_range=1.0))
        p.append(psnr(r, g, data_range=1.0))
    return np.mean(s), np.mean(p)

def compute_clip_score(gen, gt, model, preprocess):
    from PIL import Image
    def feats(frames):
        imgs  = [preprocess(Image.fromarray((f*255).astype(np.uint8))) for f in frames]
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            f = model.encode_image(batch)
        return f / f.norm(dim=-1, keepdim=True)
    gf = feats(gen); gtf = feats(gt)
    return float((gf * gtf).sum(dim=-1).mean().item())

# ── Main Loop ─────────────────────────────────────────────────────────────────

def evaluate(indices, mode, output_dir, steps, guidance):
    check_packages(mode)
    os.makedirs(output_dir, exist_ok=True)
    unet, proj = load_models()
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    vae, clip_model, clip_preprocess = None, None, None
    if mode == "video":
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, subfolder="vae").to(DEVICE).eval()
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=DEVICE)

    results = []
    print(f"\nStarting Evaluation in {mode} mode...")

    for idx in tqdm(indices, desc="Evaluating"):
        try:
            # 1. Generate Latents
            gen_latents = generate_latents(idx, unet, proj, scheduler, steps, guidance)
            
            with torch.no_grad():
                if mode == "latent":
                    # Load individual GT
                    gt_path = os.path.join(LATENT_DIR, f"video_sample_{idx:06d}.pt")
                    if not os.path.exists(gt_path):
                        gt_path = os.path.join(LATENT_DIR, f"visual_sample_{idx:06d}.pt")
                    
                    gt = torch.load(gt_path, map_location=DEVICE).float()

                    # --- DYNAMIC SHAPE CORRECTION ---
                    if gt.ndim == 3:  # Shape is (6, 32, 32)
                        # Unsqueeze to (6, 1, 32, 32) and repeat channels to match 4
                        gt = gt.unsqueeze(1).repeat(1, 4, 1, 1)
                    elif gt.ndim == 5: # Shape is (1, 6, 4, 32, 32)
                        gt = gt.squeeze(0)
                    # --------------------------------

                    # Now flattened sizes will both be 4096
                    gen_f = gen_latents.view(NUM_FRAMES, -1)
                    gt_f  = gt.view(NUM_FRAMES, -1)

                    # Cosine Similarity Calculation
                    cos = float((F.normalize(gen_f, dim=1) * F.normalize(gt_f, dim=1)).sum(dim=1).mean())
                    mse = float(F.mse_loss(gen_f, gt_f))
                    
                    results.append({"idx": idx, "cosine_sim": cos, "mse": mse})
                    tqdm.write(f"  [{idx}] Cosine: {cos:.4f} | MSE: {mse:.4f}")
                
                else:
                    # Video evaluation remains the same as previous fix
                    scaled = gen_latents
                    gen_frames_tensor = (vae.decode(scaled).sample.clamp(-1,1) + 1)/2
                    gen_frames = gen_frames_tensor.permute(0,2,3,1).detach().cpu().numpy()
                    
                    gt_frames = get_real_video_frames(idx, target_size=(gen_frames.shape[2], gen_frames.shape[1]))
                    if gt_frames is None: continue
                    
                    ssim_v, psnr_v = compute_ssim_psnr(gen_frames, gt_frames)
                    clip_v = compute_clip_score(gen_frames, gt_frames, clip_model, clip_preprocess)
                    results.append({"idx": idx, "ssim": ssim_v, "psnr": psnr_v, "clip": clip_v})

            # Explicitly clean up memory
            del gen_latents
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error at {idx}: {e}")
            torch.cuda.empty_cache()

    # Save metrics
    out_file = os.path.join(output_dir, f"metrics_{mode}_individual.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation Finished. Saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="video", choices=["latent", "video"])
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    test_start = int(291060 * 0.85)
    indices = random.sample(range(test_start, 291060), args.n_samples)
    evaluate(indices, args.mode, "./eval_results", 50, 5.0)