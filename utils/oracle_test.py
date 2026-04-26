"""
inference.py — Sub-team 5 Master Pipeline
==========================================
Takes a sample index and runs the FULL pipeline end-to-end:

  Step 1: Load pre-computed tensors (Sub-teams 3 & 4)
          visual_latent  (6, 4, 32, 32)  — from Sub-team 3 ViT
          text_embed     (512,)           — from Sub-team 4 Text MLP
          is_fast        (1,)             — from Sub-team 4 Dynamics

  Step 2: DANA Module
          Use is_fast to select beta (0.85 fast / 0.35 slow)
          Add dynamics-aware noise to visual_latent
          → noised_latent (6, 4, 32, 32)

  Step 3: Sub-team 2 VideoLDM
          Project text_embed (512) → (77, 768)
          Denoise noised_latent conditioned on text_embed
          → final_latent (6, 4, 32, 32)

  Step 4: VAE Decode
          Decode final_latent → RGB frames (6, H, W, 3)
          Save as MP4

Usage:
    # Generate single video
    python inference.py --idx 0 --output video_0.mp4

    # Generate with comparison (generated vs ground truth decoded)
    python inference.py --idx 0 --compare --output video_0.mp4

    # Generate 5 random test samples
    python inference.py --multi 5 --output_dir ./pipeline_outputs/manan_correction

    # Fast test (fewer denoising steps)
    python inference.py --idx 42 --steps 10 --output quick_test.mp4
"""
import cv2
import argparse
import os
import random
import torch
import numpy as np
import imageio
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from diffusers.utils import logging as diffusers_logging
import torch.nn.functional as F
# Sub-team 2 & 5 modules
# Copy projection.py and dana.py into the same folder as this script
from projection import EEGProjection
from dana import DANAModule

diffusers_logging.set_verbosity_error()

# ── Paths — update these if files move ───────────────────────────────────────
VISUAL_PATH    = "/home/teaching/manan_workspace/eeg2video-cs671/subteam3_vit/visual_latents.pt"
TEXT_PATH      = "/home/teaching/vishal_workspace/eeg2video-cs671/text_embeddings.pt"
FAST_PATH      = "/home/teaching/vishal_workspace/eeg2video-cs671/is_fast.pt"
CHECKPOINT_DIR = "/home/teaching/checkpoints"
VAE_MODEL_ID   = "runwayml/stable-diffusion-v1-5"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 6
FPS        = 3

# ── Denoising config ──────────────────────────────────────────────────────────
# DANA adds partial noise (not full), so we use fewer denoising steps.
# Full noise → 50 steps. DANA partial noise (beta=0.35-0.85) → 15-25 steps.
DEFAULT_STEPS    = 20
GUIDANCE_SCALE   = 7.5


# ── Model loading ─────────────────────────────────────────────────────────────

def load_all_models():
    """Load all Sub-team 2 + 5 models using absolute paths."""

    # Explicit paths as per your environment
    unet_path = "/home/teaching/checkpoints/unet_best"
    # Note: If you want to use your 100k step model, 
    # make sure you have copied/renamed it to this path!
    proj_path = "/home/teaching/checkpoints/projection_best.pth"

    if not os.path.exists(unet_path):
        raise FileNotFoundError(
            f"\n[ERROR] UNet folder not found at {unet_path}"
        )
    
    if not os.path.exists(proj_path):
        raise FileNotFoundError(
            f"\n[ERROR] Projection weights not found at {proj_path}\n"
            "Did you rename projection_step_100000.pth to projection_best.pth?"
        )

    print(f"[1/4] Loading Sub-team 2 UNet from {unet_path}...")
    unet = UNet2DConditionModel.from_pretrained(
        unet_path, torch_dtype=torch.float32
    ).to(DEVICE).eval()

    print(f"[2/4] Loading EEGProjection from {proj_path}...")
    projection = EEGProjection(input_dim=512, output_dim=768, seq_len=77).to(DEVICE)
    projection.load_state_dict(torch.load(proj_path, map_location=DEVICE))
    projection.eval()

    print("[3/4] Loading VAE decoder...")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    ).to(DEVICE).eval()

    print("[4/4] Initialising DANA module + DDIM scheduler...")
    dana = DANAModule().to(DEVICE)
    dana.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )

    print("All models loaded successfully.\n")
    return unet, projection, vae, dana, scheduler

# ── Latent → RGB frames ───────────────────────────────────────────────────────

@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> np.ndarray:
    """
    (F, 4, 32, 32) → (F, H, W, 3) uint8 RGB
    """
    scaled = latents 
    imgs   = vae.decode(scaled).sample          # (F, 3, H, W) in [-1, 1]
    imgs   = (imgs.clamp(-1, 1) + 1) / 2       # [0, 1]
    imgs   = (imgs * 255).byte()                # uint8
    return imgs.permute(0, 2, 3, 1).cpu().numpy()  # (F, H, W, 3)


# ── Core pipeline for one sample ──────────────────────────────────────────────

@torch.no_grad()
def run_pipeline(
    sample_idx:   int,
    unet:         UNet2DConditionModel,
    projection:   EEGProjection,
    vae:          AutoencoderKL,
    dana:         DANAModule,
    scheduler:    DDIMScheduler,
    num_steps:    int   = DEFAULT_STEPS,
    guidance:     float = GUIDANCE_SCALE,
) -> tuple:
    # ── Step 1: Load Ground Truth (Oracle Test) ───────────────────────────
    # We skip the 28GB file and load the individual sample directly
    GT_PATH = f"/home/teaching/TEAM_22_DATASET/processed/processed/video_sample_{sample_idx:06d}.pt"
    
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"Ground Truth sample not found at: {GT_PATH}")
    
    visual_latent = torch.load(GT_PATH, map_location=DEVICE).unsqueeze(0).float()
    
    # Load supporting data
    text_embeds = torch.load(TEXT_PATH,   map_location="cpu")
    is_fast_all = torch.load(FAST_PATH,   map_location="cpu")

    text_embed    = text_embeds[sample_idx].unsqueeze(0).to(DEVICE)
    is_fast       = is_fast_all[sample_idx].unsqueeze(0).to(DEVICE)

    is_fast_val = bool(is_fast.item())
    print(f"Sample {sample_idx} | Fast motion: {is_fast_val}")

    # ── Step 2: DANA (Adds noise to our Ground Truth) ─────────────────────
    # noised_latent, beta_tensor = dana(visual_latent, is_fast)
    # beta_used = beta_tensor[0].item()
    beta_used = 0.35
    t_start_val = int(beta_used * 1000)
    t_start_tensor = torch.tensor([t_start_val]).to(DEVICE)
    
    # Manually add noise to define the variable
    noise = torch.randn_like(visual_latent)
    noised_latent = scheduler.add_noise(visual_latent, noise, t_start_tensor)
    print(f"DANA beta: {beta_used:.2f} ({'fast' if is_fast_val else 'slow'} motion)")
    # noised_latent shape: (1, 6, 4, 32, 32)

    # --- Step 3: VideoLDM denoising ---
    # Project EEG text embedding to SD cross-attention format
    # (1, 512) -> (1, 77, 768)
    encoder_hs = projection(text_embed)
    encoder_hs = torch.clamp(encoder_hs, min=-2.0, max=2.0)
    print(f"Safety Check - Signal Max: {encoder_hs.max().item():.2f}")
    uncond_hs = torch.zeros_like(encoder_hs)

    # Stack [uncond, cond] for classifier-free guidance in one forward pass
    cond_input = torch.cat([uncond_hs, encoder_hs]) # Shape MUST be (2, 77, 768)

    # Calculate starting timestep from beta (DANA partial noise)
    # DANA adds beta fraction of noise, so we start denoising from
    # the corresponding timestep rather than T=1000
    t_start = int(beta_used * 1000)
    scheduler.set_timesteps(num_steps)

    # Filter timesteps to only those ≤ t_start (we start mid-schedule)
    timesteps = [t for t in scheduler.timesteps if t.item() <= t_start]
    if not timesteps:
        timesteps = scheduler.timesteps  # fallback: use all steps

    print(f"Denoising {NUM_FRAMES} frames × {len(timesteps)} steps "
          f"(starting from t={t_start})...")

    latents=noised_latent 
    for i, t in enumerate(timesteps):
        lat_input  = torch.cat([latents] * 2)
        lat_input_flat = lat_input.view(-1, *lat_input.shape[2:])
        t_batch = t.unsqueeze(0).expand(lat_input_flat.shape[0]).to(DEVICE)
        cond_tiled = cond_input.repeat_interleave(NUM_FRAMES, dim=0)


        noise_pred = unet(
        lat_input_flat, t_batch, 
        encoder_hidden_states=cond_tiled,
        ).sample
        noise_u  = noise_pred[:NUM_FRAMES]
        noise_c  = noise_pred[NUM_FRAMES:]
        noise_g  = noise_u + guidance * (noise_c - noise_u)

        latents = scheduler.step(noise_g, t, latents).prev_sample
        
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{len(timesteps)}")
    # ── Step 4: VAE decode ────────────────────────────────────────────────
    print("Decoding latents to RGB frames...")

    latents_flat = latents.view(-1, *latents.shape[2:])
    gen_frames = decode_latents(vae, latents_flat)
    return gen_frames, is_fast_val, beta_used


# ── Save video ────────────────────────────────────────────────────────────────

def save_video(frames: np.ndarray, path: str, fps: int = FPS):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8)
    print(f"Saved: {path}  ({len(frames)} frames @ {fps}fps)")

# --- Add these constants and function ---
VIDEO_DIR = "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/video"
VIDEO_FILES = ["1st_10min.mp4", "2nd_10min.mp4", "3rd_10min.mp4", 
               "4th_10min.mp4", "5th_10min.mp4", "6th_10min.mp4", "7th_10min.mp4"]

SAMPLES_PER_VIDEO_BLOCK = 41580  # Total samples (all subjects) per video file
SAMPLES_PER_SUB_VIDEO   = 2080   # Approx samples for one 8:40 video session
STEP_IN_FRAMES          = 6      # 24fps * 0.25s step

def extract_raw_frames(sample_idx, num_frames=6, target_size=(128, 128)):
    # 1. Map global sample index to the correct video file
    video_num = min(sample_idx // SAMPLES_PER_VIDEO_BLOCK, 6)
    
    # 2. Map to the frame within that video (handling the subject wrap-around)
    local_idx = sample_idx % SAMPLES_PER_VIDEO_BLOCK
    subject_relative_idx = local_idx % SAMPLES_PER_SUB_VIDEO
    start_frame = subject_relative_idx * STEP_IN_FRAMES
    
    video_path = os.path.join(VIDEO_DIR, VIDEO_FILES[video_num])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        # Resize to target_size (W, H) to avoid concatenation errors
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame_resized)
    
    cap.release()
    return np.array(frames)


def save_comparison(gen_frames: np.ndarray, sample_idx: int, path: str):
    """Side-by-side: generated (left) vs RAW source video (right)."""
    try:
        # Get dimensions from the generated frames (e.g., 128x128)
        H, W = gen_frames.shape[1], gen_frames.shape[2]
        
        # Pull the REAL ground truth from the MP4 source
        gt_frames = extract_raw_frames(sample_idx, num_frames=6, target_size=(W, H))
        
        if gt_frames is None or len(gt_frames) < 6:
            raise ValueError("Could not extract raw frames from video source.")

        # Create a white divider
        divider = np.ones((gen_frames.shape[0], H, 4, 3), dtype=np.uint8) * 255
        
        # Concatenate: [Gen (Left) | White Bar | Raw GT (Right)]
        comparison = np.concatenate([gen_frames, divider, gt_frames], axis=2)
        save_video(comparison, path)
        print(f"  Saved comparison: {path} (Right side is 1080p Raw Source)")
        
    except Exception as e:
        print(f"  Comparison failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EEG2Video Full Pipeline — Sub-team 5 inference"
    )
    parser.add_argument("--idx",        type=int,   default=0,
                        help="Sample index into pre-computed tensors")
    parser.add_argument("--steps",      type=int,   default=DEFAULT_STEPS,
                        help="DDIM denoising steps")
    parser.add_argument("--guidance",   type=float, default=GUIDANCE_SCALE,
                        help="Classifier-free guidance scale")
    parser.add_argument("--output",     type=str,   default="pipeline_output.mp4",
                        help="Output MP4 path")
    parser.add_argument("--compare",    action="store_true",
                        help="Also save side-by-side with ground truth")
    parser.add_argument("--multi",      type=int,   default=0,
                        help="Generate N random test samples")
    parser.add_argument("--output_dir", type=str,   default="./pipeline_outputs",
                        help="Output dir when using --multi")
    args = parser.parse_args()

    # Load all models once
    unet, projection, vae, dana, scheduler = load_all_models()

    if args.multi > 0:
        # Pick from test split (last 15% = indices 247,401 to 291,059)
        test_start = int(291060 * 0.85)
        indices    = random.sample(range(test_start, 291060), args.multi)
        print(f"Generating {args.multi} test samples: {indices}\n")

        os.makedirs(args.output_dir, exist_ok=True)
        for idx in indices:
            gen_frames, is_fast_val, beta = run_pipeline(
                idx, unet, projection, vae, dana, scheduler,
                args.steps, args.guidance
            )
            out = os.path.join(args.output_dir, f"sample_{idx:06d}.mp4")
            save_video(gen_frames, out)

            if args.compare:
                cmp = os.path.join(args.output_dir, f"sample_{idx:06d}_compare.mp4")
                save_comparison(gen_frames, idx, cmp)

            print(f"  beta={beta:.2f} | fast={is_fast_val}\n")

        print(f"\nAll {args.multi} videos saved to: {args.output_dir}/")

    else:
        gen_frames, is_fast_val, beta = run_pipeline(
            args.idx, unet, projection, vae, dana, scheduler,
            args.steps, args.guidance
        )
        save_video(gen_frames, args.output)

        if args.compare:
            cmp = args.output.replace(".mp4", "_compare.mp4")
            save_comparison(gen_frames, args.idx, cmp)

        print(f"\nDone. beta={beta:.2f} | fast={is_fast_val}")


if __name__ == "__main__":
    main()
