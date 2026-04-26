"""
generate_video.py — Sub-team 2 Video Generation (FINAL v5)
===========================================================
Updated for correct video latent shape (6, 4, 16, 16) — real VAE outputs.
Generates 128x128 pixel videos (16*8 = 128, VAE upscales 8x).
"""

import argparse
import os
import torch
import numpy as np
import imageio
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from diffusers.utils import logging as diffusers_logging
from projection import EEGProjection

diffusers_logging.set_verbosity_error()

TEXT_PATH      = "/home/teaching/TEAM_22_DATASET/processed/processed/"  # load by sample ID
CHECKPOINT_DIR = "./checkpoints"
VAE_MODEL_ID   = "runwayml/stable-diffusion-v1-5"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 6
FPS        = 3

# Video latent spatial size — MUST match what was used in training
LATENT_H = 16
LATENT_W = 16


def load_models(checkpoint_dir):
    print("Loading UNet (fp32)...")
    unet = UNet2DConditionModel.from_pretrained(
        f"{checkpoint_dir}/unet_best", torch_dtype=torch.float32
    ).to(DEVICE).eval()

    print("Loading EEGProjection...")
    proj = EEGProjection(512, 768, 77).to(DEVICE)
    proj.load_state_dict(torch.load(f"{checkpoint_dir}/projection_best.pth",
                                    map_location=DEVICE))
    proj.eval()

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    ).to(DEVICE).eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
    )
    return unet, proj, vae, scheduler


@torch.no_grad()
def decode_latents(vae, latents):
    """(F, 4, H, W) → (F, H*8, W*8, 3) uint8"""
    scaled = latents / 0.18215
    imgs   = vae.decode(scaled).sample          # (F, 3, H*8, W*8) in [-1,1]
    imgs   = (imgs.clamp(-1, 1) + 1) / 2
    imgs   = (imgs * 255).byte()
    return imgs.permute(0, 2, 3, 1).cpu().numpy()  # (F, H*8, W*8, 3)


@torch.no_grad()
def generate_video(sample_idx, unet, proj, vae, scheduler,
                   num_steps=20, guidance=7.5,
                   output_path="output.mp4", compare=False):
    # Load text embed for this sample
    sid  = f"{sample_idx:06d}"
    txt  = torch.load(f"{TEXT_PATH}text_sample_{sid}.pt",
                      map_location="cpu").flatten().unsqueeze(0).to(DEVICE)

    enc_hs   = proj(txt)                         # (1, 77, 768)
    uncond   = torch.zeros_like(enc_hs)
    cond_in  = torch.cat([uncond, enc_hs])       # (2, 77, 768)

    scheduler.set_timesteps(num_steps)

    # Start from Gaussian noise at the correct latent size (16x16)
    latents = torch.randn(NUM_FRAMES, 4, LATENT_H, LATENT_W, device=DEVICE)

    print(f"Denoising {NUM_FRAMES} frames × {num_steps} steps (latent {LATENT_H}×{LATENT_W})...")
    for i, t in enumerate(scheduler.timesteps):
        lat_in     = torch.cat([latents] * 2)
        t_batch    = t.unsqueeze(0).expand(NUM_FRAMES * 2).to(DEVICE)
        cond_tiled = cond_in.repeat_interleave(NUM_FRAMES, dim=0)

        noise_pred = unet(lat_in, t_batch,
                          encoder_hidden_states=cond_tiled).sample

        noise_u  = noise_pred[:NUM_FRAMES]
        noise_c  = noise_pred[NUM_FRAMES:]
        noise_g  = noise_u + guidance * (noise_c - noise_u)
        latents  = scheduler.step(noise_g, t, latents).prev_sample

        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{num_steps}")

    gen_frames = decode_latents(vae, latents)   # (F, 128, 128, 3)

    if compare:
        gt_video = torch.load(f"{TEXT_PATH}video_sample_{sid}.pt",
                               map_location="cpu")
        gt_frames = decode_latents(vae, gt_video.to(DEVICE))
        divider   = np.ones((NUM_FRAMES, gen_frames.shape[1], 4, 3), dtype=np.uint8) * 255
        comparison = np.concatenate([gen_frames, divider, gt_frames], axis=2)
        cmp_path  = output_path.replace(".mp4", "_compare.mp4")
        imageio.mimwrite(cmp_path, comparison, fps=FPS, quality=8)
        print(f"Comparison saved: {cmp_path}  (Left=Generated | Right=Ground Truth)")

    imageio.mimwrite(output_path, gen_frames, fps=FPS, quality=8)
    print(f"Generated: {output_path}  ({NUM_FRAMES} frames @ {FPS}fps, 128×128px)")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx",        type=int,   default=1)
    parser.add_argument("--steps",      type=int,   default=20)
    parser.add_argument("--guidance",   type=float, default=7.5)
    parser.add_argument("--output",     type=str,   default="output.mp4")
    parser.add_argument("--compare",    action="store_true")
    parser.add_argument("--multi",      type=int,   default=0)
    parser.add_argument("--output_dir", type=str,   default="./samples")
    args = parser.parse_args()

    unet, proj, vae, scheduler = load_models(CHECKPOINT_DIR)

    if args.multi > 0:
        import random, glob
        all_files = sorted(glob.glob(f"{TEXT_PATH}video_sample_*.pt"))
        total     = len(all_files)
        # Sample from test split (last 15%)
        test_start = int(total * 0.85)
        indices    = random.sample(range(test_start, total), min(args.multi, total - test_start))
        os.makedirs(args.output_dir, exist_ok=True)
        for idx in indices:
            out = os.path.join(args.output_dir, f"sample_{idx:06d}.mp4")
            generate_video(idx, unet, proj, vae, scheduler,
                           args.steps, args.guidance, out, args.compare)
    else:
        generate_video(args.idx, unet, proj, vae, scheduler,
                       args.steps, args.guidance, args.output, args.compare)


if __name__ == "__main__":
    main()