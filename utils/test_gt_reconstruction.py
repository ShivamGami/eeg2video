# test_gt_reconstruction.py
import torch
import numpy as np
import imageio
import os
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from projection import EEGProjection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "./checkpoints"
VAE_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Ground Truth Paths
REAL_VAE_PATH = "/home/teaching/vishal_workspace/eeg2video-cs671/subteam2_videoldm/real_vae_latents_50k.pt"
TEXT_PATH = "/home/teaching/vishal_workspace/eeg2video-cs671/text_embeddings_FIXED.pt"

@torch.no_grad()
def test_reconstruction(sample_idx=0, noise_level=0.3):
    print("1. Loading Models...")
    unet = UNet2DConditionModel.from_pretrained(f"{CHECKPOINT_DIR}/unet_best", torch_dtype=torch.float32).to(DEVICE)
    proj = EEGProjection(512, 768, 77).to(DEVICE)
    proj.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/projection_best.pth", map_location=DEVICE))
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, subfolder="vae", torch_dtype=torch.float32).to(DEVICE)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
    
    unet.eval(); proj.eval(); vae.eval()

    print(f"\n2. Loading Ground Truth Sample {sample_idx}...")
    gt_latents = torch.load(REAL_VAE_PATH, map_location='cpu', mmap=True)[sample_idx].to(DEVICE) # (6, 4, 32, 32)
    text_embed = torch.load(TEXT_PATH, map_location='cpu')[sample_idx].unsqueeze(0).to(DEVICE)   # (1, 512)

    # --- CRITICAL SCALING FIX ---
    print(f"   Original GT Latent STD: {gt_latents.std():.4f}")
    if gt_latents.std() < 0.5:
        print("   -> Latent is unscaled! Multiplying by (1/0.18215) so UNet sees STD=1.0")
        unet_latents = gt_latents / 0.18215
    else:
        unet_latents = gt_latents
    print(f"   Scaled Latent STD for UNet: {unet_latents.std():.4f}")

    # Project Text
    cond_input = proj(text_embed) # (1, 77, 768)

    # --- PROPER SD NOISE ADDITION ---
    t_start = int(noise_level * 1000)
    print(f"\n3. Adding Noise (Level: {noise_level}, Timestep: {t_start})")
    
    noise = torch.randn_like(unet_latents)
    t_tensor = torch.tensor([t_start] * 6, device=DEVICE).long()
    noisy_latents = scheduler.add_noise(unet_latents, noise, t_tensor)

    # --- DENOISING WITHOUT CFG BUG ---
    scheduler.set_timesteps(50) # Use 50 steps for quality
    timesteps = [t for t in scheduler.timesteps if t.item() <= t_start]
    
    print(f"\n4. Denoising for {len(timesteps)} steps...")
    latents = noisy_latents
    
    for i, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0).expand(6).to(DEVICE)
        cond_tiled = cond_input.repeat_interleave(6, dim=0)
        
        # NO CFG! Just pure prediction
        noise_pred = unet(latents, t_batch, encoder_hidden_states=cond_tiled).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # --- PROPER VAE DECODE SCALING ---
    print("\n5. Decoding to Video...")
    # UNet outputs std=1.0. VAE Decode expects std=5.5.
    # Therefore, divide by 0.18215!
    latents_for_vae = latents / 0.18215
    imgs = vae.decode(latents_for_vae).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    frames = (imgs * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    imageio.mimwrite("GT_reconstructed.mp4", frames, fps=3, quality=9)
    print("✅ SUCCESS! Saved as GT_reconstructed.mp4")

if __name__ == "__main__":
    test_reconstruction(sample_idx=0, noise_level=0.3)