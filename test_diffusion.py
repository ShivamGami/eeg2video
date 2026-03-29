import torch
from diffusers import StableDiffusionPipeline

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -------------------------
# Load model
# -------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

unet = pipe.unet
vae = pipe.vae
scheduler = pipe.scheduler

# -------------------------
# Dummy inputs (IMPORTANT)
# -------------------------
B = 1

latents = torch.randn(B, 4, 64, 64, dtype=torch.float16).to(device)
text_embeddings = torch.randn(B, 77, 768, dtype=torch.float16).to(device)

# -------------------------
# Diffusion setup
# -------------------------
scheduler.set_timesteps(20)

# -------------------------
# Diffusion loop
# -------------------------
with torch.no_grad():
    for t in scheduler.timesteps:
        t = t.to(device)

        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=text_embeddings
        ).sample

        latents = scheduler.step(
            noise_pred,
            t,
            latents
        ).prev_sample

# -------------------------
# Decode image
# -------------------------
image = vae.decode(latents / 0.18215).sample

# -------------------------
# Convert to PIL
# -------------------------
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]

from PIL import Image
image = Image.fromarray((image * 255).astype("uint8"))

image.save("phase1_output.png")

print("Phase 1 output generated ✅")