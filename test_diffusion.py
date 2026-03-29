import torch
from diffusers import StableDiffusionPipeline

# -------------------------------
# Device setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# Load model (optimized for GPU)
# -------------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe = pipe.to(device)

# -------------------------------
# Optional optimizations (safe)
# -------------------------------
pipe.enable_attention_slicing()

# If xformers is installed → faster attention
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    pass

# -------------------------------
# Inference (NO gradients)
# -------------------------------
prompt = "A beautiful girl holding a cat in her hands"

with torch.no_grad():
    image = pipe(
        prompt,
        num_inference_steps=30,   # good quality
        guidance_scale=7.5        # standard CFG
    ).images[0]

# -------------------------------
# Save output
# -------------------------------
image.save("output.png")

print("Image generated successfully ✅")