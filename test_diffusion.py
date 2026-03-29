import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  # 🔥 important for GPU
)

pipe = pipe.to(device)

# enable memory optimization (VERY important)
pipe.enable_attention_slicing()

image = pipe("a dog running in a park").images[0]

image.save("output.png")

print("Image generated successfully ✅")