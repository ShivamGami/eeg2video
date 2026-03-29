import torch
from diffusers import StableDiffusionPipeline

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# dummy embeddings
text_embeddings = torch.randn(1, 77, 768, dtype=torch.float16).to(device)

# reduced latents
latents = torch.randn(1, 4, 32, 32, dtype=torch.float16).to(device)

pipe.scheduler.set_timesteps(10)

for t in pipe.scheduler.timesteps:
    t = t.to(device)

    noise_pred = pipe.unet(
        latents,
        t,
        encoder_hidden_states=text_embeddings
    ).sample

    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

image = pipe.vae.decode(latents / 0.18215).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]

from PIL import Image
image = Image.fromarray((image * 255).astype("uint8"))

image.save("embedding_output.png")

print("Generated successfully ✅")