from eeg2video_models import generate_conditioning_tensors
import torch

eeg = torch.load("your_preprocessed_eeg.pt")   # (B, 62, 100) float32

generate_conditioning_tensors(
    vit_path      = "real_models/vit_real_data.pth",
    text_mlp_path = "real_models/text_mlp_final.pth",
    dynamics_path = "real_models/dynamics_model.pth",
    eeg_input     = eeg,
    output_dir    = "real_inputs",
)
# Saves:
#   real_inputs/visual_latents.pt   → (B, 6, 4, 32, 32)
#   real_inputs/text_embeddings.pt  → (B, 77, 512)
#   real_inputs/is_fast.pt          → (B,)

from inference import EEG2VideoPipeline

pipeline = EEG2VideoPipeline.from_checkpoints(device="cuda")
video = pipeline.run_from_real_inputs()  # (B, 6, 3, 128, 128) uint8