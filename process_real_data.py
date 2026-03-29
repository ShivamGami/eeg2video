import os
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# --- CONFIGURATION ---
SEGMENTED_DIR = "/home/teaching/vishal_workspace/eeg2video-cs671/processed_raw_clips"
TEXT_DIR = "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/Video/BLIP-caption"
OUTPUT_DIR = "/home/teaching/vishal_workspace/eeg2video-cs671/processed_features"

def encode_video_tensor(tensor_path, vae, device):
    print(f"  -> Loading {os.path.basename(tensor_path)}...")
    # Load the segmented raw video tensor: [Num_Clips, 6, 3, 256, 256]
    video_tensor = torch.load(tensor_path)
    
    # In preprocessing, values were saved as [0, 1]. The VAE needs [-1, 1].
    video_tensor = (video_tensor * 2.0) - 1.0 
    
    all_latents = []
    
    # Process one clip (6 frames) at a time to prevent GPU Memory crashes
    with torch.no_grad():
        for i in range(video_tensor.size(0)):
            clip = video_tensor[i].to(device) # [6, 3, 256, 256]
            # Compress through VAE
            latents = vae.encode(clip).latent_dist.sample() * vae.config.scaling_factor
            all_latents.append(latents.cpu())
            
    # Stack back together into [Num_Clips, 6, 4, 32, 32]
    return torch.stack(all_latents)

def process_text(text_path, tokenizer, text_encoder, device):
    print(f"  -> Processing Text: {os.path.basename(text_path)}")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().strip() 
        
    text_inputs = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = text_encoder(text_inputs.input_ids)[0]
    return embeddings.cpu()

def main():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.2, device=0)
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Models...")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
    
    # Find all preprocessed tensor files
    segmented_files = [f for f in os.listdir(SEGMENTED_DIR) if f.endswith("_segmented.pt")]
    
    for seg_file in segmented_files:
        base_name = seg_file.replace("_segmented.pt", "") # e.g., "4th_10min"
        
        tensor_path = os.path.join(SEGMENTED_DIR, seg_file)
        text_path = os.path.join(TEXT_DIR, f"{base_name}.txt")
        
        print(f"\n--- Starting {base_name} ---")
        
        # 1. Generate Video Latents
        visual_latents = encode_video_tensor(tensor_path, vae, device)
        torch.save(visual_latents, os.path.join(OUTPUT_DIR, f"{base_name}_latents.pt"))
        print(f"Saved latents: {visual_latents.shape}") 
            
        # 2. Generate Text Embeddings
        if os.path.exists(text_path):
            text_embeddings = process_text(text_path, tokenizer, text_encoder, device)
            torch.save(text_embeddings, os.path.join(OUTPUT_DIR, f"{base_name}_text_emb.pt"))
            print(f"Saved text embeddings: {text_embeddings.shape}")
        else:
            print(f"WARNING: Text file not found at {text_path}")

if __name__ == "__main__":
    main()