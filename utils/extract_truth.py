import torch
import cv2
import numpy as np
import os

# 1. Path to the EXACT ground truth tensor
# Since you used --idx 0, we look for sample 0
gt_path = "/home/teaching/TEAM_22_DATASET/processed/eeg_sample_000001.pt"

if not os.path.exists(gt_path):
    print(f"❌ Could not find {gt_path}")
else:
    # 2. Load and process
    video_tensor = torch.load(gt_path) 
    
    # Tensors are usually [Frames, C, H, W] in range [0, 1]
    # Move to [Frames, H, W, C] for OpenCV
    frames = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    frames = np.clip(frames, 0, 1) # Ensure range is 0-1
    frames = (frames * 255).astype(np.uint8)

    # 3. Write to Video
    height, width = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ABSOLUTE_GT_RAW.mp4', fourcc, 3.0, (width, height))

    for i in range(frames.shape[0]):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release()
    print("✅ ABSOLUTE_GT_RAW.mp4 created successfully.")
