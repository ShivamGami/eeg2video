import os
import cv2
import numpy as np
import torch

def segment_video_to_clips(video_path, target_fps=3, clip_duration_sec=2):
    """
    Reads a raw video, resizes to 256x256, downsamples it to target_fps, 
    and segments it into clips of exactly (target_fps * clip_duration_sec) frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or np.isnan(original_fps):
        original_fps = 30.0 # Fallback
        
    # Calculate how many frames to skip to achieve 3 FPS
    frame_skip_interval = max(1, int(round(original_fps / target_fps)))
    frames_per_clip = target_fps * clip_duration_sec # 3 * 2 = 6 frames
    
    extracted_frames = []
    current_frame_index = 0

    # Extract and downsample frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only grab the frame if it matches our 3 FPS interval
        if current_frame_index % frame_skip_interval == 0:
            # CRITICAL ADDITION: Resize real videos to 256x256 for the VAE
            frame_resized = cv2.resize(frame, (256, 256))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
            extracted_frames.append(frame_transposed)
            
        current_frame_index += 1

    cap.release()

    # Segment into 6-frame (2-second) clips
    valid_clip_count = len(extracted_frames) // frames_per_clip
    all_clips_tensors = []
    
    for i in range(valid_clip_count):
        start_idx = i * frames_per_clip
        end_idx = start_idx + frames_per_clip
        clip_frames = extracted_frames[start_idx:end_idx]
        clip_tensor = torch.tensor(np.array(clip_frames))
        all_clips_tensors.append(clip_tensor)

    if all_clips_tensors:
        return torch.stack(all_clips_tensors)
    else:
        return None

def main():
    # 1. Point to the REAL dataset and output folder
    DATASET_DIR = "/home/teaching/TEAM_22_DATASET/SEED-DV/SEED-DV/Video"
    OUTPUT_DIR = "/home/teaching/vishal_workspace/eeg2video-cs671/processed_raw_clips"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Find all real .mp4 files
    video_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} real videos to process in {DATASET_DIR}\n")
    
    # 3. Process each video and save it
    for video_file in video_files:
        video_path = os.path.join(DATASET_DIR, video_file)
        base_name = video_file.replace(".mp4", "")
        
        print(f"Processing: {video_file}...")
        
        # Run the updated segmentation function
        clips_tensor = segment_video_to_clips(video_path, target_fps=3, clip_duration_sec=2)
        
        if clips_tensor is not None:
            output_file = os.path.join(OUTPUT_DIR, f"{base_name}_segmented.pt")
            torch.save(clips_tensor, output_file)
            print(f"-> SUCCESS! Saved {base_name}_segmented.pt | Final Shape: {clips_tensor.shape}\n")
        else:
            print(f"-> Warning: Could not extract clips from {video_file}\n")

if __name__ == "__main__":
    main()