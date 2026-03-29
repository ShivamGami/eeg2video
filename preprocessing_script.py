import cv2
import numpy as np
import torch

def segment_video_to_clips(video_path, target_fps=3, clip_duration_sec=2):
    """
    Reads a raw video, downsamples it to target_fps, and segments it into 
    clips of exactly (target_fps * clip_duration_sec) frames[cite: 9].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:  # Fallback just in case OpenCV can't read the dummy FPS
        original_fps = 30.0
        
    # Calculate how many frames to skip to achieve 3 FPS [cite: 9]
    frame_skip_interval = max(1, int(round(original_fps / target_fps)))
    frames_per_clip = target_fps * clip_duration_sec # 3 * 2 = 6 frames [cite: 9]
    
    extracted_frames = []
    current_frame_index = 0

    # Extract downsampled frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only grab the frame if it matches our 3 FPS interval [cite: 9]
        if current_frame_index % frame_skip_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
            extracted_frames.append(frame_transposed)
            
        current_frame_index += 1

    cap.release()

    # Segment into 6-frame (2-second) clips [cite: 9]
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
        return torch.empty(0)


def run_local_test():
    print("1. Generating a dummy 4-second video at 30 FPS...")
    dummy_filename = "test_dummy_video.mp4"
    fps = 30
    duration = 4 # 4 seconds total
    total_frames = fps * duration
    height, width = 256, 256

    # Setup OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dummy_filename, fourcc, fps, (width, height))

    # Create random noise frames and save them to the video
    for _ in range(total_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    print(f"-> Successfully created '{dummy_filename}'\n")

    print("2. Running the preprocessing script...")
    try:
        # Segment the video into 2-sec clips at 3 FPS [cite: 9]
        clips_tensor = segment_video_to_clips(dummy_filename, target_fps=3, clip_duration_sec=2)
        
        print("3. Verifying the results...")
        print(f"-> Output Tensor Shape: {clips_tensor.shape}")
        
        expected_shape = (2, 6, 3, height, width)
        
        if tuple(clips_tensor.shape) == expected_shape:
            print("\nSUCCESS! The script correctly downsampled and extracted the frames.")
            print(f"Batch Size (Num Clips): {clips_tensor.shape[0]}")
            print(f"Frames per Clip: {clips_tensor.shape[1]}")
            print(f"Channels: {clips_tensor.shape[2]}")
        else:
            print(f"\nFAIL: Expected shape {expected_shape}, but got {tuple(clips_tensor.shape)}")
            
    except Exception as e:
        print(f"\nError running script: {e}")

if __name__ == "__main__":
    run_local_test()
    