import cv2
import sys
import os

def open_video(video_path):
    """Open and play an MP4 video file using OpenCV with looping support."""

    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        sys.exit(1)

    # Check if it's an MP4 file
    if not video_path.lower().endswith('.mp4'):
        print("Warning: File may not be an MP4 video.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")

    print("\nControls:")
    print("  SPACE - Pause / Resume")
    print("  Q or ESC - Quit")
    print("  LEFT arrow  - Rewind 5 seconds")
    print("  RIGHT arrow - Forward 5 seconds")
    print("  L - Toggle Loop ON/OFF")

    delay = int(1000 / fps) if fps > 0 else 30
    paused = False
    loop = True  # Loop enabled by default

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                if loop:
                    print("Looping video...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("End of video. Paused.")
                    paused = True
                    continue

            cv2.imshow("Video Player", frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            print("Playback stopped by user.")
            break

        elif key == ord(' '):  # SPACE
            paused = not paused
            print("Paused." if paused else "Resumed.")

        elif key == ord('l'):  # Toggle loop
            loop = not loop
            print("Loop ON" if loop else "Loop OFF")

        # LEFT arrow (may vary by system)
        elif key in [81, 2, ord('a')]:
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - fps * 5))

        # RIGHT arrow (may vary by system)
        elif key in [83, 3, ord('d')]:
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count, current + fps * 5))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python open_video.py <path_to_video.mp4>")
        print("Example: python open_video.py myvideo.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    open_video(video_path)
