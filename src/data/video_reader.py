import cv2
import os
import numpy as np

def read_video(video_path):
    """
    Read video file and return list of frames.
    Args:
        video_path (str): Path to the video file.
    Returns:
        list: List of frames (each frame is a numpy array in BGR format).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def frames_to_tensor(frames, height=64, width=128):
    """
    Convert list of frames to a tensor.
    Args:
        frames (list): List of frames (BGR format).
        height (int): Target height for resizing.
        width (int): Target width for resizing.
    Returns:
        np.ndarray: Tensor of shape (T, H, W, C) where T is number of frames.
    """
    resized_frames = []
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, (width, height))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized_frames.append(rgb)

    # Stack frames to create tensor (T, H, W, C)
    tensor = np.stack(resized_frames, axis=0)
    return tensor

if __name__ == "__main__":
    # Test with a dummy video (if available)
    print("Video reader module loaded.")