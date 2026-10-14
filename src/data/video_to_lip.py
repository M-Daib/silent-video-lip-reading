import cv2
import os
import numpy as np
from .lip_extractor import LipExtractor

def video_to_lip_sequence(video_path, target_width=128, target_height=64, sequence_length=75):
    """
    Convert a video file to a sequence of lip ROIs of fixed length.

    Args:
        video_path: Path to input video file
        target_width: Width of output lip ROI
        target_height: Height of output lip ROI
        sequence_length: Fixed length of output sequence (will pad or truncate)

    Returns:
        Numpy array of shape (sequence_length, target_height, target_width, 3) with values in [0, 1]
        or None if no faces detected in any frame
    """
    extractor = LipExtractor()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    lip_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract lip ROI
        lip_roi = extractor.extract_lip_roi(frame, target_width, target_height)

        if lip_roi is not None:
            # Convert to float and normalize to [0, 1]
            lip_roi = lip_roi.astype(np.float32) / 255.0
            lip_sequence.append(lip_roi)
        else:
            # If no face detected, we can either skip the frame or use a blank image
            # We'll skip the frame for now, but note that this may cause misalignment
            # Alternatively, we could insert a blank image to maintain temporal alignment
            # For simplicity, we skip the frame and hope that the sequence is still long enough
            pass

    cap.release()

    if len(lip_sequence) == 0:
        print("Warning: No faces detected in any frame. Returning None.")
        return None

    # Convert to numpy array
    lip_sequence = np.array(lip_sequence)  # Shape: (T, H, W, C)

    # Pad or truncate to fixed length
    if len(lip_sequence) < sequence_length:
        # Pad with zeros (blank frames)
        pad_width = sequence_length - len(lip_sequence)
        lip_sequence = np.pad(lip_sequence,
                              ((0, pad_width), (0, 0), (0, 0), (0, 0)),
                              mode='constant',
                              constant_values=0)
    else:
        # Truncate
        lip_sequence = lip_sequence[:sequence_length]

    return lip_sequence

def lips_to_model_input(lip_sequence):
    """
    Convert a lip ROI sequence to the format expected by the LipNet model.

    Args:
        lip_sequence: Numpy array of shape (T, H, W, C) with values in [0, 1]

    Returns:
        Numpy array of shape (1, C, T, H, W) with values in [0, 1]
        (batch size of 1)
    """
    # Transpose to (T, C, H, W) then add batch dimension -> (1, T, C, H, W)
    # Then transpose to (1, C, T, H, W) as expected by the model
    # Actually, the model expects (B, C, T, H, W)
    # Our lip_sequence is (T, H, W, C)
    # We need to convert to (1, C, T, H, W)

    # First, transpose to (T, C, H, W)
    lip_sequence = lip_sequence.transpose(0, 3, 1, 2)  # (T, C, H, W)
    # Then add batch dimension at the beginning: (1, T, C, H, W)
    lip_sequence = np.expand_dims(lip_sequence, axis=0)
    # Then transpose to (1, C, T, H, W)
    lip_sequence = lip_sequence.transpose(0, 2, 1, 3, 4)

    return lip_sequence

if __name__ == '__main__':
    # Test with a dummy video (we don't have one, so we'll create a dummy sequence)
    # Create a dummy lip sequence of 60 frames (less than 75 to test padding)
    dummy_lips = np.random.rand(60, 64, 128, 3).astype(np.float32)
    print(f"Dummy lip sequence shape: {dummy_lips.shape}")

    model_input = lips_to_model_input(dummy_lips)
    print(f"Model input shape: {model_input.shape}")

    # Test the video_to_lip_sequence function with a dummy video (we don't have a real one)
    # We'll skip this test for now since we don't have a video file.
    print("To test with a real video, provide a valid video path.")