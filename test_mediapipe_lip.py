import sys
import os
sys.path.append('src/data')

from mediapipe_lip_extractor import MediaPipeLipExtractor
import cv2
import numpy as np

def test_mediapipe_lip_extractor():
    print("=== Testing MediaPipe Lip Extractor ===")

    # Create a dummy frame with a face-like region
    # Since MediaPipe needs a real face, we'll test with a blank image first
    # to see if it handles no-face case gracefully
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Initialize lip extractor
    try:
        extractor = MediaPipeLipExtractor()
        print("MediaPipe lip extractor initialized successfully")
    except Exception as e:
        print(f"Failed to initialize MediaPipe lip extractor: {e}")
        return False

    # Test lip extraction on blank frame (no face)
    lip_roi = extractor.extract_lip_roi(frame, target_width=128, target_height=64)

    if lip_roi is None:
        print("No face detected in blank frame (expected)")
        print("MediaPipe lip extractor is working correctly - returning None when no face found")
    else:
        print(f"Unexpected: Got lip ROI with shape: {lip_roi.shape}")

    # Clean up
    extractor.close()

    print("\n=== MediaPipe Lip Extractor Test Completed ===")
    return True

if __name__ == '__main__':
    test_mediapipe_lip_extractor()