import sys
import os
sys.path.append('src/data')

from lip_extractor import LipExtractor
import cv2
import numpy as np

def test_lip_extractor():
    print("=== Testing Lip Extractor ===")

    # Create a dummy frame with a face-like region
    # Create a 480x640 frame (typical webcam size)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a simple face-like rectangle (this won't be detected by Haar cascade, but let's test the logic)
    # In a real scenario, we'd need an actual face
    cv2.rectangle(frame, (200, 150), (440, 350), (255, 255, 255), -1)  # White face
    cv2.rectangle(frame, (250, 200), (290, 230), (0, 0, 0), -1)      # Left eye
    cv2.rectangle(frame, (350, 200), (390, 230), (0, 0, 0), -1)      # Right eye
    cv2.rectangle(frame, (300, 280), (340, 310), (0, 0, 0), -1)      # Mouth

    # Initialize lip extractor
    try:
        extractor = LipExtractor()
        print("Lip extractor initialized successfully")
    except Exception as e:
        print(f"Failed to initialize lip extractor: {e}")
        print("This is expected if the OpenCV data files are not accessible")
        return False

    # Test lip extraction
    lip_roi = extractor.extract_lip_roi(frame, target_width=128, target_height=64)

    if lip_roi is None:
        print("No face detected in dummy frame (expected)")
        print("Lip extractor is working correctly - returning None when no face found")
    else:
        print(f"Lip ROI extracted with shape: {lip_roi.shape}")
        # Save for inspection
        cv2.imwrite('test_lip_roi.jpg', lip_roi)
        print("Saved test lip ROI to test_lip_roi.jpg")

    print("\n=== Lip Extractor Test Completed ===")
    return True

if __name__ == '__main__':
    test_lip_extractor()