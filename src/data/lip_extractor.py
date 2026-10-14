import cv2
import os
import numpy as np

class LipExtractor:
    def __init__(self, face_cascade_path=None):
        """
        Initialize the lip extractor with a face cascade classifier.
        If face_cascade_path is None, use the default OpenCV frontal face cascade.
        """
        if face_cascade_path is None:
            # Use the built-in frontal face cascade from OpenCV
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Failed to load face cascade from {face_cascade_path}")

    def extract_lip_roi(self, frame, target_width=128, target_height=64):
        """
        Extract lip region of interest from a single frame.

        Args:
            frame: Input frame (BGR format)
            target_width: Width of output lip ROI
            target_height: Height of output lip ROI

        Returns:
            Lip ROI image resized to (target_width, target_height) or None if no face found
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # No face detected, return None
            return None

        # Take the largest face (by area)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Extract mouth region:
        #   x: from face_x + 0.25*face_w to face_x + 0.75*face_w
        #   y: from face_y + 0.5*face_h to face_y + 0.75*face_h
        mouth_x1 = x + int(0.25 * w)
        mouth_x2 = x + int(0.75 * w)
        mouth_y1 = y + int(0.5 * h)
        mouth_y2 = y + int(0.75 * h)

        # Ensure the coordinates are within the frame
        mouth_x1 = max(0, mouth_x1)
        mouth_y1 = max(0, mouth_y1)
        mouth_x2 = min(frame.shape[1], mouth_x2)
        mouth_y2 = min(frame.shape[0], mouth_y2)

        # Extract mouth ROI
        mouth_roi = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

        # Resize to target dimensions
        lip_resized = cv2.resize(mouth_roi, (target_width, target_height))

        return lip_resized

    def process_video(self, video_path, output_dir, target_width=128, target_height=64):
        """
        Process a video file and save lip ROIs as images in output_dir.

        Args:
            video_path: Path to input video file
            output_dir: Directory to save lip ROI images (will be created if doesn't exist)
            target_width: Width of output lip ROI
            target_height: Height of output lip ROI

        Returns:
            Number of frames processed
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Extract lip ROI
            lip_roi = self.extract_lip_roi(frame, target_width, target_height)

            if lip_roi is not None:
                # Save lip ROI as image
                output_path = os.path.join(output_dir, f"{saved_count:04d}.jpg")
                cv2.imwrite(output_path, lip_roi)
                saved_count += 1
            else:
                # If no face detected, we can either skip the frame or use a blank image
                # For simplicity, we'll skip the frame (but this might cause misalignment)
                # Alternatively, we could insert a blank image to maintain temporal alignment
                pass

        cap.release()
        return saved_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract lip ROIs from video')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', required=True, help='Output directory for lip ROI images')
    parser.add_argument('--width', type=int, default=128, help='Width of output lip ROI')
    parser.add_argument('--height', type=int, default=64, help='Height of output lip ROI')

    args = parser.parse_args()

    extractor = LipExtractor()
    num_frames = extractor.process_video(args.video, args.output, args.width, args.height)
    print(f"Processed {num_frames} frames, saved lip ROIs to {args.output}")

if __name__ == '__main__':
    main()