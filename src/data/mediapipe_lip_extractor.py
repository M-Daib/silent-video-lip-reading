import cv2
import mediapipe as mp
import numpy as np
import os

class MediaPipeLipExtractor:
    def __init__(self, static_image_mode=False, max_num_faces=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the MediaPipe face mesh extractor for lip landmark detection.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Lip landmarks indices (based on MediaPipe face mesh)
        # Outer lip landmarks
        self.OUTER_LIP_LANDMARKS = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
            185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]

        # Inner lip landmarks
        self.INNER_LIP_LANDMARKS = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409
        ]

        # Combine outer and inner lips for a comprehensive lip region
        self.LIP_LANDMARKS = list(set(self.OUTER_LIP_LANDMARKS + self.INNER_LIP_LANDMARKS))

    def extract_lip_roi(self, frame, target_width=128, target_height=64):
        """
        Extract lip region of interest from a single frame using MediaPipe face mesh.

        Args:
            frame: Input frame (BGR format)
            target_width: Width of output lip ROI
            target_height: Height of output lip ROI

        Returns:
            Lip ROI image resized to (target_width, target_height) or None if no face found
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # No face detected
            return None

        # Get the first face (assuming single face in frame)
        face_landmarks = results.multi_face_landmarks[0]

        # Extract lip landmark coordinates
        h, w, _ = frame.shape
        lip_points = []

        for landmark_id in self.LIP_LANDMARKS:
            landmark = face_landmarks.landmark[landmark_id]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lip_points.append([x, y])

        lip_points = np.array(lip_points)

        # Calculate bounding box around lips
        x_min, y_min = lip_points.min(axis=0)
        x_max, y_max = lip_points.max(axis=0)

        # Add some padding around the lips
        padding = 0.3  # 30% padding
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - padding * width))
        y_min = max(0, int(y_min - padding * height))
        x_max = min(w, int(x_max + padding * width))
        y_max = min(h, int(y_max + padding * height))

        # Extract lip ROI
        lip_roi = frame[y_min:y_max, x_min:x_max]

        # Check if we got a valid ROI
        if lip_roi.size == 0:
            return None

        # Resize to target dimensions
        lip_resized = cv2.resize(lip_roi, (target_width, target_height))

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
            # else: skip frame (no face detected)

        cap.release()
        return saved_count

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract lip ROIs from video using MediaPipe')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', required=True, help='Output directory for lip ROI images')
    parser.add_argument('--width', type=int, default=128, help='Width of output lip ROI')
    parser.add_argument('--height', type=int, default=64, help='Height of output lip ROI')

    args = parser.parse_args()

    extractor = MediaPipeLipExtractor()
    try:
        num_frames = extractor.process_video(args.video, args.output, args.width, args.height)
        print(f"Processed {num_frames} frames, saved lip ROIs to {args.output}")
    finally:
        extractor.close()

if __name__ == '__main__':
    main()