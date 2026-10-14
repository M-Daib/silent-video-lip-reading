import cv2
import os

video_path = "grid_corpus/examples/id2_vcd_swwp2s.mpg"
if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}")
else:
    print(f"Video file found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video with OpenCV")
    else:
        print("Video opened successfully with OpenCV")
        ret, frame = cap.read()
        if ret:
            print(f"First frame shape: {frame.shape}")
            # Save the first frame for inspection
            cv2.imwrite("first_frame.jpg", frame)
            print("Saved first frame as first_frame.jpg")
        else:
            print("Failed to read first frame")
        cap.release()