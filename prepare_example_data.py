import os
import cv2
import numpy as np
import sys
sys.path.append('src/data')
from lip_extractor import LipExtractor

def extract_lips_from_video(video_path, output_dir, target_width=128, target_height=64):
    """
    Extract lip ROIs from a video and save them as images in output_dir.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save lip ROI images (will be created if doesn't exist)
        target_width: Width of output lip ROI
        target_height: Height of output lip ROI

    Returns:
        Number of frames processed (lip ROIs saved)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lip extractor
    extractor = LipExtractor()

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
        lip_roi = extractor.extract_lip_roi(frame, target_width, target_height)

        if lip_roi is not None:
            # Save lip ROI as image
            output_path = os.path.join(output_dir, f"{saved_count:04d}.jpg")
            cv2.imwrite(output_path, lip_roi)
            saved_count += 1
        # else: skip frame (no face detected)

    cap.release()
    return saved_count

def main():
    # Define the example data we have
    base_dir = "example_data"

    # We'll create the directory structure under base_dir
    lip_dir = os.path.join(base_dir, "lip")
    align_dir = os.path.join(base_dir, "GRID_align_txt")
    lists_dir = os.path.join(base_dir, "lists")

    os.makedirs(lip_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(lists_dir, exist_ok=True)

    # Define the speakers and sentences we have
    # From the downloaded files:
    #   Speaker s2: sentence swwp2s
    #   Speaker s23: sentence priazn
    speakers = {
        's2': ['swwp2s'],
        's23': ['priazn']
    }

    # Define the video files we have (we'll use the high quality ones: id2_6000_*.mpg)
    video_files = {
        's2': {
            'swwp2s': "grid_corpus/examples/id2_6000_swwp2s.mpg"
        },
        's23': {
            'priazn': "grid_corpus/examples/id23_6000_priazn.mpg"
        }
    }

    # Define the alignment files we have
    alignment_files = {
        's2': {
            'swwp2s': "grid_corpus/examples/swwp2s.align"
        },
        's23': {
            'priazn': "grid_corpus/examples/priazn.align"
        }
    }

    # Process each speaker and sentence
    for speaker, sentences in speakers.items():
        for sentence in sentences:
            print(f"Processing speaker {speaker}, sentence {sentence}")

            # 1. Extract lips from the high quality video
            video_path = video_files[speaker][sentence]
            lip_output_dir = os.path.join(lip_dir, speaker, "video", "mpg_6000", sentence)
            print(f"  Extracting lips from {video_path} to {lip_output_dir}")
            num_lips = extract_lips_from_video(video_path, lip_output_dir)
            print(f"  Saved {num_lips} lip ROIs")

            # 2. Copy the alignment file
            align_input_path = alignment_files[speaker][sentence]
            align_output_dir = os.path.join(align_dir, speaker, "align")
            os.makedirs(align_output_dir, exist_ok=True)
            align_output_path = os.path.join(align_output_dir, f"{sentence}.align")
            print(f"  Copying alignment from {align_input_path} to {align_output_path}")
            # Read the alignment file and write it to the output directory
            with open(align_input_path, 'r') as f_in:
                align_content = f_in.read()
            with open(align_output_path, 'w') as f_out:
                f_out.write(align_content)
            print(f"  Alignment copied.")

    # Create the train and val lists
    # We'll use speaker s2 for training and speaker s23 for validation
    train_speakers = ['s2']
    val_speakers = ['s23']

    train_list_path = os.path.join(lists_dir, "train.txt")
    val_list_path = os.path.join(lists_dir, "val.txt")

    with open(train_list_path, 'w') as f:
        for speaker in train_speakers:
            for sentence in speakers[speaker]:
                # The format expected by the dataset: <speaker>/video/mpg_6000/<sentence>
                line = f"{speaker}/video/mpg_6000/{sentence}\n"
                f.write(line)

    with open(val_list_path, 'w') as f:
        for speaker in val_speakers:
            for sentence in speakers[speaker]:
                line = f"{speaker}/video/mpg_6000/{sentence}\n"
                f.write(line)

    print(f"Created train list at {train_list_path}")
    print(f"Created val list at {val_list_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Lip ROIs directory: {lip_dir}")
    print(f"Alignment directory: {align_dir}")
    print(f"Train list: {train_list_path}")
    print(f"Val list: {val_list_path}")

    # Check the lip directories
    for speaker in ['s2', 's23']:
        for sentence in speakers[speaker]:
            lip_dir_full = os.path.join(lip_dir, speaker, "video", "mpg_6000", sentence)
            if os.path.exists(lip_dir_full):
                num_frames = len([f for f in os.listdir(lip_dir_full) if f.endswith('.jpg')])
                print(f"Speaker {speaker}, sentence {sentence}: {num_frames} lip ROIs")
            else:
                print(f"Speaker {speaker}, sentence {sentence}: directory not found")

    print("\nDone!")

if __name__ == '__main__':
    main()