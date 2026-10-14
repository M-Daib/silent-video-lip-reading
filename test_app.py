import sys
import os
sys.path.append('app')

# We need to set the environment variable for CUDA (if any) but we are using CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from app import model, extract_lip_sequence_from_video, lips_to_model_input, predict_sentence
from fixed_dataset import MyDataset
import torch
import numpy as np

def test_app_functions():
    print("=== Testing App Functions ===")

    # Test with the example video we have
    video_path = "grid_corpus/examples/id23_6000_priazn.mpg"
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return False

    print(f"Testing with video: {video_path}")

    # Test lip sequence extraction
    lip_sequence = extract_lip_sequence_from_video(video_path)
    if lip_sequence is None:
        print("ERROR: Failed to extract lip sequence")
        return False
    print(f"Lip sequence shape: {lip_sequence.shape}")
    print(f"Lip sequence min/max: {lip_sequence.min()}, {lip_sequence.max()}")

    # Test conversion to model input
    model_input = lips_to_model_input(lip_sequence)
    print(f"Model input shape: {model_input.shape}")

    # Test prediction
    sentence = predict_sentence(model, video_path)
    print(f"Predicted sentence: '{sentence}'")

    # Also get the ground truth for comparison
    # We know from the alignment file that the sentence is "PLACE RED IN A ZERO NOW"
    # But note: the alignment file includes silences at the beginning and end.
    # The dataset processing removes SIL and SP, so we get the words in between.
    # Let's check the alignment file we copied:
    align_path = "example_data/GRID_align_txt/s23/align/priazn.align"
    if os.path.exists(align_path):
        with open(align_path, 'r') as f:
            lines = f.readlines()
        # Extract the words (third column) and filter out SIL and SP
        words = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] not in ['SIL', 'SP']:
                words.append(parts[2])
        ground_truth = ' '.join(words)
        print(f"Ground truth from alignment: '{ground_truth}'")
    else:
        print("Alignment file not found for ground truth check")

    print("\n=== App Function Test Completed ===")
    return True

if __name__ == '__main__':
    test_app_functions()