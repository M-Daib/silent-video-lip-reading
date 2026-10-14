"""
Gradio demo for LipNet lip-reading system.
Allows users to upload a video and see the predicted sentence.
"""
import os
import cv2
import numpy as np
import torch
import gradio as gr

# Add paths
import sys
sys.path.append('temp_repos/LipNet-PyTorch')
sys.path.append('src')

from model import LipNet
from src.data.lip_extractor import LipExtractor
from src.data.video_to_lip import lips_to_model_input
from fixed_dataset import MyDataset  # For the text decoding utilities

def initialize_model():
    """Load the LipNet model with pretrained weights."""
    model = LipNet()
    weights_path = 'temp_repos/LipNet-PyTorch/pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'

    if os.path.exists(weights_path):
        pretrained_dict = torch.load(weights_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded pretrained weights: {len(pretrained_dict)} parameters")
    else:
        print("WARNING: Using random weights (pretrained weights not found)")

    model.eval()
    return model

def extract_lip_sequence_from_video(video_path, target_width=128, target_height=64, sequence_length=75):
    """
    Extract lip ROI sequence from a video file.
    Returns a numpy array of shape (sequence_length, target_height, target_width, 3) in [0, 1]
    or None if no faces detected.
    """
    extractor = LipExtractor()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    lip_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lip_roi = extractor.extract_lip_roi(frame, target_width, target_height)
        if lip_roi is not None:
            lip_roi = lip_roi.astype(np.float32) / 255.0
            lip_sequence.append(lip_roi)
        # else: skip frame

    cap.release()

    if len(lip_sequence) == 0:
        return None

    lip_sequence = np.array(lip_sequence)  # (T, H, W, C)

    # Pad or truncate to fixed length
    if len(lip_sequence) < sequence_length:
        pad_width = sequence_length - len(lip_sequence)
        lip_sequence = np.pad(lip_sequence,
                              ((0, pad_width), (0, 0), (0, 0), (0, 0)),
                              mode='constant',
                              constant_values=0)
    else:
        lip_sequence = lip_sequence[:sequence_length]

    return lip_sequence

def predict_sentence(model, video_path):
    """
    Run lip-reading prediction on a video file.
    Returns the predicted sentence as a string.
    """
    # Extract lip sequence
    lip_sequence = extract_lip_sequence_from_video(video_path)
    if lip_sequence is None:
        return "No face detected in video. Please try another video."

    # Convert to model input format
    model_input = lips_to_model_input(lip_sequence)  # (1, C, T, H, W)
    model_input_tensor = torch.from_numpy(model_input)

    # Run inference
    with torch.no_grad():
        output = model(model_input_tensor)  # (T, B, num_classes)
        pred_indices = output.argmax(-1)  # (T, B)
        pred_text = MyDataset.ctc_arr2txt(pred_indices[:, 0].cpu().numpy(), start=1)

    return pred_text.strip()

def process_video(video):
    """
    Gradio callback function.
    Input: video (either a file path or a tuple depending on Gradio version)
    Output: predicted sentence and optionally a visualization.
    """
    # Gradio provides the video as a temporary file path
    if isinstance(video, tuple):
        # Some versions of Gradio return a tuple (path, _) for video
        video_path = video[0]
    else:
        video_path = video

    if video_path is None or not os.path.exists(video_path):
        return "Error: No video provided or video file not found.", None

    try:
        # Predict sentence
        sentence = predict_sentence(model, video_path)

        # Optionally, we could also return a visualization of the lip ROIs
        # For simplicity, we just return the sentence.
        return sentence, None
    except Exception as e:
        return f"Error during processing: {str(e)}", None

# Initialize model once when the app loads
print("Initializing LipNet model...")
model = initialize_model()
print("Model initialized.")

# Create Gradio interface
with gr.Blocks(title="LipNet: Silent Video Lip Reading") as demo:
    gr.Markdown("# LipNet: Silent Video Lip Reading")
    gr.Markdown("""
    Upload a video file to see the predicted sentence from silent lip-reading.
    The model is based on the LipNet architecture and uses pretrained weights on the GRID corpus.
    Note: This is a demonstration with limited vocabulary (A-Z, space, and digits 0-9).
    """)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            submit_btn = gr.Button("Predict Sentence", variant="primary")

        with gr.Column():
            text_output = gr.Textbox(label="Predicted Sentence", placeholder="The predicted sentence will appear here...")
            # We could add a video output for visualization later
            # video_output = gr.Video(label="Processed Video (Lip ROIs)")

    submit_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[text_output]  # , video_output
    )

    gr.Markdown("""
    ### How it works
    1. Video is uploaded and frames are extracted.
    2. Face detection is performed on each frame using OpenCV Haar cascades.
    3. A mouth region of interest is extracted (central horizontal strip of the face).
    4. The lip ROIs are resized to 64x128 pixels and normalized.
    5. The sequence of lip ROIs is passed through the LipNet model (spatiotemporal CNN + GRU + CTC).
    6. The model outputs a sequence of character probabilities, which are decoded using greedy CTC decoding.
    7. The final predicted sentence is displayed.

    ### Limitations
    - The lip extraction uses a simple heuristic (not facial landmarks), which may not be accurate for all poses and lighting conditions.
    - The model was trained on the GRID corpus, which consists of highly controlled, frontal-facing videos with consistent lighting.
    - Performance may degrade significantly on unconstrained videos.
    - The vocabulary is limited to uppercase letters A-Z, space, and digits 0-9 (as in the GRID corpus).
    """)

if __name__ == "__main__":
    demo.launch()