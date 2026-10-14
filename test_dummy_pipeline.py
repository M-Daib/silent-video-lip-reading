import sys
import os
import torch
import numpy as np

# Add paths
sys.path.append('temp_repos/LipNet-PyTorch')
sys.path.append('.')

from model import LipNet
from dataset import MyDataset

def test_with_dummy_data():
    print("=== Testing LipNet with Dummy Data ===")

    # Create dummy data first
    from create_dummy_data import create_dummy_data
    create_dummy_data()

    # Load model
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
        print("WARNING: Using random weights")

    model.eval()

    # Create dataset with dummy data
    dataset = MyDataset(
        video_path='dummy_data/lip/',
        anno_path='dummy_data/GRID_align_txt',
        file_list='dummy_data/lists/train.txt',
        vid_pad=75,
        txt_pad=200,
        phase='test'  # No augmentation
    )

    print(f"Dataset size: {len(dataset)}")

    # Get one item
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    print(f"Video shape: {item['vid'].shape}")  # Should be (C, T, H, W) after transpose in dataset
    print(f"Text: {item['txt']}")
    print(f"Text length: {item['txt_len']}")
    print(f"Video length: {item['vid_len']}")

    # Prepare for model: dataset returns (C, T, H, W) but model expects (B, C, T, H, W)
    video_tensor = item['vid'].unsqueeze(0)  # Add batch dimension: (1, C, T, H, W)
    print(f"Model input shape: {video_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(video_tensor)
    print(f"Model output shape: {output.shape}")  # Should be (T, B, num_classes)

    # Decode using greedy CTC
    pred_indices = torch.argmax(output, dim=-1)  # (T, B)
    print(f"Prediction indices shape: {pred_indices.shape}")

    # Convert to text for each batch item
    for b in range(pred_indices.size(1)):
        seq = pred_indices[:, b].cpu().numpy()
        text = MyDataset.ctc_arr2txt(seq, start=1)
        print(f"Decoded text (batch {b}): '{text}'")
        print(f"Ground truth text: '{MyDataset.arr2txt(item['txt'][:item['txt_len']].cpu().numpy(), start=1)}'")

    print("\n=== Dummy Data Test Completed ===")
    return True

if __name__ == '__main__':
    test_with_dummy_data()