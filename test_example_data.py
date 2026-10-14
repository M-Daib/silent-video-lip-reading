import sys
import os
import torch
sys.path.append('temp_repos/LipNet-PyTorch')
sys.path.append('src')

from model import LipNet
from dataset import MyDataset

def test_with_example_data():
    print("=== Testing LipNet with Example GRID Data ===")

    # Create model
    model = LipNet()

    # Load pretrained weights (unseen speakers)
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

    # Create dataset with our example data
    dataset = MyDataset(
        video_path='example_data/lip/',
        anno_path='example_data/GRID_align_txt',
        file_list='example_data/lists/val.txt',  # Using validation list which has s23/priazn
        vid_pad=75,
        txt_pad=200,
        phase='test'
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return False

    # Get one item
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    print(f"Video shape: {item['vid'].shape}")  # Should be (C, T, H, W)
    print(f"Text: {item['txt']}")
    print(f"Text length: {item['txt_len']}")
    print(f"Video length: {item['vid_len']}")

    # Check what the text actually is
    txt_np = item['txt'][:item['txt_len']].cpu().numpy()
    print(f"Text numpy: {txt_np}")
    print(f"Text decoded (using arr2txt): {''.join([MyDataset.letters[n-1] for n in txt_np if n >= 1])}")

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

        # Get ground truth text
        gt_txt = item['txt'][:item['txt_len']].cpu().numpy()
        gt_text = MyDataset.arr2txt(gt_txt, start=1)
        print(f"Ground truth text: '{gt_text}'")

    print("\n=== Example Data Test Completed ===")
    return True

if __name__ == '__main__':
    test_with_example_data()