import sys
sys.path.append('C:/Users/HP/Downloads/Lep/temp_repos/LipNet-PyTorch')
from model import LipNet
import torch

print("Testing LipNet model...")
model = LipNet()
print("Model created.")

# Try to load pretrained weights (unseen speakers)
weights_path = 'C:/Users/HP/Downloads/Lep/temp_repos/LipNet-PyTorch/pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
try:
    pretrained_dict = torch.load(weights_path, map_location='cpu')
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(f"Loaded {len(pretrained_dict)} parameters from pretrained weights.")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Pretrained weights loaded successfully.")
except Exception as e:
    print(f"Error loading pretrained weights: {e}")
    print("Using random weights.")

# Create dummy input: (batch_size, channels, time, height, width)
# Based on the model: expects (B, C, T, H, W) where C=3, T=75 (from vid_padding in options), H=64, W=128
dummy_input = torch.randn(1, 3, 75, 64, 128)
print(f"Dummy input shape: {dummy_input.shape}")

# Forward pass
with torch.no_grad():
    output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Expected: (T, B, num_classes) -> (75, 1, 28)
print("Output sample (first time step, first batch):", output[0, 0, :5])

# Test CTC decoding logic (greedy)
# The output is logits, we need to apply softmax? Actually, the model returns logits.
# For CTC decoding, we take argmax over the class dimension.
y = output.argmax(-1)  # Shape: (T, B)
print(f"Argmax shape: {y.shape}")

# Convert to text using the dataset's method
sys.path.append('C:/Users/HP/Downloads/Lep/temp_repos/LipNet-PyTorch')
from dataset import MyDataset

# For each batch item, decode the sequence
for b in range(y.size(1)):
    sequence = y[:, b].cpu().numpy()
    decoded = MyDataset.ctc_arr2txt(sequence, start=1)
    print(f"Decoded text for batch {b}: '{decoded}'")

print("Test completed.")