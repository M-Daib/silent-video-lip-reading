import sys
import torch
sys.path.append('temp_repos/LipNet-PyTorch')
from model import LipNet

# Create a dummy input matching the expected format from the dataset
# Based on the dataset: (C, T, H, W) = (3, 75, 64, 128) after transpose
# But the model expects (B, C, T, H, W)
dummy_input = torch.randn(1, 3, 75, 64, 128)
print(f"Input shape: {dummy_input.shape}")

model = LipNet()

# Let's manually go through the layers to see the shapes
x = dummy_input
print(f"After conv1: {model.conv1(x).shape}")
x = model.relu(model.conv1(dummy_input))
print(f"After relu1: {x.shape}")
x = model.dropout3d(x)
print(f"After dropout3d1: {x.shape}")
x = model.pool1(x)
print(f"After pool1: {x.shape}")

x = model.relu(model.conv2(x))
print(f"After relu2: {x.shape}")
x = model.dropout3d(x)
print(f"After dropout3d2: {x.shape}")
x = model.pool2(x)
print(f"After pool2: {x.shape}")

x = model.relu(model.conv3(x))
print(f"After relu3: {x.shape}")
x = model.dropout3d(x)
print(f"After dropout3d3: {x.shape}")
x = model.pool3(x)
print(f"After pool3: {x.shape}")

# Now permute and reshape for GRU
x = x.permute(2, 0, 1, 3, 4).contiguous()
print(f"After permute: {x.shape}")
x = x.view(x.size(0), x.size(1), -1)
print(f"After view: {x.shape}")

# This should be the input to the GRU
print(f"GRU expected input size: {model.gru1.input_size}")
print(f"Actual input size: {x.shape[-1]}")