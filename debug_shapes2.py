import sys
import torch
sys.path.append('temp_repos/LipNet-PyTorch')
from model import LipNet

# Let's compute the output shape of the conv layers for an input of (1, 3, 75, 64, 128)
model = LipNet()
x = torch.randn(1, 3, 75, 64, 128)
print(f"Input: {x.shape}")

# Conv1: (3, 32, kernel=(3,5,5), stride=(1,2,2), padding=(1,2,2))
# Output: (T_out = (T + 2*pad - kernel)/stride + 1) = (75 + 2*2 - 3)/1 + 1 = (75+4-3)+1 = 77? Let's just trust the model.
# Actually, let's do it step by step with print.

x = model.conv1(x)
print(f"After conv1: {x.shape}")
x = model.relu(x)
x = model.dropout3d(x)
x = model.pool1(x)  # (1,2,2)
print(f"After pool1: {x.shape}")

x = model.conv2(x)
print(f"After conv2: {x.shape}")
x = model.relu(x)
x = model.dropout3d(x)
x = model.pool2(x)  # (1,2,2)
print(f"After pool2: {x.shape}")

x = model.conv3(x)
print(f"After conv3: {x.shape}")
x = model.relu(x)
x = model.dropout3d(x)
x = model.pool3(x)  # (1,2,2)
print(f"After pool3: {x.shape}")

# Now permute: (B, C, T, H, W) -> (T, B, C, H, W)
x = x.permute(2, 0, 1, 3, 4).contiguous()
print(f"After permute: {x.shape}")

# Then view: (T, B, C*H*W)
x = x.view(x.size(0), x.size(1), -1)
print(f"After view: {x.shape}")

# The GRU input size is the last dimension: 96*4*8? Let's see what we have.
# After pool3: we have (B, C, T, H, W) = (1, 96, T', H', W')
# We need to compute T', H', W' after the three poolings (each pool is (1,2,2) stride (1,2,2))
# Initial T=75, H=64, W=128
# After pool1: T1 = (75 + 2*2 - 2)//2 + 1? Actually, the pool is (1,2,2) so T dimension is not pooled (size 1).
# Let's compute:
# For T: kernel=1, stride=1, pad=0 -> T_out = T_in
# For H: kernel=2, stride=2, pad=0 -> H_out = H_in // 2
# For W: kernel=2, stride=2, pad=0 -> W_out = W_in // 2
# So after pool1: T=75, H=32, W=64
# After pool2: T=75, H=16, W=32
# After pool3: T=75, H=8, W=16
# Then permute: (T, B, C, H, W) = (75, 1, 96, 8, 16)
# Then view: (75, 1, 96*8*16) = (75, 1, 12288) -> but wait, 96*8*16 = 12288, not 3072.
# So I made a mistake: the model's GRU input size is 96*4*8? Let's look at the model again.

# In the model.py:
# self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
# self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
# So after conv3 (no stride change) and pool3 (stride 2 in H and W) we get:
# After conv3: (B, 96, T, H, W) where T unchanged, H and W unchanged by conv (padding=1)
# Then pool3: T unchanged, H->H/2, W->W/2.
# So let's recalculate from the beginning with the correct pooling.

# Let's do it step by step with the actual model to avoid mistakes.