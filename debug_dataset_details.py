import sys
import os
sys.path.append('temp_repos/LipNet-PyTorch')
from fixed_dataset import MyDataset
import cv2
import numpy as np

# Create dummy data first
from create_dummy_data import create_dummy_data
create_dummy_data()

# Create dataset
dataset = MyDataset(
    video_path='dummy_data/lip/',
    anno_path='dummy_data/GRID_align_txt',
    file_list='dummy_data/lists/train.txt',
    vid_pad=75,
    txt_pad=200,
    phase='test'
)

print(f"Dataset size: {len(dataset)}")
item = dataset[0]
print(f"Video shape from dataset: {item['vid'].shape}")  # Should be (C, T, H, W)

# Let's look at one frame
frame_tensor = item['vid'][:, 0, :, :]  # First frame, all channels
print(f"First frame tensor shape: {frame_tensor.shape}")
frame_np = frame_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)  # CHW -> HWC
print(f"First frame numpy shape: {frame_np.shape}")

# Save it to inspect
cv2.imwrite('debug_first_frame.jpg', frame_np)
print("Saved debug_first_frame.jpg")

# Let's also check what the dataset thinks it's loading
vid_path = os.path.join('dummy_data/lip/', 's1/video/mpg_6000/bini5a')
print(f"Looking for frames in: {vid_path}")
if os.path.exists(vid_path):
    files = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
    print(f"Found {len(files)} JPG files")
    if files:
        first_file = os.path.join(vid_path, files[0])
        print(f"First file: {first_file}")
        img = cv2.imread(first_file)
        print(f"Original image shape: {img.shape}")

        # Now let's see what the resize does
        resized = cv2.resize(img, (128, 64))  # width, height
        print(f"After resize to (128, 64): {resized.shape}")
else:
    print("Path does not exist!")