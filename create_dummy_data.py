import os
import numpy as np
import cv2

def create_dummy_data(base_dir='dummy_data'):
    # Create directory structure
    lip_dir = os.path.join(base_dir, 'lip')
    align_dir = os.path.join(base_dir, 'GRID_align_txt')
    lists_dir = os.path.join(base_dir, 'lists')

    os.makedirs(os.path.join(lip_dir, 's1', 'video', 'mpg_6000', 'bini5a'), exist_ok=True)
    os.makedirs(os.path.join(align_dir, 's1', 'align'), exist_ok=True)
    os.makedirs(lists_dir, exist_ok=True)

    # Generate 75 dummy frames (128x64x3) with random noise
    # Based on the model: height=64, width=128 (but cv2.resize takes (width, height))
    frame_count = 75
    height, width = 64, 128
    for i in range(frame_count):
        # Create a random frame (uint8)
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # Save as JPEG
        frame_path = os.path.join(lip_dir, 's1', 'video', 'mpg_6000', 'bini5a', f'{i:04d}.jpg')
        cv2.imwrite(frame_path, frame)

    # Create alignment file for "BIN RED AT 9 NOW"
    # Each word gets 15 frames (total 75)
    alignment_content = """0 15 BIN
15 30 RED
30 45 AT
45 60 9
60 75 NOW
"""
    align_path = os.path.join(align_dir, 's1', 'align', 'bini5a.align')
    with open(align_path, 'w') as f:
        f.write(alignment_content)

    # Create train and val lists (we'll use the same video for both for testing)
    train_list_path = os.path.join(lists_dir, 'train.txt')
    val_list_path = os.path.join(lists_dir, 'val.txt')

    with open(train_list_path, 'w') as f:
        f.write('s1/video/mpg_6000/bini5a\n')

    with open(val_list_path, 'w') as f:
        f.write('s1/video/mpg_6000/bini5a\n')

    print(f"Dummy data created in {base_dir}")
    print(f"Lip ROIs: {os.path.join(lip_dir, 's1', 'video', 'mpg_6000', 'bini5a')}")
    print(f"Alignment file: {align_path}")
    print(f"Train list: {train_list_path}")
    print(f"Val list: {val_list_path}")

if __name__ == '__main__':
    create_dummy_data()