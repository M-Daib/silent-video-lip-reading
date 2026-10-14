import os

# Simulate what the dataset does
video_path = 'dummy_data/lip/'
line = 's1/video/mpg_6000/bini5a'

vid = os.path.join(video_path, line.strip())
print(f'vid: {repr(vid)}')

# Normalize path
vid_norm = os.path.normpath(vid)
print(f'vid_norm: {repr(vid_norm)}')

# Split
items = vid_norm.split(os.path.sep)
print(f'items: {items}')
print(f'len(items): {len(items)}')

if len(items) >= 4:
    print(f'items[-4]: {items[-4]}')
    print(f'items[-1]: {items[-1]}')
else:
    print('Not enough items - need at least 4')
    print(f'We have: {items}')

# Let's see what the actual directory structure looks like
print("\nActual directory structure:")
for root, dirs, files in os.walk('dummy_data'):
    level = root.replace('dummy_data', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')