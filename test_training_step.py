import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Add paths
sys.path.append('temp_repos/LipNet-PyTorch')
sys.path.append('.')

from model import LipNet
from fixed_dataset import MyDataset
from tensorboardX import SummaryWriter
import numpy as np

def test_training_step():
    print("=== Testing Training Step with Dummy Data ===")

    # Load options (we'll create a minimal version)
    class Opt:
        gpu = '-1'  # Use CPU since we don't have CUDA working properly
        random_seed = 0
        base_lr = 2e-5
        vid_padding = 75
        txt_padding = 200
        batch_size = 2  # Small batch for testing
        display = 1
        test_step = 50
        max_epoch = 1  # Just one epoch for testing
        is_optimize = True
        video_path = 'dummy_data/lip/'
        anno_path = 'dummy_data/GRID_align_txt'
        train_list = 'dummy_data/lists/train.txt'
        val_list = 'dummy_data/lists/val.txt'
        save_prefix = 'weights/LipNet_test'

    opt = Opt()

    # Set environment variable for GPU (we'll use -1 for CPU)
    if opt.gpu == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    print(f"Using GPU setting: {opt.gpu}")
    print(f"CUDA visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    # Create model
    model = LipNet()
    print(f"Model created: {type(model)}")

    # Load pretrained weights if available
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

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Create datasets
    train_dataset = MyDataset(
        video_path=opt.video_path,
        anno_path=opt.anno_path,
        file_list=opt.train_list,
        vid_pad=opt.vid_padding,
        txt_pad=opt.txt_padding,
        phase='train'
    )

    val_dataset = MyDataset(
        video_path=opt.video_path,
        anno_path=opt.anno_path,
        file_list=opt.val_list,
        vid_pad=opt.vid_padding,
        txt_pad=opt.txt_padding,
        phase='test'
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
    def dataset2dataloader(dataset, num_workers=0, shuffle=True):  # Set num_workers=0 for simplicity
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False
        )

    train_loader = dataset2dataloader(train_dataset, shuffle=True)
    val_loader = dataset2dataloader(val_dataset, shuffle=False)

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, weight_decay=0., amsgrad=True)
    crit = nn.CTCLoss()

    # Test one training batch
    print("\n--- Testing one training batch ---")
    model.train()
    for i_iter, input_data in enumerate(train_loader):
        if i_iter >= 1:  # Just test one batch
            break

        vid = input_data.get('vid').to(device)
        txt = input_data.get('txt').to(device)
        vid_len = input_data.get('vid_len').to(device)
        txt_len = input_data.get('txt_len').to(device)

        print(f"Batch shapes - vid: {vid.shape}, txt: {txt.shape}, vid_len: {vid_len.shape}, txt_len: {txt_len.shape}")

        optimizer.zero_grad()
        y = model(vid)
        print(f"Model output shape: {y.shape}")  # Should be (T, B, num_classes)

        # Compute loss
        loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
        print(f"Loss: {loss.item()}")

        # Backward pass
        loss.backward()
        optimizer.step()

        # Decode predictions
        with torch.no_grad():
            pred_indices = y.argmax(-1)  # (T, B)
            pred_texts = [MyDataset.ctc_arr2txt(pred_indices[:, b].cpu().numpy(), start=1)
                         for b in range(pred_indices.size(1))]

            truth_texts = [MyDataset.arr2txt(txt[b][:txt_len[b]].cpu().numpy(), start=1)
                          for b in range(txt.size(0))]

            print(f"Predictions: {pred_texts}")
            print(f"Ground truths: {truth_texts}")

        print("Training step completed successfully!")
        break  # Just one batch

    # Test one validation batch
    print("\n--- Testing one validation batch ---")
    model.eval()
    with torch.no_grad():
        for i_iter, input_data in enumerate(val_loader):
            if i_iter >= 1:  # Just test one batch
                break

            vid = input_data.get('vid').to(device)
            txt = input_data.get('txt').to(device)
            vid_len = input_data.get('vid_len').to(device)
            txt_len = input_data.get('txt_len').to(device)

            y = model(vid)
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            print(f"Validation Loss: {loss.item()}")

            # Decode predictions
            pred_indices = y.argmax(-1)  # (T, B)
            pred_texts = [MyDataset.ctc_arr2txt(pred_indices[:, b].cpu().numpy(), start=1)
                         for b in range(pred_indices.size(1))]

            truth_texts = [MyDataset.arr2txt(txt[b][:txt_len[b]].cpu().numpy(), start=1)
                          for b in range(txt.size(0))]

            print(f"Validation Predictions: {pred_texts}")
            print(f"Validation Ground truths: {truth_texts}")
            break  # Just one batch

    print("\n=== Training Step Test Completed Successfully ===")
    return True

if __name__ == '__main__':
    test_training_step()