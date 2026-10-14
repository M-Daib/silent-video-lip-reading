"""
Training script for LipNet model.
Adapted from VIPL LipNet-PyTorch main.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Add paths
sys.path.append('temp_repos/LipNet-PyTorch')
sys.path.append('src')

from dataset import MyDataset
from model import LipNet
from options import opt as default_opt

def dataset2dataloader(dataset, num_workers=0, shuffle=True):
    """Create a DataLoader from dataset."""
    return DataLoader(
        dataset,
        batch_size=default_opt.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

def ctc_decode(y):
    """Decode CTC output to text."""
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

def test(model, net, opt):
    """Test the model on validation set."""
    with torch.no_grad():
        dataset = MyDataset(
            opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test'
        )

        print(f'num_test_data: {len(dataset.data)}')
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()

        for i_iter, input_data in enumerate(loader):
            vid = input_data.get('vid')
            txt = input_data.get('txt')
            vid_len = input_data.get('vid_len')
            txt_len = input_data.get('txt_len')

            y = net(vid)

            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))

            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0

                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 *'-'))
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))
                print(''.join(101 *'-'))

        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())

def train(model, net, opt):
    """Train the model."""
    dataset = MyDataset(
        opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train'
    )

    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(model.parameters(),
                    lr=opt.base_lr,
                    weight_decay=0.,
                    amsgrad=True)

    print(f'num_train_data: {len(dataset.data)}')
    crit = nn.CTCLoss()
    tic = time.time()

    train_wer = []
    writer = SummaryWriter()

    for epoch in range(opt.max_epoch):
        for i_iter, input_data in enumerate(loader):
            model.train()
            vid = input_data.get('vid')
            txt = input_data.get('txt')
            vid_len = input_data.get('vid_len')
            txt_len = input_data.get('txt_len')

            optimizer.zero_grad()
            y = net(vid)
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            tot_iter = i_iter + epoch*len(loader)

            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))

            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0

                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                (loss, wer, cer) = test(model, net, opt)
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter, show_lr(optimizer), loss, wer, cer))
                writer.add_scalar('val loss', loss, tot_iter)
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    writer.close()
                    return

    writer.close()

def show_lr(optimizer):
    """Show learning rate."""
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

if __name__ == '__main__':
    # Import options
    opt = __import__('options')

    # Set CUDA device
    if opt.gpu == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    print(f"Using GPU: {opt.gpu}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create model
    model = LipNet()

    # Load pretrained weights if specified
    if hasattr(opt, 'weights') and opt.weights:
        if os.path.exists(opt.weights):
            print(f"Loading pretrained weights from {opt.weights}")
            pretrained_dict = torch.load(opt.weights, map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            missed_params = [k for k in model_dict.keys() if k not in pretrained_dict.keys()]
            print(f'Loaded params/total params: {len(pretrained_dict)}/{len(model_dict)}')
            if missed_params:
                print(f'Missed params: {missed_params}')
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print(f"WARNING: Pretrained weights file not found: {opt.weights}")
            print("Training from scratch.")

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Set random seeds for reproducibility
    torch.manual_seed(opt.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.random_seed)

    # Start training
    print("Starting training...")
    train(model, model, opt)