"""
Validation script for LipNet model.
"""
import torch
import torch.nn as nn
import time
import sys
from torch.utils.data import DataLoader

# Add paths
sys.path.append('temp_repos/LipNet-PyTorch')
sys.path.append('src')

from dataset import MyDataset
from model import LipNet

def dataset2dataloader(dataset, num_workers=0, shuffle=False):
    """Create a DataLoader from dataset."""
    return DataLoader(
        dataset,
        batch_size=1,  # Validation typically uses batch size 1
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

def ctc_decode(y):
    """Decode CTC output to text."""
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

def validate(model, opt):
    """Validate the model on validation set."""
    with torch.no_grad():
        dataset = MyDataset(
            opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test'
        )

        print(f'num_val_data: {len(dataset.data)}')
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer_scores = []
        cer_scores = []
        crit = nn.CTCLoss()
        tic = time.time()

        all_predictions = []
        all_ground_truths = []

        for i_iter, input_data in enumerate(loader):
            vid = input_data.get('vid')
            txt = input_data.get('txt')
            vid_len = input_data.get('vid_len')
            txt_len = input_data.get('txt_len')

            y = model(vid)

            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer_batch = MyDataset.wer(pred_txt, truth_txt)
            cer_batch = MyDataset.cer(pred_txt, truth_txt)

            wer_scores.extend(wer_batch)
            cer_scores.extend(cer_batch)

            all_predictions.extend(pred_txt)
            all_ground_truths.extend(truth_txt)

            if(i_iter % 10 == 0):  # Print every 10 batches
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0
                print('valid_iter={},eta={},loss={},wer={},cer={}'.format(
                    i_iter, eta, loss,
                    np.array(wer_scores).mean(),
                    np.array(cer_scores).mean()))

        avg_loss = np.array(loss_list).mean()
        avg_wer = np.array(wer_scores).mean()
        avg_cer = np.array(cer_scores).mean()

        # Calculate sentence-level accuracy (exact match)
        correct_sentences = sum(1 for pred, truth in zip(all_predictions, all_ground_truths) if pred.strip() == truth.strip())
        sentence_accuracy = correct_sentences / len(all_predictions) if len(all_predictions) > 0 else 0

        print(''.join(101*'='))
        print('VALIDATION RESULTS')
        print(''.join(101*'='))
        print(f'Average Loss: {avg_loss:.6f}')
        print(f'Average WER: {avg_wer:.6f} ({avg_wer*100:.2f}%)')
        print(f'Average CER: {avg_cer:.6f} ({avg_cer*100:.2f}%)')
        print(f'Sentence-level Accuracy: {sentence_accuracy:.6f} ({sentence_accuracy*100:.2f}%)')
        print(f'Total Validation Samples: {len(all_predictions)}')
        print(''.join(101*'='))

        # Show some examples
        print('\nEXAMPLE PREDICTIONS:')
        print('-' * 80)
        for i in range(min(10, len(all_predictions))):
            pred = all_predictions[i].strip()
            truth = all_ground_truths[i].strip()
            status = '✓' if pred == truth else '✗'
            print(f'{status} Pred: "{pred}" | Truth: "{truth}"')

        return {
            'loss': avg_loss,
            'wer': avg_wer,
            'cer': avg_cer,
            'sentence_accuracy': sentence_accuracy,
            'predictions': all_predictions,
            'ground_truths': all_ground_truths
        }

if __name__ == '__main__':
    import numpy as np
    # Import options
    opt = __import__('options')

    # Set CUDA device
    if opt.gpu == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # Create model
    model = LipNet()

    # Load pretrained weights if specified
    if hasattr(opt, 'weights') and opt.weights:
        if os.path.exists(opt.weights):
            print(f"Loading pretrained weights from {opt.weights}")
            pretrained_dict = torch.load(opt.weights, map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded pretrained weights successfully")
        else:
            print(f"WARNING: Pretrained weights file not found: {opt.weights}")

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Run validation
    results = validate(model, opt)