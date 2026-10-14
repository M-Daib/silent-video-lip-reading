# LipNet Paper Review

## Objective
- Maps a variable-length sequence of video frames to text using end-to-end learning.

## Input Representation
- Raw video frames (a variable-length sequence).

## Dataset
- GRID corpus.

## Experimental Protocol
- Sentence-level, overlapped speaker split task.

## Architecture
- Combines spatiotemporal convolutions with a recurrent network.

## CTC Objective
- Trained with the connectionist temporal classification loss.

## Vocabulary
- Not specified in the abstract, but from the implementation we know it includes:
  - Blank token (for CTC)
  - Space and uppercase letters A-Z (27 characters total)

## Sequence Decoding
- Implied by the CTC loss (standard CTC decoding: argmax followed by collapse of repeated characters and removal of blanks).

## Evaluation Metric
- Sentence-level accuracy.

## Reported Results
- Achieves 95.2% accuracy in sentence-level, overlapped speaker split task.
- Surpasses human lipreaders and the prior 86.4% word-level state-of-the-art result.

## Limitations
- None mentioned in the provided abstract.

## Implementation Details from VIPL Repo
- Input dimensions: 64x64 (height x width) with 3 channels (RGB)
- Temporal length: 75 frames (vid_padding)
- Architecture:
  - Conv3d(3, 32, kernel=(3,5,5), stride=(1,2,2), padding=(1,2,2))
  - MaxPool3d(kernel=(1,2,2), stride=(1,2,2))
  - Conv3d(32, 64, kernel=(3,5,5), stride=(1,1,1), padding=(1,2,2))
  - MaxPool3d(kernel=(1,2,2), stride=(1,2,2))
  - Conv3d(64, 96, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1))
  - MaxPool3d(kernel=(1,2,2), stride=(1,2,2))
  - GRU(96*4*8, 256, bidirectional=True)  # 96*4*8 = 3072
  - GRU(512, 256, bidirectional=True)
  - Linear(512, 28)  # 27 characters + 1 blank
- Dropout: 0.5 applied after each conv and GRU layer.

## Notes
- The VIPL PyTorch implementation achieves 13.3%/4.6% WER in unseen/overlapped testing, which they claim exceeds the original paper.
- The original paper reported 95.2% sentence-level accuracy (which is 4.8% WER) on overlapped speakers.
- The VIPL repo reports 4.6% WER on overlapped speakers, which is slightly better.
- The unseen speakers WER is 13.3% (86.7% sentence-level accuracy).