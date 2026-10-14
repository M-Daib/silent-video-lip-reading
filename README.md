# Silent Video Lip Reading Using Deep Learning

This project implements an end-to-end lip-reading system based on the LipNet architecture, using the GRID corpus for training and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [Example Workflow](#example-workflow)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Presentation](#presentation)
- [References](#references)

## Project Overview
The system takes silent video as input and outputs predicted sentences through the following pipeline:
1. Video decoding and frame extraction
2. Face detection and lip region of interest (ROI) extraction
3. Preprocessing (resizing, normalization)
4. Spatiotemporal convolutional neural network (LipNet)
5. Connectionist Temporal Classification (CTC) for sequence modeling
6. Greedy decoding to produce text predictions

## Key Features
- End-to-end lip-reading pipeline from video to text
- Based on the LipNet architecture (spatiotemporal CNN + bidirectional GRU + CTC)
- Utilizes the GRID corpus for training and evaluation
- Includes a Gradio-based interactive demo for video upload and prediction
- Provides scripts for training, validation, and testing
- Modular and well-documented codebase

## System Architecture
```
Input Video
    ↓
Frame Extraction
    ↓
Face Detection (OpenCV Haar Cascades)
    ↓
Lip ROI Extraction (Heuristic: central horizontal strip of lower face)
    ↓
Preprocessing (Resize to 64x128 pixels, Normalize pixel values to [0, 1])
    ↓
LipNet Model
    ├── Spatiotemporal Convolutional Layers
    ├── Bidirectional GRU Layers
    └── Fully Connected Layer (28 outputs: 26 letters + space + blank)
    ↓
CTC Decoding (Greedy: argmax → collapse repeats → remove blanks)
    ↓
Predicted Sentence (Text Output)
```

## Technologies Used
- **Language**: Python 3.12
- **Deep Learning**: PyTorch (CPU version due to DLL issues on Windows; GPU version recommended for training)
- **Computer Vision**: OpenCV (for video processing and face detection)
- **Future Vision**: MediaPipe (for accurate lip landmark extraction - planned improvement)
- **Web Interface**: Gradio (for interactive demo)
- **Other Libraries**: NumPy, Pandas, Matplotlib, Tqdm, Jiwer, Scikit-learn, Editdistance, TensorBoardX

## Project Structure
```
silent-video-lip-reading/
├── README.md
├── requirements.txt
├── presentation/
│   └── Silent_Video_Lip_Reading_Presentation.pptx
├── app/
│   └── app.py                  # Gradio demo application
├── src/
│   ├── data/
│   │   ├── lip_extractor.py    # Face detection and lip ROI extraction
│   │   ├── video_reader.py     # Video frame reading utility
│   │   └── video_to_lip.py     # Convert video to lip ROI sequence
│   ├── models/
│   │   └── model.py            # LipNet model definition
│   ├── training/
│   │   ├── train.py            # Training script
│   │   └── validate.py         # Validation script
│   └── utils/
├── data/                       # Directory for data (lip ROIs, alignments, lists)
│   ├── lip/                    # Lip ROI images (organized by speaker/sentence)
│   │   └── <speaker>/
│   │       └── video/
│   │           └── mpg_6000/
│   │               └── <sentence>/
│   │                   ├── 0000.jpg
│   │                   ├── 0001.jpg
│   │                   └── ...
│   ├── GRID_align_txt/         # Alignment files (organized by speaker)
│   │   └── <speaker>/
│   │       └── align/
│   │           └── <sentence>.align
│   └── lists/                  # Train and validation lists
│       ├── train.txt
│       └── val.txt
├── reports/
│   ├── initial_audit.md
│   ├── final_execution_report.md
│   └── ...                     # Other reports
├── temp_repos/
│   └── LipNet-PyTorch/         # Cloned VIPL LipNet-PyTorch repository
│       ├── model.py
│       ├── dataset.py
│       ├── main.py
│       ├── options.py
│       └── pretrain/           # Pretrained weights
│           ├── LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt
│           └── LipNet_overlap_loss_0.07664558291435242_wer_0.04644484056248762_cer_0.019676921477851092.pt
├── tests/                      # Unit and integration tests
└── scripts/                    # Auxiliary scripts (e.g., data preparation)
```

## Installation Instructions
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd silent-video-lip-reading
   ```

2. **Install dependencies**:
   It is recommended to create a virtual environment first.
   ```bash
   pip install -r requirements.txt
   ```
   Note: The `requirements.txt` includes PyTorch (CPU version). For GPU support, install the appropriate PyTorch version for your CUDA toolkit (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).

3. **Prepare the data**:
   - Obtain the GRID corpus from [http://spandh.dcs.shef.ac.uk/gridcorpus/](http://spandh.dcs.shef.ac.uk/gridcorpus/).
   - Due to potential SSL/TLS issues on Windows, consider using WSL2, a Linux virtual machine, or manual download via browser.
   - Organize the data as follows:
     ```
     data/
     ├── lip/
     │   └── <speaker>/
     │       └── video/
     │           └── mpg_6000/
     │               └── <sentence>/
     │                   ├── 0000.jpg
     │                   ├── 0001.jpg
     │                   └── ...
     ├── GRID_align_txt/
     │   └── <speaker>/
     │       └── align/
     │           └── <sentence>.align
     └── lists/
         ├── train.txt
         └── val.txt
     ```
   - For quick testing, the project includes example data from two GRID corpus speakers (s2 and s23) in the `example_data/` directory. You can use the `prepare_example_data.py` script to set up this example data after downloading the example files with `download_grid_example.py`.

## Usage Instructions
### 1. Prepare the Data
   Follow the instructions above to set up the GRID corpus data in the `data/` directory.
   The expected structure is:
   - `data/lip/<speaker>/video/mpg_6000/<sentence>/` containing lip ROI images (named as 0000.jpg, 0001.jpg, ...)
   - `data/GRID_align_txt/<speaker>/align/<sentence>.align` containing alignment files
   - `data/lists/train.txt` and `data/lists/val.txt` containing the list of video paths (relative to `data/lip/`)

### 2. Run the Gradio Demo
   ```bash
   python app/app.py
   ```
   The application will start and provide a local URL (typically http://127.0.0.1:7860). Open this URL in a web browser to use the interface:
   - Upload a video file (formats supported by OpenCV, e.g., .mpg, .mp4, .avi).
   - Click the "Predict Sentence" button.
   - The system will extract lip ROIs, run inference, and display the predicted sentence.

### 3. Train the Model
   To train the model on the unseen speakers split:
   ```bash
   cd temp_repos/LipNet-PyTorch
   python main.py
   ```
   To train on the overlapped speakers split, modify `options.py`:
   ```python
   data_type = 'overlap'
   ```
   Training progress can be monitored with TensorBoard:
   ```bash
   tensorboard --logdir logs
   ```

### 4. Evaluate the Model
   After training, the model will automatically evaluate on the validation set periodically.
   For final evaluation, you can run the validation script:
   ```bash
   cd /path/to/project
   python src/training/validate.py
   ```
   Ensure that `options.py` points to the correct data paths and weights.

### 5. Use the Example Data (for quick testing)
   The project includes scripts to set up and test with example data from the GRID corpus:
   ```bash
   # Download example files (if not already done)
   python download_grid_example.py
   # Prepare example data structure
   python prepare_example_data.py
   # Test the model with the example data
   python test_example_data.py
   ```

## Example Workflow
1. **Data Preparation**: Download and organize the GRID corpus lip ROIs and alignment files.
2. **Model Initialization**: Load the LipNet model (optionally with pretrained weights).
3. **Video Input**: Upload a video file via the Gradio interface or provide a video path to the scripts.
4. **Face Detection**: Detect faces in each frame using OpenCV Haar cascades.
5. **Lip Extraction**: Extract the lip ROI from each face using a heuristic (central horizontal strip of the lower face).
6. **Preprocessing**: Resize lip ROIs to 64x128 pixels, normalize pixel values to [0, 1], and pad/truncate the sequence to 75 frames.
7. **Model Inference**: Pass the preprocessed sequence through the LipNet model to obtain character probabilities over time.
8. **CTC Decoding**: Apply greedy CTC decoding (argmax over characters, collapse repeated characters, remove blanks) to obtain the predicted sentence.
9. **Output**: Display the predicted sentence to the user.

## Limitations
- **Lip Extraction Accuracy**: The current heuristic-based lip extraction (fixed proportions of the face bounding box) is not accurate enough for meaningful predictions. This is the main bottleneck for the system's performance.
- **Dataset Access**: SSL/TLS handshake failures prevent downloading the full GRID corpus on Windows. Workarounds include using WSL2, a Linux VM, or manual download via browser.
- **GPU Utilization**: CUDA-enabled PyTorch fails to load due to DLL issues on the current Windows environment; falling back to CPU-only PyTorch. Training on the full dataset would be prohibitively slow on CPU.
- **Vocabulary**: The model predicts characters from the set: A-Z, space (and blank for CTC). Note that the GRID corpus spells out numbers (e.g., "ZERO", "ONE"), so the vocabulary is letters only.
- **Environmental Variability**: The GRID corpus is highly controlled (consistent lighting, frontal poses). Performance may degrade significantly on unconstrained videos with variations in lighting, pose, and speaking style.

## Future Improvements
- **Lip Extraction**: Replace the heuristic with MediaPipe-based facial landmark extraction to obtain precise lip contours. Apply smoothing and alignment for consistent lip ROIs over time.
- **Dataset**: Obtain the full GRID corpus via alternative download methods (WSL2, Linux VM, manual download). Consider using preprocessed lip ROIs from the VIPL BaiduYun links if accessible.
- **Training**: Enable GPU training by resolving CUDA PyTorch installation issues. Train on the full dataset to achieve reported performance.
- **Model Enhancements**: Experiment with attention mechanisms, transformer layers, or other sequence models (e.g., bidirectional LSTMs).
- **Demo Enhancements**: 
  - Add visualization of extracted lip ROIs as a video in the Gradio interface.
  - Show confidence scores and alternative predictions.
  - Allow real-time video processing (webcam input).
  - Allow upload of alignment files for ground truth comparison (for known examples from the GRID corpus).

## Presentation
A professional PowerPoint presentation is available at:
`presentation/Silent_Video_Lip_Reading_Presentation.pptx`
This presentation covers the project objective, problem statement, motivation, system architecture, workflow, technologies used, model pipeline, face detection, lip extraction, model inference, CTC decoding, Gradio interface, testing, limitations, and future improvements.

## References
- LipNet Paper:
  Assael, Y. M., Shillingford, B., Whiteson, S., & de Freitas, N. (2016).
  LipNet: End-to-End Sentence-level Lipreading.
  arXiv preprint arXiv:1611.01599.
- GRID Corpus:
  http://spandh.dcs.shef.ac.uk/gridcorpus/
- VIPL LipNet-PyTorch Implementation:
  https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch
- Other Resources:
  OpenCV: https://opencv.org/
  Gradio: https://gradio.app/
  MediaPipe: https://google.github.io/mediapool/