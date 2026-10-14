import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

prs = pptx.Presentation()

# Define helper functions
def add_title_slide(title, subtitle=None):
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    if subtitle:
        slide.placeholders[1].text = subtitle
    return slide

def add_bullet_slide(title, bullets):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.text = bullet
        p.level = 0
    return slide

# Slide 1: Title
add_title_slide('Silent Video Lip-Reading Using Deep Learning', 'A Graduation Project')

# Slide 2: Table of Contents
add_bullet_slide('Table of Contents', [
    '1. Project Objective',
    '2. Problem Statement',
    '3. Motivation',
    '4. System Architecture',
    '5. Workflow',
    '6. Technologies Used',
    '7. Model Pipeline (LipNet)',
    '8. Face Detection',
    '9. Lip Extraction',
    '10. Model Inference',
    '11. CTC Decoding',
    '12. Gradio Interface',
    '13. Testing',
    '14. Limitations',
    '15. Future Improvements',
    '16. Conclusion',
    '17. References'
])

# Slide 3: Project Objective
add_bullet_slide('Project Objective', [
    'To build an end-to-end silent video lip-reading system using deep learning.',
    'The system takes silent video as input and outputs the predicted sentence.',
    'Based on the LipNet architecture and trained on the GRID corpus.'
])

# Slide 4: Problem Statement
add_bullet_slide('Problem Statement', [
    'Lip-reading is challenging due to variations in lighting, pose, and speaking style.',
    'Existing systems often require audio or are limited to highly controlled environments.',
    'Goal: Create a speaker-independent, audio-free lip-reading system that works in real-world conditions.'
])

# Slide 5: Motivation
add_bullet_slide('Motivation', [
    'Applications in silent environments (e.g., noisy factories, submarines).',
    'Assistive technology for hearing-impaired individuals.',
    'Security and surveillance (e.g., reading lips in CCTV footage).',
    'Human-computer interaction in audio-restricted settings.'
])

# Slide 6: System Architecture
add_bullet_slide('System Architecture', [
    'Input Video → Frame Extraction',
    '→ Face Detection (OpenCV Haar Cascades)',
    '→ Lip ROI Extraction (Heuristic/MediaPipe)',
    '→ Preprocessing (Resize 64x128, Normalize [0,1])',
    '→ LipNet Model (Spatiotemporal CNN + Bi-GRU + CTC)',
    '→ CTC Decoding (Greedy)',
    '→ Text Output (Predicted Sentence)'
])

# Slide 7: Workflow
add_bullet_slide('Workflow', [
    '1. Video file is uploaded or provided.',
    '2. Frames are extracted from the video.',
    '3. Face detection is performed on each frame using OpenCV Haar cascades.',
    '4. Lip region of interest (ROI) is extracted from the face bounding box.',
    '5. Lip ROIs are resized to 64x128 pixels and normalized to [0,1].',
    '6. The sequence of lip ROIs is padded or truncated to a fixed length of 75 frames.',
    '7. The preprocessed sequence is fed into the LipNet model.',
    '8. LipNet outputs a sequence of character probabilities over time.',
    '9. Greedy CTC decoding is applied to convert probabilities to text.',
    '10. The predicted sentence is displayed to the user.'
])

# Slide 8: Technologies Used
add_bullet_slide('Technologies Used', [
    'Language: Python 3.12',
    'Deep Learning: PyTorch (CPU version due to DLL issues)',
    'Computer Vision: OpenCV (for video processing and face detection)',
    'Future Vision: MediaPipe (for accurate lip landmark extraction)',
    'Web Interface: Gradio (for interactive demo)',
    'Other: NumPy, Pandas, Matplotlib, Tqdm, Jiwer, Scikit-learn, Editdistance, TensorBoardX'
])

# Slide 9: Model Pipeline (LipNet)
add_bullet_slide('Model Pipeline (LipNet)', [
    'Input: Sequence of lip ROIs (shape: 1 x 3 x 75 x 64 x 128)',
    'Layers:',
    '  - Conv3d(3, 32, kernel=(3,5,5), stride=(1,2,2), padding=(1,2,2))',
    '  - ReLU + Dropout3d + MaxPool3d((1,2,2))',
    '  - Conv3d(32, 64, kernel=(3,5,5), stride=(1,1,1), padding=(1,2,2))',
    '  - ReLU + Dropout3d + MaxPool3d((1,2,2))',
    '  - Conv3d(64, 96, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1))',
    '  - ReLU + Dropout3d + MaxPool3d((1,2,2))',
    '  - GRU(96*4*8, 256, bidirectional=True)',
    '  - Dropout',
    '  - GRU(512, 256, bidirectional=True)',
    '  - Dropout',
    '  - Linear(512, 28)  # 27 characters (A-Z + space) + 1 blank',
    'Output: Logits over time steps (shape: 75 x 1 x 28)'
])

# Slide 10: Face Detection
add_bullet_slide('Face Detection', [
    'Method: OpenCV Haar Cascade Classifier',
    'Model: haarcascade_frontalface_default.xml',
    'Process:',
    '  - Convert frame to grayscale.',
    '  - Detect faces in the frame.',
    '  - Select the largest face (by area) if multiple faces are present.',
    'Limitations:',
    '  - May fail in poor lighting, non-frontal faces, or occlusions.',
    '  - Not as accurate as deep learning-based or landmark-based methods.'
])

# Slide 11: Lip Extraction
add_bullet_slide('Lip Extraction', [
    'Current Implementation:',
    '  - Heuristic-based extraction from face bounding box.',
    '  - Mouth x-range: [face_x + 0.25*face_width, face_x + 0.75*face_width]',
    '  - Mouth y-range: [face_y + 0.5*face_height, face_y + 0.75*face_height]',
    '  - This captures the central horizontal strip of the lower half of the face.',
    'Limitations:',
    '  - Not accurate for all poses and lighting conditions.',
    '  - Does not account for individual lip shape or movements.',
    'Future Improvements:',
    '  - Use facial landmarks (MediaPipe) to extract precise lip contours.',
    '  - Apply smoothing and alignment for consistent lip ROIs over time.'
])

# Slide 12: Model Inference
add_bullet_slide('Model Inference', [
    'Input: Preprocessed lip ROI sequence (shape: 1 x 3 x 75 x 64 x 128)',
    'Process:',
    '  - The lip sequence is passed through the LipNet model.',
    '  - The model outputs a tensor of shape (75, 1, 28) representing logits over time and characters.',
    '  - Softmax is applied to convert logits to probabilities (optional for CTC loss).',
    'Output: Sequence of character probabilities over time.'
])

# Slide 13: CTC Decoding
add_bullet_slide('CTC Decoding', [
    'CTC (Connectionist Temporal Classification) allows alignment-free sequence modeling.',
    'Decoding Steps:',
    '  1. Apply argmax over the character dimension to get predicted indices for each time step.',
    '  2. Collapse repeated indices (e.g., [A, A, A, B, B] -> [A, B]).',
    '  3. Remove blank indices (index 0 corresponds to the blank token).',
    '  4. Map remaining indices to characters using the vocabulary (A-Z, space).',
    'Example:',
    '  Indices: [2, 2, 2, 3, 3, 0, 4, 5, 5] -> [A, A, A, B, B, _, D, E, E]',
    '  After collapse: [A, B, _, D, E]',
    '  After removing blanks: [A, B, D, E] -> \"ABDE\"'
])

# Slide 14: Gradio Interface
add_bullet_slide('Gradio Interface', [
    'Features:',
    '  - Upload a video file (formats supported by OpenCV).',
    '  - The system extracts lip ROIs and runs inference.',
    '  - Displays the predicted sentence.',
    'Interface Components:',
    '  - Video input component.',
    '  - Predicted sentence output textbox.',
    '  - Submit button to trigger processing.',
    'Future Enhancements:',
    '  - Display extracted lip ROIs as a video.',
    '  - Show confidence scores or alternative predictions.',
    '  - Allow upload of alignment files for ground truth comparison (for known examples).'
])

# Slide 15: Testing
add_bullet_slide('Testing', [
    'Unit Tests:',
    '  - Model loading and forward pass.',
    '  - Lip extraction function.',
    '  - Video to lip sequence conversion.',
    '  - Dataset loading and preprocessing.',
    'Integration Tests:',
    '  - One training step (forward, loss, backward, optimizer step).',
    '  - One validation step.',
    '  - End-to-end prediction on example data.',
    'Example Data Testing:',
    '  - Used two example videos from the GRID corpus (s2/swwp2s and s23/priazn).',
    '  - Verified lip ROI extraction (75 frames per video).',
    '  - Verified alignment file parsing.',
    '  - Model inference runs without errors.'
])

# Slide 16: Limitations
add_bullet_slide('Limitations', [
    'Lip Extraction Accuracy:',
    '  - Current heuristic-based lip extraction is not accurate enough for meaningful predictions.',
    '  - This is the main bottleneck for the system\'s performance.',
    'Dataset Access:',
    '  - SSL/TLS handshake failures prevent downloading the full GRID corpus on Windows.',
    '  - Workaround: Use WSL2, Linux VM, or manual download via browser.',
    'GPU Utilization:',
    '  - CUDA-enabled PyTorch fails to load due to DLL issues; falling back to CPU.',
    '  - Training on the full dataset would be prohibitively slow on CPU.',
    'Vocabulary:',
    '  - The model predicts characters from the set: A-Z, space (and blank for CTC).',
    '  - Note: The GRID corpus spells out numbers (e.g., \"ZERO\", \"ONE\"), so the vocabulary is letters only.',
    'Environmental Variability:',
    '  - The GRID corpus is highly controlled (consistent lighting, frontal poses).',
    '  - Performance may degrade significantly on unconstrained videos.'
])

# Slide 17: Future Improvements
add_bullet_slide('Future Improvements', [
    'Lip Extraction:',
    '  - Replace heuristic with MediaPipe-based facial landmark extraction.',
    '  - Apply smoothing and alignment for consistent lip ROIs over time.',
    'Dataset:',
    '  - Obtain the full GRID corpus via alternative download methods.',
    '  - Consider using preprocessed lip ROIs from the VIPL BaiduYun links (if accessible).',
    'Training:',
    '  - Enable GPU training by resolving CUDA PyTorch installation issues.',
    '  - Train on the full dataset to achieve reported performance.',
    'Model Enhancements:',
    '  - Experiment with attention mechanisms or transformer layers.',
    '  - Investigate bidirectional LSTMs or other sequence models.',
    'Demo Enhancements:',
    '  - Add visualization of extracted lip ROIs in the Gradio interface.',
    '  - Add confidence scores and alternative predictions.',
    '  - Allow real-time video processing (webcam input).'
])

# Slide 18: Conclusion
add_bullet_slide('Conclusion', [
    'We have successfully assembled a functional lip-reading pipeline based on the LipNet architecture.',
    'The system demonstrates the complete workflow from video to text prediction.',
    'Key achievements:',
    '  - Environment setup with PyTorch and essential libraries.',
    '  - Integration of the VIPL LipNet-PyTorch implementation.',
    '  - Preparation of example data from GRID corpus examples.',
    '  - Working pipeline from video to text prediction (lip extraction -> model inference -> CTC decoding).',
    '  - Gradio demo application for interactive lip-reading prediction.',
    '  - Comprehensive documentation and reports.',
    'Next Steps:',
    '  - Obtain the full GRID corpus.',
    '  - Improve lip extraction accuracy with facial landmarks.',
    '  - Enable GPU training and conduct proper training on the full dataset.',
    '  - Evaluate the model using WER, CER, and sentence-level accuracy metrics.',
    '  - Enhance the demo with visualization and confidence scores.'
])

# Slide 19: References
add_bullet_slide('References', [
    'LipNet Paper:',
    '  Assael, Y. M., Shillingford, B., Whiteson, S., & de Freitas, N. (2016).',
    '  LipNet: End-to-End Sentence-level Lipreading.',
    '  arXiv preprint arXiv:1611.01599.',
    '',
    'GRID Corpus:',
    '  http://spandh.dcs.shef.ac.uk/gridcorpus/',
    '',
    'VIPL LipNet-PyTorch Implementation:',
    '  https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch',
    '',
    'Other Resources:',
    '  OpenCV: https://opencv.org/',
    '  Gradio: https://gradio.app/',
    '  MediaPipe: https://google.github.io/mediapool/'
])

# Slide 20: Thank You
add_title_slide('Thank You!', 'Questions?')

# Save presentation
prs.save('presentation/Silent_Video_Lip_Reading_Presentation.pptx')
print('Presentation saved to presentation/Silent_Video_Lip_Reading_Presentation.pptx')