# Implementation Decision

## Candidate Repositories Evaluated

### 1. VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch
- **Stars/Forks**: Not checked but appears to be the original implementation
- **Last Updated**: Recent (based on file timestamps)
- **Pretrained Weights**: ✅ Available for both unseen and overlapped speakers
- **Architecture**: Clear LipNet implementation matching the paper
- **Dataset Handling**: Expects pre-extracted lip ROIs (images) and alignment files
- **Demo**: Included demo.py showing end-to-end pipeline
- **Dependencies**: PyTorch, OpenCV, face_alignment (for demo only)
- **License**: MIT
- **Pros**: 
  - Pretrained weights available
  - Complete training and testing pipeline
  - Clear documentation with results
  - Demo included
- **Cons**:
  - Requires pre-processed data (lip ROIs and alignments) rather than raw video
  - Demo requires face_alignment which has additional dependencies

### 2. sailordiary/LipNet-PyTorch
- **Implementation**: Unofficial PyTorch implementation based on official Torch implementation
- **Pretrained Weights**: Not mentioned in README
- **Architecture**: LipNet implementation with configurable parameters
- **Dataset Handling**: Expects raw images organized in specific folder structure
- **Demo**: Not mentioned
- **Dependencies**: More extensive (ctcdecode, tensorboardX, torchsummary, progressbar2, scikit-image, etc.)
- **License**: Not specified
- **Pros**:
  - Might work with raw images rather than requiring pre-processing
  - More configurable
- **Cons**:
  - No pretrained weights mentioned
  - More complex dependency tree
  - Less documentation on results

### 3. osalinasv/lipnet
- **Status**: Could not clone/repository may not be accessible
- **Evaluation**: Not possible

## Decision Criteria

1. **Availability of Pretrained Weights**: Critical for getting quick results without training from scratch
2. **Implementation Completeness**: Having training, testing, and demo code
3. **Dependency Complexity**: Simpler dependencies are preferred for reproducibility
4. **Documentation Quality**: Clear instructions help with setup and understanding
5. **License Compatibility**: MIT or similar permissive licenses preferred

## Selected Implementation

**VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch** is selected because:

1. ✅ **Pretrained Weights Available**: Both unseen and overlapped speaker models provided
2. ✅ **Complete Pipeline**: Training, testing, and demo code all included
3. ✅ **Clear Documentation**: Detailed README with setup instructions and results
4. ✅ **Reasonable Dependencies**: PyTorch, OpenCV, plus face_alignment only for demo
5. ✅ **Permissive License**: MIT license
6. ✅ **Verified Results**: Reports 13.3%/4.6% WER which we can attempt to replicate

## Trade-offs and Mitigations

### Trade-off: Requires Pre-processed Data
The VIPL implementation expects:
- Lip ROI images (already cropped to mouth region)
- Text alignment files (.align)

This means we cannot use raw GRID video files directly.

### Mitigation Strategies:
1. **Use their provided data**: They offer BaiduYun links to pre-processed lip images and annotations
2. **Implement our own preprocessing**: Use the scripts/extract_lip.py as reference to create lip ROIs from raw video
3. **Hybrid approach**: Start with their provided pre-processed data to validate the pipeline, then implement our own preprocessing

## Implementation Plan

1. **Phase 1**: Use the VIPL repo as-is with their provided pre-processed data (if we can access it)
2. **Phase 2**: Implement our own preprocessing pipeline to convert raw GRID video to lip ROIs
3. **Phase 3**: Validate that our preprocessing produces comparable results to their provided data
4. **Phase 4**: Optionally fine-tune or train from scratch if compute allows

## Next Steps

1. Attempt to access the GRID corpus through official channels (https://spandh.dcs.shef.ac.uk/gridcorpus/)
2. If access is granted, download and preprocess using VIPL's extract_lip.py or our own implementation
3. If access is not granted, investigate whether we can use their BaiduYun links (requires manual download)
4. Set up the VIPL repo and test with dummy data to verify the pipeline works
5. Proceed with actual data once available

## References
- Original Paper: https://arxiv.org/abs/1611.01599
- VIPL Implementation: https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch
- Sailordiary Implementation: https://github.com/sailordiary/LipNet-PyTorch