# EMG Hand Gesture Classification - MATLAB Code

This repository contains MATLAB implementations for comparing different approaches to EMG-based hand gesture classification using the NinaPro DB2 dataset.

## Dataset Overview

- **Source**: NinaPro DB2 dataset
- **Subjects**: 40 subjects
- **Movements**: 49 hand gestures
- **Repetitions**: 6 per movement
- **Sensors**: 12 EMG channels
- **Sampling Rate**: 2000 Hz

## Folder Structure

### `MSC_SVM/` - Magnitude Squared Coherence Approach
Primary research implementation comparing MSC-based features with standard EMG features for SVM classification.

**Key Features:**
- Novel MSC feature extraction using inter-channel coherence
- Standard EMG feature comparison (14 traditional features)
- Performance: ~70% accuracy with MSC features
- Execution order: 1_Export_MSC.m → 2_Run_SVM.m → 3_Plots_SVM.m → 4_Run_SVM_StandardFeatures.m

## Common Configuration

All implementations use consistent parameters for fair comparison:
- **Window Size**: 300ms (600 samples)
- **Window Overlap**: 50%
- **Train/Test Split**: Repetitions [1,3,4,6] for training, [2,5] for testing
- **SVM Kernel**: Polynomial (quadratic, order=2)
- **Regularization**: C = 10

## Performance Summary

| Method | Features | Accuracy Range | Key Advantages |
|--------|----------|----------------|----------------|
| **MSC-SVM** | Inter-channel coherence | ~70% | Captures muscular coordination |
| **Standard EMG-SVM** | 14 traditional features | 56-81% | Comprehensive representation |

## Dependencies

- MATLAB Signal Processing Toolbox
- MATLAB Statistics and Machine Learning Toolbox
- MATLAB Wavelet Toolbox
- MATLAB Deep Learning Toolbox (for CNN only)

## Getting Started

1. Start with `MSC_SVM/` folder for the main research implementation
2. Execute scripts in numerical order within MSC_SVM
3. Refer to individual folder README files for detailed implementation notes

## Research Contribution


This work demonstrates that traditional feature engineering approaches (MSC and standard EMG features) with SVM outperform deep learning methods for EMG gesture classification when working with limited training data, while providing better computational efficiency.
