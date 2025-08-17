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

### `Others/` - Additional Classification Methods
Supplementary implementations exploring alternative approaches for comparison.

**Contains:**
- **AllFeatures_SVM.m**: Comprehensive SVM using 14 standard EMG features (56-81% accuracy)
- **CNN.m**: Convolutional Neural Network implementation (13-29% accuracy, limited by dataset size)
- **Others_SVM.m**: Minimal feature set SVM using only 6 basic features (~81% accuracy)
- **Results files**: CSV files with accuracy and timing comparisons

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
| **CNN** | Raw EMG signals | 13-29% | End-to-end learning |
| **Minimal-SVM** | 6 basic features | ~81% | Computational efficiency |

## Dependencies

- MATLAB Signal Processing Toolbox
- MATLAB Statistics and Machine Learning Toolbox
- MATLAB Wavelet Toolbox
- MATLAB Deep Learning Toolbox (for CNN only)

## Getting Started

1. Start with `MSC_SVM/` folder for the main research implementation
2. Execute scripts in numerical order within MSC_SVM
3. Explore `Others/` folder for alternative approaches and comparisons
4. Refer to individual folder README files for detailed implementation notes

## Research Contribution

This work demonstrates that traditional feature engineering approaches (MSC and standard EMG features) with SVM outperform deep learning methods for EMG gesture classification when working with limited training data, while providing better computational efficiency.