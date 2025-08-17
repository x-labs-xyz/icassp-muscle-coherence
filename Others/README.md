# Others: Additional Classification Methods

This folder contains additional classification approaches for EMG-based hand gesture recognition, extending beyond the MSC analysis in the main MSC_SVM folder.

## Overview

This collection includes implementations of:
1. **Standard EMG feature-based SVM** - Traditional time/frequency domain features
2. **Convolutional Neural Network (CNN)** - Deep learning approach
3. **Reduced feature set SVM** - Minimal feature subset analysis

All approaches use the same NinaPro DB2 dataset and train/test split as the MSC analysis for direct comparison.

## Files

### MATLAB Scripts (.m files)

#### `AllFeatures_SVM.m` - Comprehensive EMG Feature SVM
- **Purpose**: Trains SVM using 14 standard EMG features extracted from literature
- **Features Extracted**:
  - **Time-domain**: IEMG, Variance, Waveform Length, Slope Sign Change, Zero Crossing, Willison Amplitude, MAV, RMS, MAVS
  - **Frequency-domain**: Mean Frequency (MNF), Power Spectrum Ratio (PSR)  
  - **Wavelet**: Marginal Discrete Wavelet Transform (mDWT) using 4th level decomposition
  - **Statistical**: Histogram of EMG (HEMG) with 10 bins, Autoregressive Coefficients (ARC) 4th order
- **Performance**: 56-81% accuracy range across 40 subjects
- **Output**: `SVMAll_Accuracy_and_Times.csv`

#### `CNN.m` - Convolutional Neural Network
- **Purpose**: Implements CNN for raw EMG signal classification
- **Architecture**:
  - Input layer: 600×1×12 (window_length × channels)
  - Conv2D layer 1: 5×1 filters, 16 feature maps, ReLU activation
  - Conv2D layer 2: 5×1 filters, 32 feature maps, ReLU activation  
  - Fully connected: 49 outputs (movements)
  - Batch normalization after each conv layer
- **Training**: Adam optimizer, 10 epochs, batch size 64
- **Window**: 300ms with 50% overlap
- **Performance**: 13-29% accuracy (limited by small dataset and simple architecture)
- **Output**: `CNN_Accuracy_and_Times.csv`

#### `Others_SVM.m` - Reduced Feature Set SVM
- **Purpose**: Tests SVM performance with minimal feature subset
- **Features**: Only 6 basic time-domain features (IEMG, VAR, WL, SSC, ZC, WAMP)
- **Performance**: ~81% accuracy for single subject (Subject 1)
- **Use Case**: Demonstrates feature selection impact and computational efficiency

### Results Files (.csv files)

#### `CNN_Accuracy_and_Times.csv`
- **Content**: CNN classification results for 40 subjects
- **Columns**: Subject, Accuracy (%), Training Time (s), Testing Time (s)
- **Key Findings**: 
  - Poor performance (13-29% accuracy) due to limited training data
  - High computational cost (40-128s training time)
  - Subjects 1-5 show 0% accuracy (likely training issues)

#### `SVM_Accuracy_Results.csv`
- **Content**: SVM accuracy results using standard EMG features
- **Columns**: Subject, Accuracy (%)
- **Performance Range**: 56.3-81.5% accuracy across subjects
- **Average**: ~69% accuracy

#### `SVMAll_Accuracy_and_Times.csv`
- **Content**: Comprehensive SVM results with timing information
- **Columns**: Subject, Accuracy (%), Training Time (s), Testing Time (s)
- **Key Findings**:
  - Training time: 7-50 seconds per subject
  - Testing time: 2-4 seconds per subject
  - More efficient than CNN while achieving better performance

## Key Parameters

### Common Settings
- **Dataset**: NinaPro DB2 (40 subjects, 49 movements, 6 repetitions)
- **Sampling Rate**: 2000 Hz
- **Window Size**: 300ms (600 samples)
- **Window Overlap**: 50%
- **Train/Test Split**: Repetitions [1,3,4,6] for training, [2,5] for testing

### SVM Configuration
- **Kernel**: Polynomial (quadratic, order=2)
- **Regularization**: C = 10
- **Scaling**: Auto kernel scale
- **Standardization**: Enabled
- **Multi-class**: One-vs-all coding

### CNN Configuration  
- **Optimizer**: Adam
- **Epochs**: 10
- **Batch Size**: 64
- **Input Shape**: [600, 1, 12] (time × width × channels)

## Performance Comparison

| Method | Features | Accuracy Range | Avg Training Time | Key Advantages |
|--------|----------|----------------|-------------------|----------------|
| **MSC-SVM** | MSC mean values | ~70% | 7-25s | Captures inter-muscular coordination |
| **AllFeatures-SVM** | 14 standard features | 56-81% | 7-50s | Comprehensive feature representation |
| **CNN** | Raw EMG | 13-29% | 40-128s | End-to-end learning (limited by data) |
| **Others-SVM** | 6 basic features | ~81% (1 subject) | Fast | Minimal computational requirements |

## Dependencies

- MATLAB Signal Processing Toolbox
- MATLAB Statistics and Machine Learning Toolbox  
- MATLAB Deep Learning Toolbox (for CNN)
- MATLAB Wavelet Toolbox

## Usage Notes

1. **Data Path**: Scripts expect EMG data files in the same directory
2. **Subject Processing**: Most scripts process all 40 subjects sequentially
3. **Memory Requirements**: CNN requires more memory due to 4D data arrays
4. **Execution Time**: CNN training significantly longer than SVM approaches

## Research Insights

- **Feature Engineering vs Deep Learning**: Traditional features with SVM outperform CNN on this dataset size
- **Computational Efficiency**: SVM methods provide better accuracy/time tradeoff
- **Feature Redundancy**: Minimal feature set (Others_SVM) achieves comparable performance
- **Inter-subject Variability**: Large performance differences across subjects (56-81% range)

This analysis demonstrates that for EMG gesture classification with limited data, carefully engineered features with SVM can outperform deep learning approaches while being more computationally efficient.