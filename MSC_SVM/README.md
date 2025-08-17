# MSC_SVM: Magnitude Squared Coherence for EMG Classification

This repository contains MATLAB code for comparing Magnitude Squared Coherence (MSC) features with standard EMG features for hand gesture classification using Support Vector Machines (SVM).

## Overview

The code processes electromyography (EMG) data from the NinaPro DB2 dataset to classify hand gestures using two different feature extraction approaches:

1. **MSC-based features**: Uses magnitude squared coherence between EMG sensor pairs
2. **Standard EMG features**: Uses traditional time-domain, frequency-domain, and wavelet features

## Dataset

- **Source**: NinaPro DB2 dataset
- **Subjects**: 40 subjects
- **Movements**: 49 hand gestures
- **Repetitions**: 6 per movement
- **Sensors**: 12 EMG channels
- **Sampling Rate**: 2000 Hz

## Files (Execute in Order)

### 1. `1_Export_MSC.m` - MSC Feature Extraction
- Processes filtered and normalized EMG signals from the `Data` folder
- Calculates Power Spectral Density (PSD), Cross Power Spectral Density (CPSD), and Magnitude Squared Coherence (MSC)
- Uses Welch method with:
  - Window size: 600 samples (300ms)
  - Overlap: 50%
- Creates subject folders containing MSC matrices for each movement
- Output: 6×1 cell arrays (one per repetition) for each subject/movement combination

### 2. `2_Run_SVM.m` - MSC-based SVM Classification
- Imports MSC matrices and creates feature matrix using mean MSC values between sensor pairs
- Trains quadratic polynomial SVM with:
  - Regularization parameter (C): 10
  - Polynomial order: 2
  - One-vs-all coding for multi-class classification
- **Train/Test Split**: 
  - Training: repetitions [1, 3, 4, 6]
  - Testing: repetitions [2, 5]
- **Performance**: ~70% average accuracy across subjects
- Outputs: accuracy, precision, recall, F1-score, AUC, and computation times

### 3. `3_Plots_SVM.m` - Results Visualization
- Generates plots and tables from SVM classification results
- Creates various visualizations for performance analysis

### 4. `4_Run_SVM_StandardFeatures.m` - Standard EMG Feature Classification
- Extracts 14 traditional EMG features:
  - **Time-domain**: IEMG, Variance, Waveform Length, Slope Sign Change, Zero Crossing, Willison Amplitude, MAV, RMS, MAVS
  - **Frequency-domain**: Mean Frequency (MNF), Power Spectrum Ratio (PSR)
  - **Wavelet**: Marginal Discrete Wavelet Transform (mDWT)
  - **Statistical**: Histogram of EMG (HEMG), Autoregressive Coefficients (ARC)
- Uses same SVM configuration and train/test split as MSC approach
- **Performance**: 56-81% accuracy range across subjects
- Window-based feature extraction with 300ms windows and 50% overlap

## Key Parameters

- **Window Length**: 300ms (600 samples at 2000 Hz)
- **Window Overlap**: 50%
- **SVM Kernel**: Polynomial (quadratic)
- **Regularization Parameter**: C = 10
- **Feature Normalization**: Z-score normalization using training data statistics

## Results

The comparison between MSC-based and standard EMG features provides insights into:
- Effectiveness of coherence-based features for gesture classification
- Computational efficiency differences between approaches
- Subject-specific performance variations
- Feature interpretability and physiological relevance

## Data Structure

```
MSC_SVM/
├── Data/                           # EMG signals and labels
├── Subject1/                       # MSC matrices for Subject 1
│   ├── MSC_S1_mov1.mat
│   ├── MSC_S1_mov2.mat
│   └── ...
├── Results_S1.mat                  # Classification results for Subject 1
├── MSC_Accuracy_and_Times.csv     # MSC approach performance
└── SVMAll_Accuracy_and_Times.csv  # Standard features performance
```

## Dependencies

- MATLAB Signal Processing Toolbox
- MATLAB Statistics and Machine Learning Toolbox
- MATLAB Wavelet Toolbox

## Authors

Last modified: 17/03/2025 by Costanza Armanini

## Citation

If you use this code, please cite the associated research paper on MSC-based EMG classification.