This repository contains three Python scripts for the identification of Topologically Associating Domain (TAD) boundaries using convolutional neural networks (CNNs) and channel attention mechanisms. These models are designed to handle genomic data and predict TAD boundaries with varying levels of feature input and model complexity.

## Introduction

### multiTAD

**multiTAD** is designed for multi-size identification of TAD boundaries, incorporating a complex architecture that uses three parallel submodels (M1, M2, M3). Each submodel processes different input sizes to capture TAD boundaries of varying scales. The model uses a range of genomic features including transcription-related factors, histone modifications, and chromatin accessibility to distinguish between boundary and non-boundary regions.

### singleTAD

**singleTAD** assesses the preference of each cell line for TAD boundary size by using a single-size input. It is effectively one of the submodels from multiTAD.

### multiTAD-L

**multiTAD-L** is a lightweight version of multiTAD that utilizes only three features: CTCF, RAD21, and SMC3. This model is particularly useful when only transcription-related factors data are available. Despite its reduced complexity, multiTAD-L achieves predictive performance comparable to the full multiTAD model and demonstrates robust cross-cell line capabilities.

## Requirements

The models are implemented in Python and require the following dependencies:

- Python 3.6+
- PyTorch 1.7+
- Torchvision 0.8+
- NumPy
- Pandas
- scikit-learn
- tqdm

You can install these dependencies via pip:

```bash

pip install torch torchvision numpy pandas scikit-learn tqdm

```

## Dataset

The script expects the input data to be in the form of TSV files, stored in a structured directory format. Each cell line and data type should have its own directory, with files named according to a specific convention: `<chipname>.<pos/neg>.tsv`.

## Directory Structure

Make sure to organize your data and scripts in the following structure for optimal performance:

```kotlin

project/
│
├── multiTAD.py
├── singleTAD.py
├── multiTAD-L.py
│
└── data/
    └── [data files and folders]

```

### Utility Functions

- `extract`: Loads and preprocesses data from TSV files.
- `pred_model`: Evaluates the model on a validation or test set, calculating metrics and losses.
- `eval_results`: Computes various evaluation metrics such as AUC, precision, recall, and F-measure.
- `main`: Orchestrates the training and evaluation process, handling data loading, model initialization, and looping through epochs.

## Usage

Before running the scripts, you must configure the input data paths in each script to point to the directories where your genomic data is stored. Ensure that the data is formatted as required by each model's specifications.

Here are example commands to run each model script from the command line:
```bash

# For running multiTAD
python multiTAD.py

# For running singleTAD
python singleTAD.py

# For running multiTAD-L
python multiTAD-L.py

```
