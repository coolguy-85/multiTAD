# multiTAD:an Attention-Based Deep Learning Model for Identifying TAD Boundaries through Multi-Scale Feature Integration

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

## Data Requirements

The models in this repository are designed to process genomic data specifically formatted in TSV (Tab-Separated Values) files. Below are the detailed requirements for the data structure and format:

### General Format

- **File Type**: All input files should be in TSV format.
- **Naming Convention**: Files should be named using the pattern `<chipname>.<pos/neg>.tsv`, where `<chipname>` refers to the type of genomic data (e.g., CTCF, RAD21), and `<pos/neg>` indicates whether the data is associated with positive or negative examples of TAD boundaries.

### File Content

- No header row should be included; all rows must contain data.
- Missing values should be handled prior to processing, either replaced by a defined numeric value (e.g., 0.0) or by using imputation techniques.

### Directory Structure

- **Data Directory**: Data should be organized within a `data/` directory at the root level of the project.
- **Subdirectories**: Each cell line or experimental condition should have its own subdirectory within the `data/` directory. For example, data for the GM12878 cell line should be in `data/GM12878/`.

### Example Structure

Here is an example of how to structure the data directory for optimal compatibility with the scripts:

```objectivec
objectivec复制代码
data/
│
├── GM12878/
│   ├── CTCF.pos.tsv
│   ├── CTCF.neg.tsv
│   ├── RAD21.pos.tsv
│   └── RAD21.neg.tsv
│
├── IMR90/
│   ├── CTCF.pos.tsv
│   ├── CTCF.neg.tsv
│   ├── RAD21.pos.tsv
│   └── RAD21.neg.tsv
│
└── K562/
    ├── CTCF.pos.tsv
    ├── CTCF.neg.tsv
    ├── RAD21.pos.tsv
    └── RAD21.neg.tsv

```

### Preprocessing

Before feeding the data into the models, ensure that:

- All files are correctly formatted and placed in the appropriate subdirectories.
- Any preprocessing steps required to normalize or scale the data are completed.
- Data splits (training, validation, test) are defined as needed, either externally or within the script configurations.

By adhering to these data requirements, users can ensure that the models will perform as expected and can effectively learn to identify TAD boundaries from genomic features.
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
