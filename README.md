# Drug Discovery Classification with Ersilia Model Hub

This repository contains a machine learning project for drug discovery that leverages the Ersilia Model Hub for molecular featurization. The project demonstrates how to download a dataset from the Therapeutics Data Commons (TDC), featurize molecules using Ersilia models, and build a classification model.

## Project Overview

This project applies quality assessment principles similar to those used in clinical laboratory EQA (External Quality Assessment) to drug discovery. Just as in laboratory EQA where samples are provided to clients for testing and results are compared against reference values, in this project we:

1. Download a dataset with known molecular properties (our "samples")
2. Process these molecules through featurization (our "testing procedure")
3. Build a model to predict properties and evaluate against known values (our "quality assessment")

## Project Structure

The repository is organized as follows:
- `data/`: Contains downloaded datasets and featurized data
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `scripts/`: Python scripts for data processing, featurization, and model building
- `models/`: Saved model checkpoints

## Installation

### Prerequisites
- Python 3.8+
- Conda environment manager

### Setup
1. Clone this repository:
```bash
git clone https://github.com/your-username/ersilia-os-outreachy-contributions.git
cd ersilia-os-outreachy-contributions
```

2. Create and activate a conda environment:
```bash
conda create -n ersilia python=3.10
conda activate ersilia
```

3. Install required packages:
```bash
pip install ersilia tdc xgboost scikit-learn matplotlib seaborn pandas numpy
```

## Usage

### 1. Download a Dataset

To download a classification dataset from the Therapeutics Data Commons (TDC):

```bash
python scripts/download_dataset.py
```

This script will:
- List available datasets from TDC
- Download the Blood-Brain Barrier Penetration (BBBP) dataset by default
- Save train, validation, test, and full datasets to the `data/` folder

To download a different dataset, modify the `dataset_name` and `dataset_group` variables in the script.

### 2. Featurize Molecules

To featurize molecules using an Ersilia model:

```bash
python scripts/featurize_molecules.py --input data/BBBP_full.csv --model eos4e40
```

Replace `eos4e40` with the ID of the Ersilia model you want to use for featurization. You can browse available models on the [Ersilia Model Hub](https://ersilia.io/model-hub) with the "Representation" label.

### 3. Build and Evaluate a Model

To train and evaluate a machine learning model using the featurized data:

```bash
python scripts/build_ml_model.py --input data/featurized_BBBP_full.csv --model-type xgboost
```

This script will:
- Train an XGBoost classifier (or Random Forest if specified)
- Evaluate the model performance (accuracy, precision, recall, F1 score)
- Generate visualizations (confusion matrix, ROC curve)
- Save the trained model to the `models/` folder

## Dataset Details

The default dataset used in this project is the Blood-Brain Barrier Penetration (BBBP) dataset from TDC. This is a binary classification task that predicts whether a compound can penetrate the blood-brain barrier.

- **Task**: Binary classification
- **Endpoint**: Blood-brain barrier penetration
- **Positive class**: Compounds that can penetrate the blood-brain barrier
- **Negative class**: Compounds that cannot penetrate the blood-brain barrier
- **Number of compounds**: ~2,000
- **Data source**: Martins et al. 2012

## Featurization

For molecular featurization, we use the Ersilia Model Hub, which provides access to pre-trained AI/ML models. The featurization process converts molecular structures (SMILES strings) into numerical features that can be used for machine learning.

## Model Evaluation

The model is evaluated using standard classification metrics:
- Accuracy: Overall correctness of predictions
- Precision: Proportion of positive identifications that were actually correct
- Recall: Proportion of actual positives that were identified correctly
- F1 Score: Harmonic mean of precision and recall

Visualizations include:
- Confusion Matrix: Shows true positives, false positives, true negatives, and false negatives
- ROC Curve: Shows the trade-off between true positive rate and false positive rate

## Acknowledgements

- [Ersilia Open Source Initiative](https://ersilia.io) for providing the Model Hub
- [Therapeutics Data Commons](https://tdcommons.ai) for providing the datasets
- Outreachy program for the opportunity to contribute to open source