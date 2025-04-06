# Mutagenicity prediction project

This is a project created as part of contributing to the [Ersilia Open Source Initiative](https://www.ersilia.io/).

The goal is to develop a mutagenicity prediction model using the [Ersilia Model Hub](https://www.ersilia.io/model-hub) and datasets from various sources (e.g., [Therapeutics Data Commons](https://tdcommons.ai/overview)).

I have chosen the [AMES mutagenicity dataset](https://tdcommons.ai/single_pred_tasks/tox#ames-mutagenicity) from Therapeutics Data Commons (TDC) because of the large size of the dataset.
The Ames test is a widely used method to determine whether a chemical can cause mutations in DNA. This is crucial in drug development since compounds that damage DNA can lead to cancer or other severe health issues.

## Table of Contents
[1. Dataset and task description](#dataset-and-task-description)
[2. Data acquisition](#data-acquisition)
[3. Data featurization](#data-featurization)
[4. Data modeling](#data-modeling)
[5. Model evaluation](#model-evaluation)
[6. Model deployment](#model-deployment)

## Dataset and task description
These are the dataset descriptions as provided by Therapeutics Data Commons (TDC):

**Dataset description**: Mutagenicity means the ability of a drug to induce genetic alterations. Drugs that can cause damage to the DNA can result in cell death or other severe adverse effects. Nowadays, the most widely used assay for testing the mutagenicity of compounds is the Ames experiment which was invented by a professor named Ames. The Ames test is a short-term bacterial reverse mutation assay detecting a large number of compounds which can induce genetic damage and frameshift mutations. The dataset is aggregated from four papers.

**Task description**: Binary classification. Given a drug SMILES string, predict whether it is mutagenic (1) or not mutagenic (0).

## Data acquisition
To get *single prediction* data from Therapeutics Data Commons (TDC), I will use the [fetch_dataset.py](scripts/fetch_dataset.py) script.

I'm fetching only single prediction datasets since this is what we will be using to create a model for now.

List single prediction datasets available in TDC:
```bash
python scripts/fetch_dataset.py --list
```

Download a dataset:
```bash
python scripts/fetch_dataset.py --dataset_name AMES --dataset_group Tox
```

## Data Featurization
We need features to enable us create a decent prediction model. The data we downloaded looks like this so far and doesn't have much to work with:
```
Drug_ID,Drug,Y
Drug 1,O=[N+]([O-])c1c2c(c3ccc4cccc5ccc1c3c45)CCCC2,1
Drug 2,O=c1c2ccccc2c(=O)c2c1ccc1c2[nH]c2c3c(=O)c4ccccc4c(=O)c3c3[nH]c4c(ccc5c(=O)c6ccccc6c(=O)c54)c3c12,0
Drug 3,[N-]=[N+]=CC(=O)NCC(=O)NN,1
Drug 4,[N-]=[N+]=C1C=NC(=O)NC1=O,1
Drug 6,CCCCN(CC(O)C1=CC(=[N+]=[N-])C(=O)C=C1)N=O,1
```

For featurization we will use variouse Ersilia models and determine which one is the best for our task.

We will use the [featurization.py](scripts/featurization.py) script to featurize the dataset.

Here we pass in a dataset path which we got from the `fetch_dataset.py` script, a model identifier, and a feature name and the script will featurize the dataset.

Example if the model returns a single outcome:
```bash
python scripts/featurization.py --dataset_name AMES --model_id eos3b5e --feature_name mol_weight
```

Example if the model returns multiple outcomes:
```bash
python scripts/featurization.py --dataset_name AMES --model_id eos9gg2
```

This will featurize the dataset and save it to the data folder. Each subsequent run with a new model will update the featurized datasets.

## Data Modeling
The process of creating a model can be seen in [notebooks/analysis.ipynb](notebooks/analysis.ipynb).