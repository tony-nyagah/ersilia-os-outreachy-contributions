"""
Script to download a classification dataset from Therapeutics Data Commons (TDC)
and save it to the data folder.
"""

import os
import pandas as pd
from tdc.single_pred import ADME
from tdc.single_pred import Tox
from tdc.utils import retrieve_dataset_names

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)


def list_available_datasets() -> None:
    """List all available classification datasets in TDC."""
    print("Available ADME datasets:")
    adme_datasets = retrieve_dataset_names("ADME")
    for dataset in adme_datasets:
        print(f"- {dataset}")

    print("\nAvailable Toxicity datasets:")
    tox_datasets = retrieve_dataset_names("Tox")
    for dataset in tox_datasets:
        print(f"- {dataset}")


def download_dataset(dataset_name: str, dataset_group: str) -> pd.DataFrame:
    """
    Download a specific dataset from TDC.

    Args:
        dataset_name (str): Name of the dataset to download
        dataset_group (str): Group of the dataset ('ADME' or 'Tox')

    Returns:
        pandas.DataFrame: The downloaded dataset
    """
    print(f"Downloading {dataset_name} dataset from {dataset_group} group...")

    if dataset_group == "ADME":
        data = ADME(name=dataset_name)
    elif dataset_group == "Tox":
        data = Tox(name=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset group: {dataset_group}")

    # Get the split
    split = data.get_split()
    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    # Save the datasets
    train_path = os.path.join(data_dir, f"{dataset_name}_train.csv")
    valid_path = os.path.join(data_dir, f"{dataset_name}_valid.csv")
    test_path = os.path.join(data_dir, f"{dataset_name}_test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Also save the full dataset
    full_df = pd.concat([train_df, valid_df, test_df])
    full_path = os.path.join(data_dir, f"{dataset_name}_full.csv")
    full_df.to_csv(full_path, index=False)

    print(f"Dataset saved to {data_dir}")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(valid_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Full dataset: {len(full_df)} samples")

    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of features: {train_df.shape[1] - 1}")  # Exclude the label column

    # Check if it's a classification task by examining the unique values in the target column
    unique_values = train_df["Y"].nunique()
    is_classification = (
        unique_values <= 10
    )  # Assuming classification if 10 or fewer unique values

    if is_classification:
        print(f"Task type: Classification (found {unique_values} unique classes)")
        if unique_values == 2:
            print("This is a binary classification dataset ✓")
        else:
            print(
                f"This is a multi-class classification dataset with {unique_values} classes ✓"
            )
        # Check class distribution
        print("\nClass distribution:")
        for split_name, df in zip(
            ["Train", "Valid", "Test"], [train_df, valid_df, test_df]
        ):
            class_counts = df["Y"].value_counts()
            print(f"{split_name} set: {dict(class_counts)}")
    else:
        print("This is NOT a classification dataset")

    return full_df


if __name__ == "__main__":
    # List available datasets
    list_available_datasets()

    # Example: Download a classification dataset (Blood-Brain Barrier Penetration)
    # This is a binary classification task that predicts whether a compound can penetrate the blood-brain barrier
    dataset_name = "bbb_martins"
    dataset_group = "ADME"

    dataset = download_dataset(dataset_name, dataset_group)

    # Display the first few rows of the dataset
    print("\nFirst few rows of the dataset:")
    print(dataset.head())

    print(
        "\nTo download a different dataset, modify the dataset_name and dataset_group variables in this script."
    )
