"""
Script to download a classification dataset from Therapeutics Data Commons (TDC)
and save it to the data folder.

Usage:
    python fetch_dataset.py --dataset_name AMES --dataset_group Tox
"""

import os
import argparse
from tdc.single_pred import ADME, Tox
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


def download_dataset(
    dataset_name: str, dataset_group: str
) -> tuple[str, str, str, str]:
    """
    Download a specific dataset from TDC and save to files.

    Args:
        dataset_name (str): Name of the dataset to download
        dataset_group (str): Group of the dataset ('ADME' or 'Tox')

    Returns:
        tuple: A tuple containing the train, validation, test, and full dataset file paths
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

    # Define paths
    train_path = os.path.join(data_dir, f"{dataset_name}_train.csv")
    valid_path = os.path.join(data_dir, f"{dataset_name}_valid.csv")
    test_path = os.path.join(data_dir, f"{dataset_name}_test.csv")
    full_path = os.path.join(data_dir, f"{dataset_name}_full.csv")

    # Save the datasets directly
    split["train"].to_csv(train_path, index=False)
    split["valid"].to_csv(valid_path, index=False)
    split["test"].to_csv(test_path, index=False)

    # For the full dataset, we need to concatenate but we'll do it minimally
    # Using pandas just for the concat and save operation
    import pandas as pd

    pd.concat([split["train"], split["valid"], split["test"]]).to_csv(
        full_path, index=False
    )

    print(f"Dataset saved to {data_dir}")
    print(f"Files created:")
    print(f"- {train_path}")
    print(f"- {valid_path}")
    print(f"- {test_path}")
    print(f"- {full_path}")

    return (train_path, valid_path, test_path, full_path)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download dataset from TDC")
    parser.add_argument(
        "--dataset_name", type=str, help="Name of the dataset to download"
    )
    parser.add_argument(
        "--dataset_group", type=str, help="Group of the dataset ('ADME' or 'Tox')"
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.list or (not args.dataset_name and not args.dataset_group):
        # List available datasets if requested or if no dataset specified
        list_available_datasets()
        if not args.list:
            print(
                "\nNo dataset specified. Use --dataset_name and --dataset_group to download a dataset."
            )
            print(
                "Example: python fetch_dataset.py --dataset_name AMES --dataset_group Tox"
            )
    elif args.dataset_name and args.dataset_group:
        # Download the specified dataset
        download_dataset(args.dataset_name, args.dataset_group)

    else:
        print("Error: Both --dataset_name and --dataset_group must be specified.")
        print(
            "Example: python fetch_dataset.py --dataset_name AMES --dataset_group Tox"
        )
        print("Or use --list to see available datasets.")
