"""
Script to view and download datasets available from Therapeutics Data Commons (TDC) and save it to the data folder.
Usage:
    python fetch_dataset.py --list --dataset_group AMES
    python fetch_dataset.py --dataset_name AMES --dataset_group Tox
"""

import os
import argparse
from tdc.single_pred import ADME, Tox, HTS, QM, Yields, Epitope, Develop, CRISPROutcome
from tdc.utils import retrieve_dataset_names

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)

DATASET_GROUP_NAMES = {
    "ADME": ADME,
    "Tox": Tox,
    "HTS": HTS,
    "QM": QM,
    "Yields": Yields,
    "Epitope": Epitope,
    "Develop": Develop,
    "CRISPROutcome": CRISPROutcome,
}


def list_available_single_prediction_datasets() -> None:
    """List single prediction datasets available in TDC."""
    print("Available single response datasets:")
    for dataset_name in DATASET_GROUP_NAMES:
        dataset = retrieve_dataset_names(dataset_name)
        print(f"Datasets available for {dataset_name}:")
        for dataset in dataset:
            print(f" - {dataset}")


def download_dataset(dataset_group: str, dataset_name: str) -> tuple[str, str, str]:
    """
    Download a specific dataset from TDC and save to files.
    Args:
        dataset_name (str): Name of the dataset to download
        dataset_group (str): Group of the dataset ('ADME', 'Tox', ...)
    Returns:
        tuple: A tuple containing the train, validation and test file paths
    """
    data = ""
    print(f"Downloading {dataset_name} dataset from {dataset_group} group...")

    try:
        if dataset_group in DATASET_GROUP_NAMES:
            data = DATASET_GROUP_NAMES[dataset_group](name=dataset_name)
    except Exception as e:
        raise ValueError(f"Unsupported dataset group: {dataset_group}")

    # Get the split
    split = data.get_split()

    # Define paths
    train_path = os.path.join(data_dir, f"{dataset_name}_train.csv")
    valid_path = os.path.join(data_dir, f"{dataset_name}_valid.csv")
    test_path = os.path.join(data_dir, f"{dataset_name}_test.csv")

    # Save the datasets directly
    split["train"].to_csv(train_path, index=False)
    split["valid"].to_csv(valid_path, index=False)
    split["test"].to_csv(test_path, index=False)

    print(f"Dataset saved to {data_dir}")
    print(f"Files created:")
    print(f"- {train_path}")
    print(f"- {valid_path}")
    print(f"- {test_path}")

    return (train_path, valid_path, test_path)


def parse_arguments():
    """Parse command line arguments."""
    # dataset_choices =
    parser = argparse.ArgumentParser(description="Download dataset from TDC")
    parser.add_argument(
        "--dataset_name", type=str, help="Name of the dataset to download"
    )
    parser.add_argument(
        "--dataset_group",
        type=str,
        help="Group of the dataset ('ADME', 'Tox', 'HTS'...)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.list and not (args.dataset_name or args.dataset_group):
        list_available_single_prediction_datasets()
    elif args.dataset_name and args.dataset_group:
        # Download the specified dataset
        download_dataset(args.dataset_group, args.dataset_name)

    else:
        print("Error: Both --dataset_name and --dataset_group must be specified.")
        print(
            "Example: python fetch_dataset.py --dataset_name AMES --dataset_group Tox"
        )
        print("Or use --list to see available datasets.")
