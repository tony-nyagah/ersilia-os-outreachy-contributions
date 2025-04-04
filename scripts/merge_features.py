"""
Script to merge features from multiple featurized datasets.

Usage:
    python merge_features.py --datasets AMES_train_featurized_eos3b5e.csv AMES_train_featurized_eos4wt0.csv --output AMES_train_combined.csv
"""

import os
import pandas as pd
import argparse

# Get the absolute path to the data directory
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def load_dataset(file_path):
    """Load a dataset from a CSV file."""
    if not os.path.isabs(file_path):
        file_path = os.path.join(data_dir, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return pd.read_csv(file_path)


def merge_datasets(datasets, output_file):
    """Merge multiple datasets based on Drug_ID."""
    if not datasets:
        raise ValueError("No datasets provided for merging")

    # Load the first dataset
    merged_df = load_dataset(datasets[0])
    print(f"Loaded first dataset: {datasets[0]} with shape {merged_df.shape}")

    # Merge with other datasets
    for dataset in datasets[1:]:
        df = load_dataset(dataset)
        print(f"Merging with dataset: {dataset} with shape {df.shape}")

        # Get columns to merge (exclude common columns like Drug_ID, Drug, Y)
        common_cols = ["Drug_ID", "Drug", "Y"]
        merge_cols = [
            col for col in df.columns if col not in common_cols or col == "Drug_ID"
        ]

        # Merge on Drug_ID
        merged_df = pd.merge(merged_df, df[merge_cols], on="Drug_ID", how="inner")
        print(f"Merged dataset shape: {merged_df.shape}")

    # Save the merged dataset
    output_path = os.path.join(data_dir, output_file)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged dataset saved to: {output_path}")

    return output_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge features from multiple featurized datasets."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="List of featurized dataset files to merge",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename for the merged dataset",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Merge datasets
    merge_datasets(args.datasets, args.output)
