import os
import pandas as pd
import argparse
from ersilia import ErsiliaModel

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)


def drop_if_exists(df: pd.DataFrame, columns_to_drop: list[str]):
    """
    Drop columns from a DataFrame if they exist.

    Args:
        df (pd.DataFrame): The DataFrame to drop columns from.
        columns_to_drop (list[str]): List of column names to drop.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns dropped.
    """
    # Filter to only include columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    # Drop the existing columns
    if existing_columns:
        return df.drop(columns=existing_columns)
    return df


def get_datasets(prefix: str):
    """
    Get datasets from the data directory with a given prefix.

    Args:
        prefix (str): The prefix to filter the dataset names.

    Returns:
        dict: A dictionary of dataset names and their file paths.
    """
    datasets = {}
    try:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_name = os.path.basename(file)
                if file_name.startswith(prefix) and file_name.endswith(".csv"):
                    datasets[file_name] = os.path.join(root, file)
    except Exception as e:
        print(f"Error occurred while getting datasets: {e}")
    return datasets


def featurize_file(file_path: str, ersilia_model_id: str, feature_name: str = None):
    """
    Featurize a single file using an Ersilia model.

    Args:
        file_path (str): Path to the CSV file.
        ersilia_model_id (str): ID of the Ersilia model to use.
        feature_name (str, optional): Custom name for the feature column if model returns a single result.

    Returns:
        pd.DataFrame: Featurized DataFrame.
    """
    print(f"Processing {os.path.basename(file_path)}...")

    # Get output file path
    file_name = os.path.basename(file_path)
    featurized_file_name = file_name.replace(".csv", "_featurized.csv")
    featurized_file_path = os.path.join(data_dir, featurized_file_name)

    # Read input file
    df = pd.read_csv(file_path)
    smiles = df["Drug"].tolist()

    # Initialize and run the model
    ersilia_model = ErsiliaModel(ersilia_model_id)
    ersilia_model.serve()

    # Run the model and save output to a temporary file
    temp_output_file = os.path.join(data_dir, "temp_output.csv")
    ersilia_model.run(smiles, output=temp_output_file)
    ersilia_model.close()

    # Process model output
    model_results = pd.read_csv(temp_output_file)
    model_results = drop_if_exists(model_results, ["key"])
    model_results = model_results.rename(columns={"input": "Drug"})

    # Rename outcome/score column if feature_name is provided and model returns a single result
    if (
        feature_name
        and len(model_results.columns) == 2
        and any(col in model_results.columns for col in ["outcome", "score"])
    ):
        column_to_rename = "outcome" if "outcome" in model_results.columns else "score"
        model_results = model_results.rename(columns={column_to_rename: feature_name})
        print(f"Renamed '{column_to_rename}' column to '{feature_name}'")

    # Merge with original data
    result_df = df.merge(model_results, on="Drug", how="left")

    # Check if a featurized file already exists
    if os.path.exists(featurized_file_path):
        print(f"Found existing featurized file: {featurized_file_path}")
        existing_df = pd.read_csv(featurized_file_path)

        # Get columns that are in result_df but not in existing_df
        new_columns = [
            col for col in result_df.columns if col not in existing_df.columns
        ]

        if not new_columns:
            print(f"All features already exist in {featurized_file_path}, skipping...")
            return existing_df

        print(f"Adding new features: {new_columns}")

        # Use pd.concat to add new columns efficiently instead of adding them one by one
        # This avoids DataFrame fragmentation and improves performance
        new_features_df = result_df[new_columns].copy()
        existing_df = pd.concat([existing_df, new_features_df], axis=1)

        # Save updated dataframe
        existing_df.to_csv(featurized_file_path, index=False)
        return existing_df
    else:
        # Save as new featurized dataset
        result_df.to_csv(featurized_file_path, index=False)
        print(f"Featurized file saved to {featurized_file_path}")
        return result_df


def featurize_dataset(dataset_name: str, model_id: str, feature_name: str = None):
    """
    Featurize all files for a dataset using an Ersilia model.

    Args:
        dataset_name (str): Name of the dataset.
        model_id (str): ID of the Ersilia model to use.
        feature_name (str, optional): Custom name for the feature column if model returns a single result.

    Returns:
        list: Paths to the featurized files.
    """
    available_datasets = get_datasets(dataset_name)

    train_file = f"{dataset_name}_train.csv"
    test_file = f"{dataset_name}_test.csv"
    valid_file = f"{dataset_name}_valid.csv"

    # Check if all required files exist
    missing_files = []
    for file in [train_file, test_file, valid_file]:
        if file not in available_datasets:
            missing_files.append(file)

    if missing_files:
        print(f"Error: The following files are missing: {missing_files}")
        print(f"Available datasets: {list(available_datasets.keys())}")
        return None

    featurized_files = []
    for file in [train_file, test_file, valid_file]:
        file_path = available_datasets[file]
        try:
            featurized_df = featurize_file(file_path, model_id, feature_name)
            if featurized_df is not None:
                featurized_file_name = file.replace(".csv", "_featurized.csv")
                featurized_files.append(os.path.join(data_dir, featurized_file_name))
        except Exception as e:
            print(f"Error featurizing {file}: {e}")

    return featurized_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument(
        "--feature_name",
        type=str,
        help="Custom name for the feature column if model returns a single result",
    )
    args = parser.parse_args()
    featurize_dataset(args.dataset_name, args.model_id, args.feature_name)
