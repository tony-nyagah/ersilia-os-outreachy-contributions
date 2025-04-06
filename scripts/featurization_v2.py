import os
from numpy.core.fromnumeric import trace
import pandas as pd
import argparse
from ersilia import ErsiliaModel
from mypy_extensions import Arg

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)


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


def convert_to_dataframe(file_path: str):
    """
    Convert a CSV file to a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The DataFrame representation of the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error occurred while converting {file_path} to DataFrame: {e}")
        return None


def get_output_path(dataset_path: str):
    """
    Get the output path for the featurized dataset.

    Args:
        dataset_path (str): The path to the dataset CSV file.

    Returns:
        str: The output path for the featurized dataset.
    """
    dataset_name = os.path.basename(dataset_path)
    try:
        # check if the output file already exists
        output_path = f"{data_dir}/{os.path.splitext(dataset_name)[0]}_featurized.csv"
        return output_path
    except Exception as e:
        print(f"Error occurred while getting output path: {e}")
        return None


def add_features_to_df(dataset_path: str, ersilia_model_id: str):
    """
    Add features to a DataFrame using an Ersilia model.
    Args:
        dataset_path (str): The path to the dataset CSV file.
        ersilia_model_id (str): The ID of the Ersilia model to use.
    Returns:
        pd.DataFrame: The DataFrame with added features.
    """
    model = ErsiliaModel(ersilia_model_id)
    model.serve()
    try:
        df = convert_to_dataframe(dataset_path)
        smiles = df["Drug"].tolist()
        features = process_model_output(model, smiles, ersilia_model_id)
        if features:
            df = pd.concat([df, pd.DataFrame(features)], axis=1)
            output_path = get_output_path(dataset_path)
            df.to_csv(output_path, index=False)
            return df
    except Exception as e:
        print(f"Error occurred while adding features to DataFrame: {e}")
        return None


def featurize_dataset(dataset_name: str, model_id: str):
    """
    Take an dataset name and an Ersilia model ID and use that to featurize a dataset.
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

    ersilia_model = ErsiliaModel(model_id)
    try:
        # First serve the model
        ersilia_model.serve()

        # Process each dataset file
        featurized_files = []
        for file in [train_file, test_file, valid_file]:
            file_path = available_datasets[file]
            print(f"Processing {file}...")

            # Read the dataset
            df = convert_to_dataframe(file_path)
            if df is None:
                print(f"Failed to read {file}")
                continue

            # Get SMILES strings
            if "Drug" not in df.columns:
                print(f"Error: 'Drug' column not found in {file}")
                continue

            smiles = df["Drug"].tolist()

            # Run the model
            try:
                results = ersilia_model.run(smiles)

                # Process results - extract features
                if results:
                    # Get model-specific feature information
                    feature_info = get_feature_info(model_id)

                    # Initialize feature columns
                    feature_data = {
                        feature_name: [] for feature_name in feature_info["names"]
                    }

                    # Process each result
                    for result in results:
                        if "output" in result and isinstance(result["output"], list):
                            # Handle based on feature count
                            if feature_info["count"] == 1:
                                # Single feature models
                                try:
                                    feature_data[feature_info["names"][0]].append(
                                        float(result["output"][0])
                                    )
                                except (ValueError, TypeError, IndexError):
                                    feature_data[feature_info["names"][0]].append(None)
                            else:
                                # Multi-feature models
                                for i, feature_name in enumerate(feature_info["names"]):
                                    try:
                                        if i < len(result["output"]):
                                            feature_data[feature_name].append(
                                                float(result["output"][i])
                                            )
                                        else:
                                            feature_data[feature_name].append(None)
                                    except (ValueError, TypeError):
                                        feature_data[feature_name].append(None)
                        else:
                            # No output data
                            for feature_name in feature_info["names"]:
                                feature_data[feature_name].append(None)

                    # Add all features to dataframe
                    for feature_name, values in feature_data.items():
                        df[feature_name] = values

                    # Save featurized dataset
                    output_path = get_output_path(file_path)
                    df.to_csv(output_path, index=False)
                    featurized_files.append(output_path)
                    print(f"Featurized {file} saved to {output_path}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

        # Close the model
        ersilia_model.close()

        if featurized_files:
            return featurized_files
        else:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            ersilia_model.close()
        except:
            pass
        return None


def get_feature_info(model_id):
    """
    Get information about features for a specific model.

    Args:
        model_id (str): The ID of the Ersilia model.

    Returns:
        dict: A dictionary containing feature information.
    """
    # Model-specific feature information
    feature_info = {
        "eos3b5e": {"count": 1, "names": ["molecular_weight"]},
        "eos2r5a": {"count": 1, "names": ["morgan_fingerprint"]},
        "eos9gg2": {
            "count": 8,
            "names": [
                "pca_1",
                "pca_2",
                "pca_3",
                "pca_4",
                "umap_1",
                "umap_2",
                "tsne_1",
                "tsne_2",
            ],
        },
    }

    # Return the model-specific feature info if available, otherwise use a generic one
    if model_id in feature_info:
        return feature_info[model_id]
    else:
        return {"count": 1, "names": [f"{model_id}_feature"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    args = parser.parse_args()
    featurize_dataset(args.dataset_name, args.model_id)
