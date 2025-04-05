import os
import pandas as pd
import argparse
from ersilia import ErsiliaModel

# Get the absolute path to the data directory
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)


def get_output_path(dataset_path: str, model_id: str) -> str:
    """Generate the output path for the featurized dataset."""
    dataset_name = os.path.basename(dataset_path)
    return f"{data_dir}/{os.path.splitext(dataset_name)[0]}_featurized_{model_id}.csv"


def extract_smiles(dataset_path: str) -> tuple:
    """Extract SMILES strings from the dataset."""
    df = pd.read_csv(dataset_path)
    smiles_list = df["Drug"].tolist()
    return df, smiles_list


def process_model_output(model, smiles_list: list, feature_name: str) -> dict:
    """Process the output from an Ersilia model using the run method.

    Returns a dictionary of feature columns and their values.
    """
    # Create a temporary file for the SMILES strings
    temp_smiles_file = os.path.join(data_dir, "temp_smiles.csv")
    temp_output_file = os.path.join(data_dir, "temp_output.csv")

    try:
        # Write SMILES to temp file
        with open(temp_smiles_file, "w") as f:
            f.write("\n".join(smiles_list))

        # Run the model with output to a file
        model.run(input=temp_smiles_file, output=temp_output_file)

        # Read the results
        result_df = pd.read_csv(temp_output_file)

        # Check for outcome column (standard format)
        if "outcome" in result_df.columns:
            # Initialize a dictionary to store feature columns
            feature_columns = {}

            # Process each row in the outcome column
            for idx, value in enumerate(result_df["outcome"]):
                # Handle different types of outcome values
                if isinstance(value, str):
                    # Try to parse as a list if it looks like one
                    if value.startswith("[") and value.endswith("]"):
                        try:
                            # Convert string representation of list to actual list
                            import ast

                            value_list = ast.literal_eval(value)

                            # If it's a list with multiple values, create multiple features
                            if isinstance(value_list, list) and len(value_list) > 1:
                                for i, val in enumerate(value_list):
                                    col_name = f"{feature_name}_{i+1}"
                                    if col_name not in feature_columns:
                                        feature_columns[col_name] = [None] * len(
                                            smiles_list
                                        )
                                    feature_columns[col_name][idx] = val
                            elif isinstance(value_list, list) and len(value_list) == 1:
                                # Single value in a list
                                if feature_name not in feature_columns:
                                    feature_columns[feature_name] = [None] * len(
                                        smiles_list
                                    )
                                feature_columns[feature_name][idx] = value_list[0]
                        except (ValueError, SyntaxError):
                            # If parsing fails, treat as a single value
                            if feature_name not in feature_columns:
                                feature_columns[feature_name] = [None] * len(
                                    smiles_list
                                )
                            try:
                                feature_columns[feature_name][idx] = float(
                                    value.strip("[]")
                                )
                            except ValueError:
                                feature_columns[feature_name][idx] = None
                    else:
                        # Regular string value
                        if feature_name not in feature_columns:
                            feature_columns[feature_name] = [None] * len(smiles_list)
                        try:
                            feature_columns[feature_name][idx] = float(value)
                        except ValueError:
                            feature_columns[feature_name][idx] = None

                # Handle actual list objects
                elif isinstance(value, list):
                    if len(value) > 1:
                        # Multiple values in a list
                        for i, val in enumerate(value):
                            col_name = f"{feature_name}_{i+1}"
                            if col_name not in feature_columns:
                                feature_columns[col_name] = [None] * len(smiles_list)
                            feature_columns[col_name][idx] = val
                    elif len(value) == 1:
                        # Single value in a list
                        if feature_name not in feature_columns:
                            feature_columns[feature_name] = [None] * len(smiles_list)
                        feature_columns[feature_name][idx] = value[0]

                # Handle numeric values
                elif isinstance(value, (int, float)):
                    if feature_name not in feature_columns:
                        feature_columns[feature_name] = [None] * len(smiles_list)
                    feature_columns[feature_name][idx] = value

                # Handle None or other types
                else:
                    if feature_name not in feature_columns:
                        feature_columns[feature_name] = [None] * len(smiles_list)
                    feature_columns[feature_name][idx] = None

            if feature_columns:
                print(
                    f"Successfully processed model output with {len(feature_columns)} features"
                )
                return feature_columns
            else:
                print("Warning: No features extracted from outcome column")
                return None

        # Check for specific feature columns (like PCA, UMAP, tSNE)
        elif any(
            col.startswith(("pca_", "umap_", "tsne_")) for col in result_df.columns
        ):
            # Initialize a dictionary to store feature columns
            feature_columns = {}

            # Get all feature columns (pca, umap, tsne)
            feature_cols = [
                col
                for col in result_df.columns
                if col.startswith(("pca_", "umap_", "tsne_"))
            ]

            if feature_cols:
                print(
                    f"Found {len(feature_cols)} feature columns: {', '.join(feature_cols)}"
                )

                # Process each feature column
                for col in feature_cols:
                    col_name = f"{feature_name}_{col}"  # e.g., ChemSpaceProjectionDrugbank_pca_1
                    feature_columns[col_name] = result_df[col].tolist()

                print(
                    f"Successfully processed model output with {len(feature_columns)} features"
                )
                return feature_columns
            else:
                print("Warning: No feature columns found in output file")
                return None
        else:
            print(
                "Warning: Neither 'outcome' column nor feature columns found in output file"
            )
            print(f"Available columns: {', '.join(result_df.columns)}")
            return None
    except Exception as e:
        print(f"Error processing output file: {e}")
        return None
    finally:
        # Clean up temporary files
        for file_path in [temp_smiles_file, temp_output_file]:
            if os.path.exists(file_path):
                os.remove(file_path)


def featurize_dataset(
    dataset_path: str, ersilia_model_id: str, feature_name: str
) -> str:
    """
    Featurize a dataset using an Ersilia model.

    Args:
        dataset_path (str): Path to the dataset to featurize.
        ersilia_model_id (str): The identifier of the Ersilia model to use.
        feature_name (str): Name of the feature to use for featurization, to be added as a new column in the dataset.

    Returns:
        str: Path to the featurized dataset.
    """
    # Initialize and serve the model
    model = ErsiliaModel(ersilia_model_id)
    model.serve()

    try:
        # Setup
        output_path = get_output_path(dataset_path, ersilia_model_id)
        df, smiles_list = extract_smiles(dataset_path)

        print(
            f"Featurizing {os.path.basename(dataset_path)} dataset using {ersilia_model_id}..."
        )

        # Process the model output
        feature_columns = process_model_output(model, smiles_list, feature_name)
        if feature_columns:
            # Add each feature column to the dataframe
            for col_name, values in feature_columns.items():
                # Check if lengths match, if not, handle the mismatch
                if len(values) != len(df):
                    print(
                        f"Warning: Length mismatch between feature values ({len(values)}) and dataset ({len(df)})"
                    )
                    # Create a new column with NaN values
                    df[col_name] = None

                    # Map values to the dataframe based on SMILES
                    # First, create a mapping from SMILES to feature value
                    smiles_to_feature = {}
                    for i, smiles in enumerate(smiles_list[: len(values)]):
                        smiles_to_feature[smiles] = values[i]

                    # Then, fill in the values where SMILES match
                    for i, smiles in enumerate(df["Drug"]):
                        if smiles in smiles_to_feature:
                            df.at[i, col_name] = smiles_to_feature[smiles]
                else:
                    # If lengths match, just assign the values directly
                    df[col_name] = values
        else:
            # If no features were extracted, add an empty column
            df[feature_name] = None

        # Save the featurized dataset
        df.to_csv(output_path, index=False)
        print(f"Featurized dataset saved to: {output_path}")

        return output_path

    finally:
        # Ensure model is closed even if an error occurs
        model.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Featurize a chemical dataset using an Ersilia model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the dataset CSV file containing SMILES strings in a 'Drug' column",
        required=True,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Ersilia model ID to use for featurization (e.g., 'eos3b5e')",
        required=True,
    )
    parser.add_argument(
        "--feature_name",
        type=str,
        help="Name of the feature column to add to the dataset",
        default="Feature",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Get dataset path - if it's not an absolute path, assume it's relative to data directory
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(data_dir, dataset_path)

    # Ensure the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        exit(1)

    # Featurize the dataset
    featurize_dataset(dataset_path, args.model_id, args.feature_name)
