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


def process_model_output(model, smiles_list: list) -> list:
    """Process the output from an Ersilia model using the run method."""
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
        if "outcome" in result_df.columns:
            # Extract values from the outcome column
            results = []
            for value in result_df["outcome"]:
                # The values are stored as strings representing lists like "[123.45]"
                if isinstance(value, str):
                    if value.startswith("[") and value.endswith("]"):
                        try:
                            # Remove brackets and convert to float
                            cleaned_value = float(value.strip("[]"))
                            results.append(cleaned_value)
                        except ValueError:
                            results.append(None)
                    else:
                        # Try to convert directly to float if not in brackets
                        try:
                            results.append(float(value))
                        except ValueError:
                            results.append(None)
                # Handle actual list objects
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        results.append(value[0])
                    else:
                        try:
                            results.append(float(value[0]))
                        except (ValueError, TypeError):
                            results.append(None)
                else:
                    # Use the value as is if it's already a number
                    if isinstance(value, (int, float)):
                        results.append(value)
                    else:
                        results.append(None)

            print("Successfully processed model output")
            return results
        else:
            print("Warning: 'outcome' column not found in output file")
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
        results = process_model_output(model, smiles_list)
        if results:
            df[feature_name] = results
        else:
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
