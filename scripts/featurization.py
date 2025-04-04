import os
import pandas as pd
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


def try_predict_method(model, smiles_list: list) -> list:
    """Try to featurize using the predict method."""
    results = model.predict(smiles_list)

    # Extract values from nested lists if needed
    if results and isinstance(results[0], list):
        results = [r[0] if isinstance(r, list) and len(r) > 0 else r for r in results]

    print("Successfully featurized using predict method")
    return results


def try_run_method(model, smiles_list: list) -> list:
    """Try to featurize using the run method with file output."""
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
            # The values might be stored as strings representing lists like "[123.45]"
            results = []
            for value in result_df["outcome"]:
                if (
                    isinstance(value, str)
                    and value.startswith("[")
                    and value.endswith("]")
                ):
                    # Extract the number from the string representation of a list
                    try:
                        # Remove brackets and convert to float
                        cleaned_value = float(value.strip("[]"))
                        results.append(cleaned_value)
                    except ValueError:
                        results.append(None)
                elif isinstance(value, list) and len(value) > 0:
                    # If it's already a list, take the first element
                    results.append(value[0])
                else:
                    # Otherwise, use the value as is
                    results.append(value)

            print("Successfully featurized using run method with file output")
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


def featurize_dataset(dataset_path: str, ersilia_model_id: str) -> str:
    """
    Featurize a dataset using an Ersilia model.

    Args:
        dataset_path (str): Path to the dataset to featurize.
        ersilia_model_id (str): The identifier of the Ersilia model to use.

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

        # Try predict method first
        try:
            results = try_predict_method(model, smiles_list)
            df["MolWeight"] = results
        except Exception as e:
            print(f"Error with predict method: {e}")
            print("Trying alternative approach with run method...")

            # Fall back to run method
            results = try_run_method(model, smiles_list)
            if results:
                df["MolWeight"] = results
            else:
                df["MolWeight"] = None

        # Save the featurized dataset
        df.to_csv(output_path, index=False)
        print(f"Featurized dataset saved to: {output_path}")

        return output_path

    finally:
        # Ensure model is closed even if an error occurs
        model.close()


if __name__ == "__main__":
    """
    Example usage for testing
    """
    ersilia_model_id = "eos3b5e"
    dataset_path = os.path.join(data_dir, "AMES_test.csv")
    featurize_dataset(dataset_path, ersilia_model_id)
