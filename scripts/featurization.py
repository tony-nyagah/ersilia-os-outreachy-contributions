import os
import pandas as pd
from ersilia import ErsiliaModel

# Get the absolute path to the data directory
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)


def featurize_dataset(dataset_path: str, ersilia_model_id: str):
    """
    Featurize a dataset using an Ersilia model.

    Args:
        dataset_path (str): Path to the dataset to featurize.
        ersilia_model_id (str): The identifier of the Ersilia model to use.
    """
    # Initialize and serve the model
    model = ErsiliaModel(ersilia_model_id)
    model.serve()

    dataset_name = os.path.basename(dataset_path)
    dataset_name_featurized = f"{data_dir}/{os.path.splitext(dataset_name)[0]}_featurized_{ersilia_model_id}.csv"

    df = pd.read_csv(dataset_path)
    smiles_list = df["Drug"].tolist()

    print(f"Featurizing {dataset_name} dataset using {ersilia_model_id}...")

    try:
        # First approach: use predict method which returns results directly
        results = model.predict(smiles_list)

        # Extract values from nested lists if needed
        if results and isinstance(results[0], list):
            results = [
                r[0] if isinstance(r, list) and len(r) > 0 else r for r in results
            ]

        df["MolWeight"] = results
        print(f"Successfully featurized using predict method")

    except Exception as e:
        print(f"Error with predict method: {e}")
        print("Trying alternative approach with run method...")

        # Create a temporary file for the SMILES strings
        temp_smiles_file = os.path.join(data_dir, "temp_smiles.csv")
        with open(temp_smiles_file, "w") as f:
            f.write("\n".join(smiles_list))

        # Run the model with output to a file
        temp_output_file = os.path.join(data_dir, "temp_output.csv")
        model.run(input=temp_smiles_file, output=temp_output_file)

        # Read the results
        try:
            result_df = pd.read_csv(temp_output_file)
            if "outcome" in result_df.columns:
                df["MolWeight"] = result_df["outcome"].tolist()
                print(f"Successfully featurized using run method with file output")
            else:
                print(f"Warning: 'outcome' column not found in output file")
                df["MolWeight"] = None

            # Clean up temporary files
            if os.path.exists(temp_smiles_file):
                os.remove(temp_smiles_file)
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)

        except Exception as e2:
            print(f"Error processing output file: {e2}")
            df["MolWeight"] = None

    # Close the model
    model.close()

    # Save the featurized dataset
    df.to_csv(dataset_name_featurized, index=False)
    print(f"Featurized dataset saved to: {dataset_name_featurized}")

    return dataset_name_featurized


if __name__ == "__main__":
    """
    Example usage for testing
    """
    ersilia_model_id = "eos3b5e"
    dataset_path = os.path.join(data_dir, "AMES_test.csv")
    featurize_dataset(dataset_path, ersilia_model_id)
