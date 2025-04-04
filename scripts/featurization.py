import os
import pandas as pd
import glob

from ersilia import ErsiliaModel

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def featurize_dataset(dataset_path: str, ersilia_model_id: str):
    """
    Featurize a dataset using an Ersilia model.

    Args:
        dataset_path (str): Path to the dataset to featurize.
        ersilia_model_id (str): The identifier of the Ersilia model to use.
    """
    model: ErsiliaModel = ErsiliaModel(ersilia_model_id)
    model.serve()

    dataset_name = os.path.basename(dataset_path)
    print(f"Featurizing {dataset_name} dataset using {ersilia_model_id}...")


if __name__ == "__main__":
    """
    Example usage for testing
    """
    ersilia_model_id = "eos3b5e"

    for n in ["train", "valid", "test"]:
        dataset = glob.glob(os.path.join(data_dir, f"*{n}.csv"))
        dataset_name = os.path.basename(dataset[0])
        print(f"Featurizing {dataset_name} dataset using {ersilia_model_id}...")
