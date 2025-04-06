import os
import pandas as pd
from ersilia import ErsiliaModel

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)


def drop_if_exists(df: pd.DataFrame, columns_to_drop: list[str]):
    # Filter to only include columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    # Drop the existing columns
    if existing_columns:
        return df.drop(columns=existing_columns)
    return df


def featurize_dataset(csv_path: str, ersilia_model_id: str):
    file_name = os.path.basename(csv_path)
    featurized_file_name = file_name.replace(".csv", "_featurized.csv")
    featurized_file_path = os.path.join(data_dir, featurized_file_name)

    df = pd.read_csv(csv_path)
    df = df.head()
    smiles = df["Drug"].tolist()

    ersilia_model = ErsiliaModel(ersilia_model_id)
    ersilia_model.serve()

    temp_output_file = os.path.join(data_dir, "temp_output.csv")
    ersilia_model.run(smiles, output=temp_output_file)
    ersilia_model.close()

    model_results = pd.read_csv(temp_output_file)
    model_results = drop_if_exists(model_results, ["key"])
    model_results = model_results.rename(columns={"input": "Drug"})

    result_df = df.merge(model_results, on="Drug", how="left")
    return result_df


if __name__ == "__main__":
    csv_path = os.path.join(data_dir, "AMES_valid.csv")
    ersilia_model_id = "eos9gg2"

    featurized_df = featurize_dataset(csv_path, ersilia_model_id)
    print(featurized_df)
