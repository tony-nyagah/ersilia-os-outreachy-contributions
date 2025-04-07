import pandas as pd
import numpy as np
import ast
import os

data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(data_dir, exist_ok=True)


def extract_numeric_from_list_str(value):
    """Extract numeric value from string representation of a list."""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            # Extract the number from the list string
            return float(value.strip("[]"))
        except ValueError:
            # If it's a more complex list, try to parse it
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[0]  # Return the first element
                return np.nan
            except:
                return np.nan
    return value


def fix_list_strings_in_dataframe(df):
    """Fix all columns that might contain string representations of lists."""
    df_copy = df.copy()

    # Get all columns except ID, SMILES, and target columns
    columns_to_check = [
        col for col in df_copy.columns if col not in ["Drug_ID", "Drug", "Y"]
    ]

    # Apply the conversion to each column
    for col in columns_to_check:
        try:
            # Check if the column might contain list strings
            sample = df_copy[col].iloc[0]
            if (
                isinstance(sample, str)
                and sample.startswith("[")
                and sample.endswith("]")
            ):
                print(f"Converting list strings in column: {col}")
                df_copy[col] = df_copy[col].apply(extract_numeric_from_list_str)
        except (IndexError, TypeError):
            continue

    return df_copy


# Example usage:
if __name__ == "__main__":
    # Load datasets
    train_data = pd.read_csv(f"{data_dir}/AMES_train_featurized.csv")
    valid_data = pd.read_csv(f"{data_dir}/AMES_valid_featurized.csv")
    test_data = pd.read_csv(f"{data_dir}/AMES_test_featurized.csv")

    # Fix list strings in each dataset
    train_data_fixed = fix_list_strings_in_dataframe(train_data)
    valid_data_fixed = fix_list_strings_in_dataframe(valid_data)
    test_data_fixed = fix_list_strings_in_dataframe(test_data)

    # Save the fixed datasets
    train_data_fixed.to_csv(f"{data_dir}/AMES_train_featurized.csv", index=False)
    valid_data_fixed.to_csv(f"{data_dir}/AMES_valid_featurized.csv", index=False)
    test_data_fixed.to_csv(f"{data_dir}/AMES_test_featurized.csv", index=False)

    print("Fixed datasets have been saved.")
