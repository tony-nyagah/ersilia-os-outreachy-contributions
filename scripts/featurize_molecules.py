#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to featurize molecules using a representation model from the Ersilia Model Hub.
This script takes the downloaded dataset and converts the molecular structures
into numerical features that can be used for machine learning.
"""

import os
import sys
import pandas as pd
import numpy as np
from ersilia import ErsiliaModel
import argparse

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(data_dir, exist_ok=True)

def featurize_molecules(input_file, model_id, output_file=None):
    """
    Featurize molecules using an Ersilia model.
    
    Args:
        input_file (str): Path to the input CSV file containing molecules
        model_id (str): Ersilia model ID for the featurizer
        output_file (str, optional): Path to save the featurized data
        
    Returns:
        pandas.DataFrame: DataFrame with the original data and added features
    """
    print(f"Featurizing molecules using Ersilia model {model_id}...")
    
    # Read the input file
    df = pd.read_csv(input_file)
    print(f"Read {len(df)} molecules from {input_file}")
    
    # Check if the dataset has a 'Drug' or 'SMILES' column
    smiles_col = None
    for col in ['Drug', 'SMILES', 'smiles', 'Smiles', 'drug', 'compound', 'Compound']:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError("Could not find a column containing SMILES strings in the dataset")
    
    print(f"Using column '{smiles_col}' as the source of SMILES strings")
    
    # Initialize the Ersilia model
    print(f"Initializing Ersilia model {model_id}...")
    model = ErsiliaModel(model_id)
    
    # Serve the model
    print("Serving the model...")
    model.serve()
    
    try:
        # Get the SMILES strings
        smiles_list = df[smiles_col].tolist()
        
        # Featurize the molecules in batches to avoid memory issues
        batch_size = 100
        all_features = []
        
        for i in range(0, len(smiles_list), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{(len(smiles_list)-1)//batch_size + 1}...")
            batch = smiles_list[i:i+batch_size]
            
            # Run the model on the batch
            features = model.predict(batch)
            all_features.extend(features)
        
        # Convert features to a DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Rename columns to avoid conflicts
        features_df.columns = [f'feature_{i}' for i in range(features_df.shape[1])]
        
        # Combine the original data with the features
        result_df = pd.concat([df, features_df], axis=1)
        
        # Save the featurized data if output_file is provided
        if output_file:
            result_df.to_csv(output_file, index=False)
            print(f"Featurized data saved to {output_file}")
        
        # Print some statistics about the features
        print(f"\nFeaturization complete!")
        print(f"Number of features generated: {features_df.shape[1]}")
        
        return result_df
    
    finally:
        # Close the model to free resources
        print("Closing the model...")
        model.close()

def main():
    parser = argparse.ArgumentParser(description='Featurize molecules using an Ersilia model')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file containing molecules')
    parser.add_argument('--model', '-m', required=True, help='Ersilia model ID for the featurizer')
    parser.add_argument('--output', '-o', help='Output CSV file to save the featurized data')
    
    args = parser.parse_args()
    
    # If output is not specified, create a default output filename
    if not args.output:
        input_basename = os.path.basename(args.input)
        output_basename = f"featurized_{input_basename}"
        args.output = os.path.join(data_dir, output_basename)
    
    # Featurize the molecules
    featurize_molecules(args.input, args.model, args.output)

if __name__ == "__main__":
    main()
