"""
Script to train a machine learning model on the featurized AMES dataset.

Usage:
    python train_model.py --train AMES_train_featurized_eos3b5e.csv --valid AMES_valid_featurized_eos3b5e.csv --test AMES_test_featurized_eos3b5e.csv --feature MolWeight
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import joblib

# Get the absolute path to the data directory
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
# Create models directory if it doesn't exist
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(models_dir, exist_ok=True)


def load_dataset(file_path):
    """Load a dataset from a CSV file."""
    if not os.path.isabs(file_path):
        file_path = os.path.join(data_dir, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return pd.read_csv(file_path)


def prepare_data(df, feature_name=None):
    """Prepare the data for model training.

    If feature_name is provided, only that feature will be used.
    If feature_name is None, all numerical columns except 'Drug_ID', 'Drug', and 'Y' will be used.
    """
    # If feature_name is provided, check if it exists
    if feature_name:
        if feature_name not in df.columns:
            raise ValueError(f"Feature column '{feature_name}' not found in dataset")

        # Check if there are any missing values
        if df[feature_name].isna().any():
            print(
                f"Warning: {df[feature_name].isna().sum()} missing values in {feature_name}"
            )
            print("Dropping rows with missing values")
            df = df.dropna(subset=[feature_name])

        # Prepare X (features) and y (target)
        X = df[[feature_name]]
    else:
        # Use all numerical columns except 'Drug_ID', 'Drug', and 'Y'
        exclude_cols = ["Drug_ID", "Drug", "Y"]
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not feature_cols:
            raise ValueError("No feature columns found in dataset")

        print(f"Using {len(feature_cols)} features: {', '.join(feature_cols)}")

        # Check for missing values
        missing_counts = df[feature_cols].isna().sum()
        if missing_counts.sum() > 0:
            print("Warning: Missing values detected in features:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} missing values")
            print("Dropping rows with missing values")
            df = df.dropna(subset=feature_cols)

        # Prepare X (features) and y (target)
        X = df[feature_cols]

    # Prepare y (target)
    y = df["Y"]

    return X, y


def train_model(X_train, y_train, model_type="rf"):
    """Train a machine learning model."""
    print(f"Training {model_type} model...")

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == "gb":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the model
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_model(model, scaler, X, y, dataset_name):
    """Evaluate the model on a dataset."""
    # Scale the features
    X_scaled = scaler.transform(X)

    # Make predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)

    # Print metrics
    print(f"\nPerformance on {dataset_name} dataset:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # Calculate and print confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


def save_model(model, scaler, feature_name, model_type):
    """Save the trained model and scaler."""
    model_filename = os.path.join(
        models_dir, f"ames_mutagenicity_{model_type}_{feature_name}.joblib"
    )
    scaler_filename = os.path.join(
        models_dir, f"ames_mutagenicity_{model_type}_{feature_name}_scaler.joblib"
    )

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"\nModel saved to: {model_filename}")
    print(f"Scaler saved to: {scaler_filename}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a machine learning model on the featurized AMES dataset."
    )
    parser.add_argument(
        "--train",
        type=str,
        help="Path to the featurized training dataset",
        required=True,
    )
    parser.add_argument(
        "--valid",
        type=str,
        help="Path to the featurized validation dataset",
        required=True,
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Path to the featurized test dataset",
        required=True,
    )
    parser.add_argument(
        "--feature",
        type=str,
        help="Name of the feature column to use for training",
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "gb"],
        default="rf",
        help="Type of model to train (rf: Random Forest, gb: Gradient Boosting)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Load datasets
    print(f"Loading datasets...")
    train_df = load_dataset(args.train)
    valid_df = load_dataset(args.valid)
    test_df = load_dataset(args.test)

    print(f"Train dataset shape: {train_df.shape}")
    print(f"Validation dataset shape: {valid_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")

    # Prepare data
    X_train, y_train = prepare_data(train_df, args.feature)
    X_valid, y_valid = prepare_data(valid_df, args.feature)
    X_test, y_test = prepare_data(test_df, args.feature)

    # Train model
    model, scaler = train_model(X_train, y_train, model_type=args.model)

    # Evaluate model
    train_metrics = evaluate_model(model, scaler, X_train, y_train, "training")
    valid_metrics = evaluate_model(model, scaler, X_valid, y_valid, "validation")
    test_metrics = evaluate_model(model, scaler, X_test, y_test, "test")

    # Save model
    if args.feature:
        save_model(model, scaler, args.feature, args.model)
    else:
        save_model(model, scaler, "all_features", args.model)

    # Print feature importance (for Random Forest)
    if args.model == "rf":
        print("\nFeature Importance:")
        importance = model.feature_importances_
        for i, feat in enumerate(X_train.columns):
            print(f"{feat}: {importance[i]:.4f}")
