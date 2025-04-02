#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to build and evaluate a machine learning model using the featurized data.
This script implements various ML algorithms and evaluates their performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import argparse

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create directories if they don't exist
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def load_data(input_file):
    """
    Load the featurized data from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file containing featurized molecules
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    print(f"Loading data from {input_file}...")
    
    # Read the input file
    df = pd.read_csv(input_file)
    print(f"Read {len(df)} samples from {input_file}")
    
    # Identify feature columns (those starting with 'feature_')
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    
    if not feature_cols:
        raise ValueError("No feature columns found in the dataset. Make sure the data has been featurized.")
    
    print(f"Found {len(feature_cols)} feature columns")
    
    # Identify the target column
    target_candidates = ['Y', 'y', 'target', 'Target', 'label', 'Label', 'class', 'Class']
    target_col = None
    
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Could not find a target column in the dataset")
    
    print(f"Using '{target_col}' as the target column")
    
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y, feature_cols, target_col

def train_and_evaluate_model(X, y, model_type='xgboost', test_size=0.2, random_state=42):
    """
    Train and evaluate a machine learning model.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        model_type (str): Type of model to train ('xgboost' or 'random_forest')
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred, y_prob)
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    if model_type.lower() == 'xgboost':
        print("Training XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=random_state
        )
    elif model_type.lower() == 'random_forest':
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Get probability estimates for ROC curve
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = None
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, y_prob

def evaluate_model(y_test, y_pred, y_prob=None):
    """
    Evaluate the model performance.
    
    Args:
        y_test (numpy.ndarray): True target values
        y_pred (numpy.ndarray): Predicted target values
        y_prob (numpy.ndarray, optional): Predicted probabilities for the positive class
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Print metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def plot_results(y_test, y_pred, y_prob=None, output_dir=None):
    """
    Plot the results of the model evaluation.
    
    Args:
        y_test (numpy.ndarray): True target values
        y_pred (numpy.ndarray): Predicted target values
        y_prob (numpy.ndarray, optional): Predicted probabilities for the positive class
        output_dir (str, optional): Directory to save the plots
    """
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'notebooks')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot ROC curve if probabilities are available
    if y_prob is not None:
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")

def save_model(model, model_name, output_dir=None):
    """
    Save the trained model.
    
    Args:
        model: The trained model
        model_name (str): Name of the model
        output_dir (str, optional): Directory to save the model
    """
    if output_dir is None:
        output_dir = models_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    import joblib
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Build and evaluate a machine learning model')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file containing featurized molecules')
    parser.add_argument('--model-type', '-m', default='xgboost', choices=['xgboost', 'random_forest'],
                        help='Type of model to train')
    parser.add_argument('--output-dir', '-o', default=None, help='Directory to save the model and plots')
    parser.add_argument('--model-name', '-n', default=None, help='Name of the model')
    
    args = parser.parse_args()
    
    # Set default model name if not provided
    if args.model_name is None:
        args.model_name = f"{os.path.basename(args.input).split('.')[0]}_{args.model_type}"
    
    # Load the data
    X, y, feature_cols, target_col = load_data(args.input)
    
    # Train and evaluate the model
    model, X_train, X_test, y_train, y_test, y_pred, y_prob = train_and_evaluate_model(
        X, y, model_type=args.model_type
    )
    
    # Evaluate the model
    metrics = evaluate_model(y_test, y_pred, y_prob)
    
    # Plot the results
    plot_results(y_test, y_pred, y_prob, args.output_dir)
    
    # Save the model
    save_model(model, args.model_name, args.output_dir)

if __name__ == "__main__":
    main()
