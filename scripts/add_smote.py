"""
SMOTE Implementation for AMES Mutagenicity Prediction
This script demonstrates how to implement SMOTE for handling class imbalance
"""

# Set the required environment variable for array API compatibility
import os

os.environ["SCIPY_ARRAY_API"] = "1"

# Import required libraries
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_class_distribution(y, title):
    """Plot the class distribution of a dataset"""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class (0: Non-mutagenic, 1: Mutagenic)")
    plt.ylabel("Count")

    # Add count labels on top of bars
    for i, count in enumerate(Counter(y).values()):
        plt.text(i, count + 5, str(count), ha="center")

    plt.tight_layout()
    return plt


def apply_smote(
    X_train, y_train, random_state=42, sampling_strategy="auto", k_neighbors=5
):
    """
    Apply SMOTE to the training data

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    random_state : int
        Random seed for reproducibility
    sampling_strategy : str or float
        If 'auto', the minority class is resampled to match the majority class.
        If float, it specifies the ratio of the number of samples in the minority class
        over the number of samples in the majority class after resampling.
    k_neighbors : int
        Number of nearest neighbors to use for SMOTE

    Returns:
    --------
    X_train_resampled : array-like
        Resampled training features
    y_train_resampled : array-like
        Resampled training labels
    """
    # Check for NaN values
    nan_count = np.isnan(X_train).sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values in the data. Handling missing values...")
        # If X_train is a numpy array, convert to DataFrame for easier handling
        if isinstance(X_train, np.ndarray):
            import pandas as pd

            X_train_df = pd.DataFrame(X_train)
        else:
            X_train_df = X_train.copy()

        # Simple imputation strategy: fill NaN with mean values
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="mean")
        if isinstance(X_train, np.ndarray):
            X_train_clean = imputer.fit_transform(X_train_df)
        else:
            X_train_clean = pd.DataFrame(
                imputer.fit_transform(X_train_df), columns=X_train_df.columns
            )
        print("Missing values have been imputed with mean values.")
    else:
        X_train_clean = X_train

    # Print original class distribution
    print("Original class distribution:")
    print(Counter(y_train))

    # Apply SMOTE
    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_clean, y_train)

    # Print new class distribution
    print("\nClass distribution after SMOTE:")
    print(Counter(y_train_resampled))

    return X_train_resampled, y_train_resampled


# Example usage (to be copied to your notebook)
"""
# Visualize class distribution before SMOTE
plt_before = plot_class_distribution(y_train, 'Class Distribution Before SMOTE')
plt_before.savefig('../figures/class_distribution_before_smote.png')
plt_before.show()

# Apply SMOTE to balance the training data
X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

# Visualize class distribution after SMOTE
plt_after = plot_class_distribution(y_train_smote, 'Class Distribution After SMOTE')
plt_after.savefig('../figures/class_distribution_after_smote.png')
plt_after.show()

# Train models with SMOTE-resampled data
print("\nTraining Random Forest with SMOTE-resampled data...")
rf_model_smote = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model_smote.fit(X_train_smote, y_train_smote)

# Evaluate on validation set
y_pred_rf_valid_smote = rf_model_smote.predict(X_valid_scaled)
print("\nRandom Forest with SMOTE Results on Validation Set:")
print("Accuracy:", accuracy_score(y_valid, y_pred_rf_valid_smote))
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_rf_valid_smote))

# Evaluate on test set
y_pred_rf_test_smote = rf_model_smote.predict(X_test_scaled)
print("\nRandom Forest with SMOTE Results on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf_test_smote))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf_test_smote))

# Train XGBoost with SMOTE-resampled data
print("\nTraining XGBoost with SMOTE-resampled data...")
xgb_model_smote = xgb.XGBClassifier(random_state=42)
xgb_model_smote.fit(X_train_smote, y_train_smote)

# Evaluate on validation set
y_pred_xgb_valid_smote = xgb_model_smote.predict(X_valid_scaled)
print("\nXGBoost with SMOTE Results on Validation Set:")
print("Accuracy:", accuracy_score(y_valid, y_pred_xgb_valid_smote))
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_xgb_valid_smote))

# Evaluate on test set
y_pred_xgb_test_smote = xgb_model_smote.predict(X_test_scaled)
print("\nXGBoost with SMOTE Results on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb_test_smote))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb_test_smote))

# Compare ROC curves with and without SMOTE
plt.figure(figsize=(10, 8))

# Original Random Forest
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')

# SMOTE Random Forest
y_pred_proba_rf_smote = rf_model_smote.predict_proba(X_test_scaled)[:, 1]
fpr_rf_smote, tpr_rf_smote, _ = roc_curve(y_test, y_pred_proba_rf_smote)
roc_auc_rf_smote = auc(fpr_rf_smote, tpr_rf_smote)
plt.plot(fpr_rf_smote, tpr_rf_smote, label=f'Random Forest with SMOTE (AUC = {roc_auc_rf_smote:.3f})')

# Original XGBoost
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')

# SMOTE XGBoost
y_pred_proba_xgb_smote = xgb_model_smote.predict_proba(X_test_scaled)[:, 1]
fpr_xgb_smote, tpr_xgb_smote, _ = roc_curve(y_test, y_pred_proba_xgb_smote)
roc_auc_xgb_smote = auc(fpr_xgb_smote, tpr_xgb_smote)
plt.plot(fpr_xgb_smote, tpr_xgb_smote, label=f'XGBoost with SMOTE (AUC = {roc_auc_xgb_smote:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - With and Without SMOTE')
plt.legend(loc='lower right')
plt.savefig('../figures/roc_curves_with_smote.png')
plt.show()
"""
