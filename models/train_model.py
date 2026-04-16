"""
AuthentiCuisine AI - Model Training Module

Trains and evaluates authenticity detection model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from models.feature_extraction import extract_features


def train_model(data_path="data/sample_reviews.csv"):
    """
    Train ML model on dataset.

    Args:
        data_path (str): Path to dataset CSV

    Returns:
        trained model
    """

    df = pd.read_csv(data_path)

    # Extract features
    X = [extract_features(text) for text in df["review"]]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    return model