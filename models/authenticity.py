"""
AuthentiCuisine AI - Authenticity Detection (ML-based)

Uses trained ML model instead of rule-based scoring.
"""

import numpy as np
from models.feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier


class AuthenticityDetector:
    """
    ML-based authenticity predictor.
    """

    def __init__(self):
        # Placeholder model (you should load trained model ideally)
        self.model = RandomForestClassifier(n_estimators=100)

        # Dummy training (for demo purposes only)
        X_dummy = [
            [0.8, 20, 0.7, 0, 0],
            [-0.9, 5, 0.4, 3, 0.2]
        ]
        y_dummy = [1, 0]

        self.model.fit(X_dummy, y_dummy)

    def evaluate(self, text: str) -> float:
        """
        Predict authenticity probability.

        Returns:
            float: Probability (0 to 1)
        """

        features = extract_features(text)
        features = np.array(features).reshape(1, -1)

        prob = self.model.predict_proba(features)[0][1]

        return round(float(prob), 3)