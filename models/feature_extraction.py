"""
AuthentiCuisine AI - Feature Extraction Module

Transforms raw text into numerical features for ML model input.
"""

from textblob import TextBlob

EXAGGERATION_WORDS = ["best", "amazing", "perfect", "worst", "terrible"]


def extract_features(text: str) -> list:
    """
    Extract numerical features from review text.

    Features:
    - Sentiment polarity
    - Review length
    - Lexical diversity
    - Exaggeration usage
    - Capitalization ratio

    Returns:
        list: Feature vector
    """

    words = text.split()

    # Sentiment polarity (-1 to 1)
    sentiment = TextBlob(text).sentiment.polarity

    # Length
    length = len(words)

    # Lexical diversity
    unique_ratio = len(set(words)) / max(len(words), 1)

    # Exaggeration detection
    exaggeration = sum(w in text.lower() for w in EXAGGERATION_WORDS)

    # ALL CAPS detection
    caps_ratio = sum(1 for w in words if w.isupper()) / max(len(words), 1)

    return [
        sentiment,
        length,
        unique_ratio,
        exaggeration,
        caps_ratio
    ]