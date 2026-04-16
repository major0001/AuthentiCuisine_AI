"""
AuthentiCuisine AI - Explainability Module (Research-Level)

Provides feature-based explanations for authenticity predictions.
"""

from models.feature_extraction import extract_features


def generate_explanation(text: str, result: dict) -> dict:
    """
    Generate structured explanation for prediction.

    Args:
        text (str): Cleaned review text
        result (dict): Output from pipeline

    Returns:
        dict: Explanation breakdown
    """

    features = extract_features(text)

    sentiment = features[0]
    length = features[1]
    diversity = features[2]
    exaggeration = features[3]
    caps_ratio = features[4]

    explanation = []

    # Sentiment reasoning
    if sentiment > 0.5:
        explanation.append("Strong positive sentiment detected")
    elif sentiment < -0.5:
        explanation.append("Strong negative sentiment detected")

    # Length reasoning
    if length < 6:
        explanation.append("Very short review reduces authenticity confidence")
    elif length > 20:
        explanation.append("Detailed review increases credibility")

    # Lexical diversity
    if diversity < 0.5:
        explanation.append("Low word diversity suggests repetition or spam-like content")

    # Exaggeration
    if exaggeration > 1:
        explanation.append("Multiple exaggeration keywords detected (e.g., 'best', 'amazing')")

    # Caps usage
    if caps_ratio > 0.3:
        explanation.append("Excessive capital letters detected, indicating emotional bias")

    # Final decision reasoning
    if result["final_score"] > 0.7:
        decision = "The review is likely authentic."
    elif result["final_score"] > 0.4:
        decision = "The review shows mixed authenticity signals."
    else:
        decision = "The review is likely inauthentic or manipulated."

    return {
        "factors": explanation,
        "decision": decision
    }