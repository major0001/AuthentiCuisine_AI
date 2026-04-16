"""
AuthentiCuisine AI - Credibility Scoring Module

Estimates how trustworthy a review is.
"""


class CredibilityScorer:
    """
    Combines sentiment confidence and linguistic clarity.
    """

    def evaluate(self, sentiment_score: float, text: str) -> float:
        """
        Compute credibility score.

        Args:
            sentiment_score (float): Confidence from sentiment model
            text (str): Cleaned text

        Returns:
            float: Credibility score (0 to 1)
        """

        length_factor = min(len(text.split()) / 40, 1.0)

        credibility = (0.6 * sentiment_score) + (0.4 * length_factor)

        return round(credibility, 3)