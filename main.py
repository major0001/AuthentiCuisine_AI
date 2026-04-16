"""
AuthentiCuisine AI - Main Processing Pipeline (Research-Level)

Coordinates preprocessing, feature extraction, ML prediction,
and explainability.
"""

from utils.preprocess import clean_text
from models.sentiment import SentimentAnalyzer
from models.authenticity import AuthenticityDetector
from models.credibility import CredibilityScorer
from utils.explain import generate_explanation


class AuthentiCuisinePipeline:
    """
    End-to-end AI pipeline for evaluating restaurant review authenticity.
    """

    def __init__(self):
        # Initialize all system components
        self.sentiment = SentimentAnalyzer()
        self.authenticity = AuthenticityDetector()
        self.credibility = CredibilityScorer()

    def analyze_review(self, review: str) -> dict:
        """
        Run full pipeline on input review.

        Args:
            review (str): Raw user review

        Returns:
            dict: Structured analysis output
        """

        # -------------------------------
        # Step 1: Preprocessing
        # -------------------------------
        cleaned = clean_text(review)

        # -------------------------------
        # Step 2: Sentiment Analysis
        # -------------------------------
        sentiment_result = self.sentiment.analyze(cleaned)

        # -------------------------------
        # Step 3: ML Authenticity Prediction
        # -------------------------------
        authenticity_score = self.authenticity.evaluate(cleaned)

        # -------------------------------
        # Step 4: Credibility Scoring
        # -------------------------------
        credibility_score = self.credibility.evaluate(
            sentiment_result["score"], cleaned
        )

        # -------------------------------
        # Step 5: Final Weighted Score
        # -------------------------------
        final_score = (
            0.35 * sentiment_result["score"] +
            0.35 * authenticity_score +
            0.30 * credibility_score
        )

        final_score = round(final_score, 3)

        # -------------------------------
        # Step 6: Explainability
        # -------------------------------
        explanation = generate_explanation(cleaned, {
            "final_score": final_score
        })

        # -------------------------------
        # Final Output
        # -------------------------------
        return {
            "cleaned_text": cleaned,
            "sentiment": sentiment_result,
            "authenticity": authenticity_score,
            "credibility": credibility_score,
            "final_score": final_score,
            "explanation": explanation
        }