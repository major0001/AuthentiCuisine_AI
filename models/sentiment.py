"""
AuthentiCuisine AI - Sentiment Analysis Module

Uses a pretrained transformer model to classify sentiment.
"""

from transformers import pipeline
from config import SENTIMENT_MODEL


class SentimentAnalyzer:
    """
    Wrapper class for sentiment analysis using HuggingFace pipeline.
    """

    def __init__(self):
        # Load pretrained sentiment model
        self.model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of input text.

        Args:
            text (str): Cleaned review text

        Returns:
            dict: Sentiment label and confidence
        """

        result = self.model(text)[0]

        return {
            "label": result["label"],
            "score": result["score"]
        }