"""
AuthentiCuisine AI - Text Preprocessing Module

Handles cleaning and normalization of input reviews before analysis.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Cleans raw review text by:
    - Lowercasing
    - Removing punctuation
    - Removing extra whitespace

    Args:
        text (str): Raw user input

    Returns:
        str: Cleaned text
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text