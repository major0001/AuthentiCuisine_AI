"""
AuthentiCuisine AI - Configuration File

This file stores all configurable parameters used across the system.
Keeping these centralized ensures consistency and easy adjustments.
"""

# -------------------------------
# MODEL SETTINGS
# -------------------------------

# Pretrained sentiment model (HuggingFace)
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Thresholds for classification
AUTHENTICITY_THRESHOLD = 0.6
CREDIBILITY_THRESHOLD = 0.5

# -------------------------------
# SYSTEM SETTINGS
# -------------------------------

MAX_TEXT_LENGTH = 512  # Max tokens for transformer models

# -------------------------------
# UI SETTINGS
# -------------------------------

APP_TITLE = "AuthentiCuisine AI"
TAGLINE = "Detecting Truth in Every Bite"