"""
AuthentiCuisine AI - Streamlit Interface (Research-Level)

Interactive UI for evaluating restaurant review authenticity.
"""
import os
import warnings

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Suppress transformers logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress HuggingFace tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import streamlit as st
from main import AuthentiCuisinePipeline

from transformers.utils import logging
logging.set_verbosity_error()
# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AuthentiCuisine AI",
    layout="centered"
)

# -------------------------------
# App Header
# -------------------------------
st.title("🍽️ AuthentiCuisine AI")
st.subheader("Detecting Truth in Every Bite")

st.markdown(
    "An explainable AI system for evaluating restaurant review authenticity "
    "using sentiment, linguistic features, and machine learning."
)

# -------------------------------
# Initialize Pipeline (cached)
# -------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = AuthentiCuisinePipeline()

pipeline = st.session_state.pipeline

# -------------------------------
# User Input
# -------------------------------
review = st.text_area(
    "Enter a restaurant review:",
    height=150
)

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("Analyze Review"):

    if not review.strip():
        st.warning("Please enter a review.")
    else:
        result = pipeline.analyze_review(review)

        # -------------------------------
        # Results Section
        # -------------------------------
        st.write("## 📊 Analysis Results")

        # Final Score
        st.metric("Final Authenticity Score", result["final_score"])

        # Decision Label
        if result["final_score"] > 0.7:
            st.success("Highly Authentic Review")
        elif result["final_score"] > 0.4:
            st.warning("Moderately Authentic Review")
        else:
            st.error("Low Authenticity - Possibly Fake")

        # -------------------------------
        # Breakdown
        # -------------------------------
        st.write("### 🔍 Score Breakdown")

        st.write(f"**Sentiment Score:** {result['sentiment']['score']:.2f}")
        st.write(f"**Authenticity (ML):** {result['authenticity']}")
        st.write(f"**Credibility:** {result['credibility']}")

        # -------------------------------
        # Explanation Section
        # -------------------------------
        st.write("### 🧠 Explanation")

        for factor in result["explanation"]["factors"]:
            st.write(f"- {factor}")

        st.info(result["explanation"]["decision"])

        # -------------------------------
        # Visualization
        # -------------------------------
        import pandas as pd

        chart_data = pd.DataFrame({
            "Metric": ["Authenticity", "Credibility"],
            "Score": [
                result["authenticity"],
                result["credibility"]
            ]
        })

        st.write("### 📈 Score Visualization")
        st.bar_chart(chart_data.set_index("Metric"))