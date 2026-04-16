"""
AuthentiCuisine AI - Batch Testing Script

Runs multiple reviews through the system for evaluation.
"""

import pandas as pd
from main import AuthentiCuisinePipeline

pipeline = AuthentiCuisinePipeline()

df = pd.read_csv("data/sample_reviews.csv")

results = []

for review in df["review"]:
    result = pipeline.analyze_review(review)
    results.append({
        "review": review,
        "sentiment": result["sentiment"]["label"],
        "authenticity": result["authenticity"],
        "credibility": result["credibility"]
    })

output_df = pd.DataFrame(results)

print(output_df)
output_df.to_csv("data/results.csv", index=False)