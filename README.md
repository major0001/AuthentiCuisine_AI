# AuthentiCuisine AI  
### Detecting Truth in Every Bite

AuthentiCuisine AI is an **Explainable AI system** designed to evaluate the **authenticity of restaurant reviews** using Natural Language Processing (NLP), Machine Learning, and feature-based reasoning.

---

## Overview

Online reviews heavily influence consumer decisions, yet many are **fake, biased, or misleading**.

AuthentiCuisine AI addresses this problem by:

- Analyzing review sentiment
- Extracting linguistic and structural features
- Predicting authenticity using a machine learning model
- Providing **transparent, explainable outputs**

---

## Key Features

- Sentiment Analysis (positive / negative / neutral)
- Machine Learning-based Authenticity Prediction
- Credibility Scoring System
- Feature Engineering (length, diversity, exaggeration, etc.)
- Explainable AI (human-readable reasoning)
- Interactive UI (Streamlit-based)

---

## System Architecture

```

Input Review
↓
Text Preprocessing
↓
Feature Extraction
↓
ML Model (Random Forest)
↓
Scoring System
↓
Explainability Layer
↓
Final Output (Score + Explanation)

```

---

## Technologies Used

| Category            | Tools / Libraries              |
|--------------------|-------------------------------|
| Language           | Python                        |
| Frontend           | Streamlit                     |
| NLP                | Transformers, TextBlob        |
| Machine Learning   | Scikit-learn                  |
| Data Processing    | Pandas, NumPy                 |

---

## Feature Engineering

The system extracts the following features:

- Sentiment polarity
- Review length
- Lexical diversity
- Exaggeration keyword frequency
- Capitalization ratio

These features are used as input to the machine learning model.

---

## Machine Learning Model

- Model: **Random Forest Classifier**
- Input: Feature vectors from text
- Output: Probability score (0–1)

### Why Random Forest?
- Handles mixed feature types
- Robust to noise
- Interpretable

---

## Scoring Mechanism

Final authenticity score is computed using a weighted approach:

```

Final Score =
0.35 × Sentiment +
0.35 × Authenticity +
0.30 × Credibility

```

---

## Explainability

The system provides explanations such as:

- “Short review reduces authenticity confidence”
- “High exaggeration detected”
- “Low lexical diversity suggests repetition”

This ensures transparency and trust in predictions.

---

## User Interface

Built using **Streamlit**, the UI allows users to:

- Input a review
- Analyze authenticity in real-time
- View scores and breakdowns
- See explanation and visualizations

---

## Project Structure

```

authenticuisine_ai/
│
├── app.py                      # Streamlit UI
├── main.py                     # Core pipeline
│
├── models/
│   ├── authenticity.py         # ML model inference
│   ├── sentiment.py            # Sentiment analysis
│   ├── credibility.py          # Credibility scoring
│   ├── feature_extraction.py   # Feature engineering
│   └── train_model.py          # Model training
│
├── utils/
│   ├── preprocess.py           # Text cleaning
│   └── explain.py              # Explainability logic
│
├── data/
│   └── dataset.csv             # Training dataset
│
├── requirements.txt
└── README.md

```



## Installation

### 1. change to the working directory

```bash
cd authenticuisine-ai
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
```

Activate:

* Windows:

```bash
.venv\Scripts\activate
```

* Mac/Linux:

```bash
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## Model Training (Optional)

To train the model:

```bash
python models/train_model.py
```

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

##  Use Cases

* Detect fake restaurant reviews
* Assist customers in decision-making
* Support review moderation systems
* Research in explainable AI systems

---

## Future Improvements

* SHAP-based explainability
* Deep learning models (BERT fine-tuning)
* Real-world large-scale datasets
* API deployment
* Mobile/web integration

---

## Conclusion

AuthentiCuisine AI demonstrates how combining:

* NLP
* Machine Learning
* Feature Engineering
* Explainable AI

can create a **powerful and transparent system** for evaluating review authenticity.


---

## License

This project is for academic and research purposes.

### The end