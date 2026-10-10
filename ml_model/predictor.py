"""
Epidemic Outbreak Early Warning Radar
Predictor Module - Uses trained Random Forest model for inference
"""

import os
import re
import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

# Paths to saved model artifacts
_BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_BASE, "saved_model", "rf_outbreak_model.joblib")
SCALER_PATH = os.path.join(_BASE, "saved_model", "scaler.joblib")
FEATURE_COLS_PATH = os.path.join(_BASE, "saved_model", "feature_columns.joblib")

OUTBREAK_THRESHOLD = 6

KEYWORD_WEIGHTS = {
    "fever": 3, "high temperature": 3, "chills": 2, "cough": 3,
    "dry cough": 3, "coughing": 2, "sore throat": 2, "throat pain": 2,
    "headache": 2, "body ache": 2, "fatigue": 2, "tiredness": 1,
    "shortness of breath": 3, "difficulty breathing": 3, "rash": 2,
    "rashes": 2, "vomiting": 2, "nausea": 2, "diarrhea": 2,
    "loss of taste": 3, "loss of smell": 3, "runny nose": 2,
    "sneezing": 1, "viral": 2, "infection": 2, "flu": 3,
    "influenza": 3, "dengue": 4, "malaria": 4, "covid": 4,
    "corona": 4, "epidemic": 4, "outbreak": 4, "pandemic": 4,
    "hospitalized": 3, "quarantine": 3,
}


def load_model() -> Tuple[Any, Any, List[str]]:
    """Load the trained Random Forest model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found! Please run 'ml_model/train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features_from_text(text: str, post_count: int = 50, unique_locations: int = 3) -> Tuple[Dict[str, Any], int]:
    """Convert raw text to the feature vector expected by the model."""
    cleaned = clean_text(text)
    row: Dict[str, Any] = {}
    scores_list: List[int] = []

    for kw, weight in KEYWORD_WEIGHTS.items():
        count = int(cleaned.count(kw))
        row[kw] = count
        scores_list.append(count * int(weight))

    symptom_score: int = sum(scores_list)
    row["symptom_score"] = symptom_score
    row["post_count_last_hour"] = post_count
    row["unique_locations"] = unique_locations
    now = datetime.now()
    row["hour_of_day"] = now.hour
    row["day_of_week"] = now.weekday()
    return row, symptom_score


def predict_outbreak(text: str, post_count: int = 50, unique_locations: int = 3) -> Dict[str, Any]:
    """
    Run outbreak prediction on a given text.
    Returns:
        - prediction (0 or 1)
        - probability of outbreak (0-1)
        - symptom_score
        - alert message
    """
    model, scaler, feature_cols = load_model()

    features, symptom_score = extract_features_from_text(text, post_count, unique_locations)
    feature_df = pd.DataFrame([features])[feature_cols]
    feature_scaled = scaler.transform(feature_df)

    prediction = model.predict(feature_scaled)[0]
    proba = model.predict_proba(feature_scaled)[0][1]  # probability of outbreak

    alert = None
    if prediction == 1 or symptom_score >= OUTBREAK_THRESHOLD:
        alert = "🚨 ALERT: Possible Outbreak Detected!"

    return {
        "prediction": int(prediction),
        "outbreak_probability": float("{:.4f}".format(float(proba))),
        "symptom_score": symptom_score,
        "alert": alert,
        "risk_level": get_risk_level(proba, symptom_score),
        "timestamp": datetime.now().isoformat(),
    }


def get_risk_level(probability: float, symptom_score: int) -> str:
    """Map probability and symptom score to a human-readable risk level."""
    combined = (float(probability) + min(float(symptom_score) / 20.0, 1.0)) / 2.0
    if combined >= 0.7:
        return "CRITICAL"
    elif combined >= 0.5:
        return "HIGH"
    elif combined >= 0.3:
        return "MODERATE"
    else:
        return "LOW"


def batch_predict(texts_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Predict outbreak probability for a batch of texts."""
    results: List[Dict[str, Any]] = []
    for item in texts_list:
        text = str(item.get("text", ""))
        result = predict_outbreak(
            text,
            post_count=int(item.get("post_count", 50)),
            unique_locations=int(item.get("unique_locations", 3)),
        )
        if len(text) > 100:
            truncated_text: str = "".join([text[i] for i in range(100)])
            result["text"] = truncated_text + "..."
        else:
            result["text"] = text
        result["source"] = item.get("source", "Unknown")
        result["location"] = item.get("location", "Unknown")
        results.append(result)
    return results


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Many people in Chennai are reporting fever, cough and dengue symptoms.",
        "It's a beautiful sunny day. Feeling great!",
        "Hospitals overwhelmed with influenza and COVID patients. Outbreak suspected!",
    ]
    print("Testing Predictor Module\n" + "=" * 50)
    for t_val in test_texts:
        text_str = str(t_val)
        res_data = predict_outbreak(text_str)
        outbreak_prob = float(res_data.get('outbreak_probability', 0))
        msg_prefix: str = "".join([text_str[i] for i in range(min(len(text_str), 70))])
        print(f"\nText: {msg_prefix}...")
        print(f"  Risk Level: {res_data.get('risk_level', 'LOW')}")
        print(f"  Outbreak Prob: {outbreak_prob * 100.0:.1f}%")
        print(f"  Symptom Score: {res_data.get('symptom_score', 0)}")
        if res_data.get("alert"):
            print(f"  {res_data.get('alert')}")
