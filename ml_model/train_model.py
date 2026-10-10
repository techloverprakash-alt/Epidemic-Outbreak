"""
Epidemic Outbreak Early Warning Radar
ML Model Training Script - Random Forest Classifier
Trains a model to predict potential epidemic outbreaks based on symptom keyword frequencies.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import random
import joblib  # type: ignore
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from typing import Any, Dict, List, Tuple

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "rf_outbreak_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

SYMPTOM_KEYWORDS = [
    "fever", "high temperature", "chills", "cough", "dry cough", "coughing",
    "sore throat", "throat pain", "headache", "body ache", "fatigue",
    "tiredness", "shortness of breath", "difficulty breathing", "rash",
    "rashes", "vomiting", "nausea", "diarrhea", "loss of taste",
    "loss of smell", "runny nose", "sneezing", "viral", "infection",
    "flu", "influenza", "dengue", "malaria", "covid", "corona",
    "epidemic", "outbreak", "pandemic", "hospitalized", "quarantine",
]

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

OUTBREAK_THRESHOLD = 40  # Secondary guard (rarely triggered; is_outbreak flag is primary)


# ─── Dataset Generation ────────────────────────────────────────────────────────

def generate_training_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate a synthetic training dataset simulating health post feature vectors.
    Each sample has symptom keyword counts, aggregate features, and label.
    """
    records: List[Dict[str, Any]] = []
    for i in range(n_samples):
        # Randomly decide if this is an outbreak scenario (30% of time)
        is_outbreak: bool = random.random() < 0.30

        row: Dict[str, Any] = {}
        # Generate keyword frequencies
        scores_list: List[int] = []
        for kw in SYMPTOM_KEYWORDS:
            weight = int(KEYWORD_WEIGHTS.get(kw, 1))
            if is_outbreak:
                # Higher chance of keyword appearing
                count = int(np.random.poisson(lam=max(0.5, float(weight) * 0.8)))
            else:
                count = int(np.random.poisson(lam=max(0.05, float(weight) * 0.1)))
            row[kw] = count
            scores_list.append(count * weight)

        symptom_score: int = sum(scores_list)
        row["symptom_score"] = symptom_score
        row["post_count_last_hour"] = random.randint(50, 500) if is_outbreak else random.randint(1, 60)
        row["unique_locations"] = random.randint(3, 15) if is_outbreak else random.randint(1, 5)
        row["hour_of_day"] = random.randint(0, 23)
        row["day_of_week"] = random.randint(0, 6)

        # Label: 1 = outbreak, 0 = normal
        # Label is driven by is_outbreak flag (symptom counts are input features, not ground truth)
        row["label"] = 1 if (is_outbreak or symptom_score >= OUTBREAK_THRESHOLD) else 0
        records.append(row)

    return pd.DataFrame(records)


# ─── Model Training ────────────────────────────────────────────────────────────

def train_model() -> Tuple[Any, Any, List[str]]:
    print("=" * 60)
    print("  Epidemic Outbreak - Random Forest Model Training")
    print("=" * 60)

    # 1. Generate Dataset
    print("\n[1] Generating synthetic training dataset...")
    df: pd.DataFrame = generate_training_dataset(n_samples=3000)
    print(f"    >> Dataset shape: {df.shape}")
    label_list: List[int] = [int(v) for v in df['label'].tolist()]
    outbreak_sum: int = sum(label_list)
    print(f"    >> Outbreak samples: {outbreak_sum} ({outbreak_sum/len(label_list)*100.0:.1f}%)")
    print(f"    >> Normal samples:   {len(label_list) - outbreak_sum}")

    # Save dataset for reference
    dataset_path: str = os.path.join(MODEL_DIR, "training_data.csv")
    df.to_csv(dataset_path, index=False)
    print(f"    >> Dataset saved to: {dataset_path}")

    # 2. Features & Labels
    feature_cols: List[str] = [c for c in df.columns if c != "label"]
    X: pd.DataFrame = df[feature_cols]
    y: pd.Series = df["label"]

    # 3. Train/Test Split
    print("\n[2] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Scaling
    scaler: StandardScaler = StandardScaler()
    X_train_sc: np.ndarray = scaler.fit_transform(X_train)
    X_test_sc: np.ndarray = scaler.transform(X_test)

    # 5. Random Forest Training
    print("[3] Training Random Forest Classifier...")
    rf: RandomForestClassifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)
    print("    >> Model training complete!")

    # 6. Evaluation
    print("\n[4] Evaluating model on test set...")
    y_pred: np.ndarray = rf.predict(X_test_sc)
    acc: float = accuracy_score(y_test, y_pred)

    print(f"\n    [OK] Accuracy: {acc*100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Outbreak"]))
    print("--- Confusion Matrix ---")
    cm: np.ndarray = confusion_matrix(y_test, y_pred)
    print(f"    True Negatives:  {cm[0][0]}")
    print(f"    False Positives: {cm[0][1]}")
    print(f"    False Negatives: {cm[1][0]}")
    print(f"    True Positives:  {cm[1][1]}")

    # 7. Feature Importance
    print("\n[5] Top 10 Most Important Features:")
    importances: pd.Series = pd.Series(rf.feature_importances_, index=feature_cols)
    top10: pd.Series = importances.sort_values(ascending=False).head(10)
    for feat, imp in top10.items():
        print(f"    {feat:<30} {imp:.4f}")

    # 8. Save model artifacts
    print("\n[6] Saving model artifacts...")
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    print(f"    >> Model saved:   {MODEL_PATH}")
    print(f"    >> Scaler saved:  {SCALER_PATH}")
    print(f"    >> Features saved:{FEATURE_COLS_PATH}")

    print("\n[DONE] Training pipeline complete!")
    return rf, scaler, feature_cols


if __name__ == "__main__":
    train_model()
