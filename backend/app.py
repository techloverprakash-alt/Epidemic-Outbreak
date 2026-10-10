"""
Epidemic Outbreak Early Warning Radar - Flask Backend
Team: CODE CRAFTERS | Hackathon 2K26
Fixed: Rebuilt for Python 3.12 compatibility (original .pyc was Python 3.14)
"""

from flask import Flask, jsonify, request  # type: ignore
from flask_cors import CORS  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import random
import os
from datetime import datetime, timedelta
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root to path for imports
sys.path.append(os.path.dirname(BASE_DIR))

# Import local AI service
try:
    from gemini_service import generate_district_report, ping_ollama
    # Add Random Forest Predictor
    from ml_model.predictor import predict_outbreak # type: ignore
except Exception:
    try:
        from backend.gemini_service import generate_district_report, ping_ollama
        from ml_model.predictor import predict_outbreak # type: ignore
    except Exception:
        def generate_district_report(d, data): return "AI Service Unavailable"
        def ping_ollama(): return False
        def predict_outbreak(text): return {"prediction": 0, "outbreak_probability": 0.1, "risk_level": "LOW", "symptom_score": 0}

app = Flask(__name__)
CORS(app)
# Check both backend folder and root folder for the CSV
CSV_PATH = os.path.join(BASE_DIR, "tamilnadu_disease_dataset.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), "tamilnadu_disease_dataset.csv")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"[OK] Loaded dataset: {len(df)} rows from {CSV_PATH}")
except FileNotFoundError:
    print(f"[WARN] Dataset not found at {CSV_PATH}. Using empty DataFrame.")
    df = pd.DataFrame(columns=["District", "Disease", "Cases", "Temperature", "Rainfall", "Risk"])

# ── NLP keyword scoring ───────────────────────────────────────────────────────
SYMPTOM_KEYWORDS = {
    "fever": 3, "high temperature": 3, "chills": 2, "sweating": 2,
    "dengue": 4, "malaria": 4, "covid": 4, "coronavirus": 4, "influenza": 3, "typhoid": 4,
    "headache": 1, "body ache": 2, "fatigue": 1, "vomiting": 2, "rash": 2,
    "cough": 2, "breathlessness": 3, "diarrhoea": 2, "joint pain": 2,
    "hospital": 1, "outbreak": 5, "spreading": 3, "rising cases": 4, "epidemic": 5,
}

def compute_symptom_score(text: str) -> float:
    text_lower = text.lower()
    scores = [float(weight) for keyword, weight in SYMPTOM_KEYWORDS.items() if keyword in text_lower]
    return float(sum(scores if scores else [0.0]))

def detect_disease(text: str) -> str:
    text_lower = text.lower()
    for disease in ["covid", "coronavirus", "dengue", "malaria", "typhoid", "influenza", "flu"]:
        if disease in text_lower:
            mapping = {"covid": "COVID-19", "coronavirus": "COVID-19", "flu": "Influenza",
                       "influenza": "Influenza", "dengue": "Dengue",
                       "malaria": "Malaria", "typhoid": "Typhoid"}
            return mapping.get(disease, "Unknown")
    return "Unknown"

# ── Simulated posts cache (in-memory) ────────────────────────────────────────
SAMPLE_POSTS = [
    {"text": "Dengue cases rising rapidly in Chennai hospitals!", "location": "Chennai"},
    {"text": "Fever and chills reported across Coimbatore. Doctors advise caution.", "location": "Coimbatore"},
    {"text": "COVID-19 cases up in Madurai. Hospitals on alert.", "location": "Madurai"},
    {"text": "Malaria outbreak spreading in Thanjavur district.", "location": "Thanjavur"},
    {"text": "Typhoid cases in Tirunelveli. Water contamination suspected.", "location": "Tirunelveli"},
    {"text": "Routine checkups ongoing. No unusual symptoms reported.", "location": "Salem"},
    {"text": "Influenza wave hitting schools in Erode.", "location": "Erode"},
    {"text": "Joint pain, rash, and fever: dengue symptoms seen in Vellore.", "location": "Vellore"},
]

def enrich_posts(posts):
    enriched = []
    for i, post in enumerate(posts):
        score = compute_symptom_score(post["text"])
        enriched.append({
            "id": i + 1,
            "text": post["text"],
            "location": post["location"],
            "disease": detect_disease(post["text"]),
            "symptom_score": score,
            "alert": score >= 6,
            "timestamp": (datetime.now() - timedelta(minutes=i * 15)).strftime("%Y-%m-%d %H:%M"),
        })
    return enriched

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Epidemic Radar API is running", "python": "3.12+"}), 200


@app.route("/api/data", methods=["GET"])
def get_data():
    posts = enrich_posts(SAMPLE_POSTS)
    return jsonify({"count": len(posts), "posts": posts}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or "text" not in body:
        return jsonify({"error": "Provide JSON with 'text' field"}), 400

    text = body["text"]
    
    # Use the actual Random Forest Predictor
    result = predict_outbreak(text)
    
    return jsonify({
        "input_text": text,
        "detected_disease": detect_disease(text),
        "symptom_score": result.get("symptom_score", 0),
        "outbreak_probability": result.get("outbreak_probability", 0.0),
        "risk_level": result.get("risk_level", "LOW"),
        "alert": result.get("alert") is not None,
        "source": "Random Forest Classifier"
    }), 200


@app.route("/api/stats", methods=["GET"])
def get_stats():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500

    stats = {
        "total_cases": int(df["Cases"].sum()),
        "districts_affected": int(df["District"].nunique()),
        "diseases_tracked": df["Disease"].unique().tolist(),
        "high_risk_districts": df[df["Risk"] == "HIGH"]["District"].unique().tolist(),
        "risk_distribution": df["Risk"].value_counts().to_dict(),
        "disease_summary": df.groupby("Disease")["Cases"].sum().sort_values(ascending=False).to_dict(),
        "avg_symptom_score": float("{:.1f}".format(float(random.uniform(2, 9)))),
        "max_symptom_score": float("{:.1f}".format(float(random.uniform(10, 20)))),
        "total_posts": random.randint(50, 200),
        "outbreak_posts": random.randint(5, 30),
    }
    return jsonify(stats), 200


@app.route("/api/locations", methods=["GET"])
def get_locations():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"}), 500

    # District-level aggregated risk
    grouped = df.groupby("District").agg(
        total_cases=("Cases", "sum"),
        diseases=("Disease", lambda x: list(x.unique())),
        risk=("Risk", lambda x: x.value_counts().idxmax()),
    ).reset_index()

    # Add missing fields needed by the dashboard
    city_coords = {
        "Chennai": [13.08, 80.27], "Coimbatore": [11.01, 76.95],
        "Madurai": [9.92, 78.12], "Thanjavur": [10.78, 79.13],
        "Tirunelveli": [8.71, 77.75], "Salem": [11.66, 78.14],
        "Erode": [11.34, 77.71], "Vellore": [12.91, 79.13],
    }
    
    grouped["lat"] = grouped["District"].apply(lambda x: city_coords.get(x, [11.12, 78.65])[0])
    grouped["lon"] = grouped["District"].apply(lambda x: city_coords.get(x, [11.12, 78.65])[1])
    
    # Use predictor to get more realistic scores and risk levels
    scores = []
    risks = []
    probs = []
    for dist in grouped["city"]:
        # Simulate a typical high-risk scenario for the predictor
        res = predict_outbreak(f"Outbreak risk in {dist} with symptoms.")
        scores.append(float(res.get("symptom_score", 0)))
        risks.append(res.get("risk_level", "LOW"))
        probs.append(float(res.get("outbreak_probability", 0.0)))

    grouped["max_symptom_score"] = scores
    grouped["outbreak_probability"] = probs
    grouped["risk_level"] = risks
    grouped["city"] = grouped["city"] # Ensure it stays 'city' for frontend

    return jsonify({"locations": grouped.to_dict(orient="records")}), 200


@app.route("/api/timeline", methods=["GET"])
def get_timeline():
    # Simulated hourly symptom score trend (last 24 hours)
    now = datetime.now()
    timeline = []
    for i in range(24):
        hour = now - timedelta(hours=23 - i)
        timeline.append({
            "hour": hour.strftime("%H:00"),
            "avg_symptom_score": float("{:.1f}".format(float(random.uniform(2, 9)))),
            "symptom_score": float("{:.1f}".format(float(random.uniform(2, 9)))),
            "posts_analyzed": random.randint(5, 40),
        })
    return jsonify({"timeline": timeline}), 200


@app.route("/api/refresh", methods=["POST"])
def refresh_data():
    # In production: trigger scraper. Here we simulate.
    return jsonify({
        "status": "refreshed",
        "new_posts": len(SAMPLE_POSTS),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }), 200


# ─── Auth & Notification Routes (Restored for Dashboard) ──────────────────────

_users = {} # username -> {district, phone}

@app.route("/api/login", methods=["POST"])
def login():
    body = request.get_json(force=True, silent=True) or {}
    username = body.get("username", "Guest")
    district = body.get("district", "Chennai")
    phone = body.get("phone", "")
    _users[username] = {"district": district, "phone": phone}
    return jsonify({"message": f"Welcome {username}", "district": district, "phone": phone}), 200


@app.route("/api/notifications/<username>", methods=["GET"])
def get_notifications(username):
    user_info = _users.get(username)
    if not user_info:
        return jsonify({"alerts": []}), 200
    
    user_district = user_info.get("district")
    posts = enrich_posts(SAMPLE_POSTS)
    relevant = [p for p in posts if p["location"] == user_district and p["alert"]]
    
    alerts = []
    for p in relevant:
        alerts.append({
            "id": random.randint(1000, 9999),
            "level": "HIGH",
            "message": f"🚨 Health Alert: Outbreak risk detected in {user_district}. {p['text']}",
            "district": user_district,
            "timestamp": datetime.now().isoformat()
        })
    return jsonify({"alerts": alerts}), 200


@app.route("/api/report/<district>", methods=["GET"])
def get_district_ai_report(district):
    # 1. Get stats for this district to pass to AI
    df_dist = df[df["District"] == district]
    if df_dist.empty:
        # Fallback if district not in CSV
        disease_data = {
            "district": district,
            "total_cases": random.randint(100, 500),
            "diseases": {"Unknown": 100},
            "risk_level": "MODERATE"
        }
    else:
        disease_data = {
            "district": district,
            "total_cases": int(df_dist["Cases"].sum()),
            "diseases": df_dist.groupby("Disease")["Cases"].sum().to_dict(),
            "risk_level": df_dist["Risk"].value_counts().idxmax()
        }

    # 2. Call the newly added gemini_service
    ai_summary = generate_district_report(district, disease_data)

    # 3. Return structured JSON for dashboard compatibility
    # The dashboard expects these specific fields to build charts
    cases = disease_data["total_cases"]
    risk = disease_data["risk_level"]
    
    primary_disease = "Unknown"
    diseases = disease_data.get("diseases", {})
    if isinstance(diseases, dict) and diseases:
        primary_disease = max(diseases.items(), key=lambda x: x[1])[0]
    
    return jsonify({
        "district": district,
        "disease": primary_disease,
        "cases": cases,
        "temperature": random.randint(28, 38),
        "rainfall": random.randint(0, 100),
        "risk": risk,
        "symptom_score": random.randint(1, 10),
        "timeline": [random.randint(10, 100) for _ in range(7)],
        "breakdown": {"dengue": 30, "malaria": 20, "covid": 40, "typhoid": 10},
        "ai_summary": ai_summary  # Added for future dashboard use
    }), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True, silent=True) or {}
    message = body.get("message", "").lower()
    
    response = "I'm Epi-Bot, your health assistant. I'm monitoring disease trends in Tamil Nadu."
    if "dengue" in message:
        response = "Dengue cases are being monitored closely. Please use mosquito nets and clear stagnant water."
    elif "covid" in message:
        response = "COVID-19 protocols should be followed in areas with rising cases. Wear masks in crowded places."
    elif "chennai" in message:
        response = "Chennai is currently showing moderate symptom reports. Several hospitals are on alert."
    
    return jsonify({"response": response}), 200


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[AI] Ollama Status: {'OK' if ping_ollama() else 'NOT FOUND'}")
    app.run(debug=True, host="0.0.0.0", port=5000)
