"""
Epidemic Outbreak Early Warning Radar
Data Collection Module - Scraper & Symptom Keyword Analyzer
Uses BeautifulSoup to scrape (simulated) health news and social media posts
"""

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
import random
import re
import nltk  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from faker import Faker  # type: ignore
from datetime import datetime, timedelta
import pymongo  # type: ignore
import time
from typing import Any, Dict, List, Optional, Union

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

fake = Faker()

# ─── Symptom Keywords ──────────────────────────────────────────────────────────
SYMPTOM_KEYWORDS = {
    "fever": 3,
    "high temperature": 3,
    "chills": 2,
    "cough": 3,
    "dry cough": 3,
    "coughing": 2,
    "sore throat": 2,
    "throat pain": 2,
    "headache": 2,
    "body ache": 2,
    "fatigue": 2,
    "tiredness": 1,
    "shortness of breath": 3,
    "difficulty breathing": 3,
    "rash": 2,
    "rashes": 2,
    "vomiting": 2,
    "nausea": 2,
    "diarrhea": 2,
    "loss of taste": 3,
    "loss of smell": 3,
    "runny nose": 2,
    "sneezing": 1,
    "viral": 2,
    "infection": 2,
    "flu": 3,
    "influenza": 3,
    "dengue": 4,
    "malaria": 4,
    "covid": 4,
    "corona": 4,
    "epidemic": 4,
    "outbreak": 4,
    "pandemic": 4,
    "hospitalized": 3,
    "quarantine": 3,
}

DISEASE_TAGS = ["COVID-19", "Dengue", "Influenza", "Malaria", "Unknown"]

# ─── Simulated Social Media Post Templates ─────────────────────────────────────
POST_TEMPLATES = [
    "Feeling terrible today – {symptom1} and {symptom2} since yesterday.",
    "My neighbor just got diagnosed with {disease}. Very scary!",
    "Hospitals near {location} are overflowing with {symptom1} patients.",
    "ALERT: Cases of {disease} rising in {location}. Stay safe!",
    "{count} people in {location} reported {symptom1} and {symptom2} this week.",
    "I've been dealing with {symptom1} for 3 days. Could it be {disease}?",
    "Breaking news: {disease} outbreak suspected in {location}.",
    "Health department warns about {disease} spread in {location} area.",
    "My whole family has {symptom1}. The doctor says it might be {disease}.",
    "Feeling okay today but had {symptom1} yesterday near {location}.",
    "Local clinic reports surge in {symptom1} cases in {location}.",
    "Travel advisory issued for {location} due to {disease} concerns.",
    "So many people sick with {symptom1} and {symptom2} around {location}.",
    "Just recovered from {disease}! Symptoms included {symptom1} and {symptom2}.",
    "Reports of {disease} cases emerging in {location} suburb.",
]

LOCATIONS = [
    "Chennai", "Mumbai", "Delhi", "Bangalore", "Kolkata",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Coimbatore", "Madurai", "Trichy", "Salem", "Vellore",
    "New York", "London", "Tokyo", "Paris", "Berlin",
    "Sydney", "Dubai", "Singapore", "Geneva", "Toronto"
]

SYMPTOMS_LIST = list(SYMPTOM_KEYWORDS.keys())


def generate_simulated_post(outbreak_mode=False):
    """Generate a single simulated social media / news post."""
    template = random.choice(POST_TEMPLATES)
    if outbreak_mode:
        symptom1 = random.choice(["fever", "cough", "dengue", "viral", "high temperature"])
        symptom2 = random.choice(["chills", "rash", "vomiting", "headache"])
        disease = random.choice(["COVID-19", "Dengue", "Influenza"])
    else:
        symptom1 = random.choice(SYMPTOMS_LIST)
        symptom2 = random.choice(SYMPTOMS_LIST)
        disease = random.choice(DISEASE_TAGS)

    post = template.format(
        symptom1=symptom1,
        symptom2=symptom2,
        disease=disease,
        location=random.choice(LOCATIONS),
        count=random.randint(5, 300),
    )
    return post


def scrape_simulated_health_news(num_posts=50, outbreak_mode=False):
    """
    Simulate scraping health-related posts.
    In real scenario, replace this with actual BeautifulSoup scraping logic.
    """
    posts = []
    for _ in range(num_posts):
        text = generate_simulated_post(outbreak_mode=outbreak_mode)
        posts.append({
            "text": text,
            "source": random.choice(["Twitter", "Reddit", "HealthNews", "LocalNews"]),
            "location": random.choice(LOCATIONS),
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat(),
        })
    return posts


def scrape_real_health_news() -> List[Dict[str, Any]]:
    """
    Attempt to scrape real health headlines from a public RSS / news site.
    Falls back gracefully if network is unavailable.
    """
    articles: List[Dict[str, Any]] = []
    try:
        url = "https://www.who.int/rss-feeds/news-english.xml"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")[:20]
        for item in items:
            title_tag = item.find("title")
            desc_tag = item.find("description")
            title = str(title_tag.text) if title_tag else ""
            description = str(desc_tag.text) if desc_tag else ""
            articles.append({
                "text": f"{title}. {description}",
                "source": "WHO News",
                "location": "Global",
                "timestamp": datetime.now().isoformat(),
            })
    except Exception as e:
        print(f"[WARN] Real scraping failed: {e}. Using simulated data only.")
    return articles


# ─── NLP Processing ────────────────────────────────────────────────────────────

def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)         # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_symptom_features(text: str) -> Dict[str, Any]:
    """
    Extract symptom keyword frequencies and compute a weighted symptom score.
    Returns a dict of feature counts + total score.
    """
    cleaned = clean_text(text)
    features: Dict[str, Any] = {kw: 0 for kw in SYMPTOM_KEYWORDS}
    scores_list: List[int] = []
    for keyword, weight in SYMPTOM_KEYWORDS.items():
        v_count = int(cleaned.count(keyword))
        if v_count > 0:
            features[keyword] = v_count
            scores_list.append(v_count * int(weight))

    total_score: int = sum(scores_list)

    features["symptom_score"] = total_score
    features["word_count"] = len(cleaned.split())
    return features


def process_posts(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a list of raw posts and return enriched records."""
    processed: List[Dict[str, Any]] = []
    for post in posts:
        text_val = str(post.get("text", ""))
        features = extract_symptom_features(text_val)
        score_val = int(features.get("symptom_score", 0))
        record: Dict[str, Any] = {
            **post,
            "cleaned_text": clean_text(text_val),
            "features": features,
            "symptom_score": score_val,
            "label": 1 if score_val >= 6 else 0,  # 1 = possible outbreak
            "processed_at": datetime.now().isoformat(),
        }
        processed.append(record)
    return processed


# ─── MongoDB Storage ───────────────────────────────────────────────────────────

def get_mongo_collection(uri="mongodb://localhost:27017/", db="epidemic_radar", collection="posts"):
    """Return a MongoDB collection handle."""
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client[db][collection]


def save_to_mongodb(records, collection):
    """Insert processed records into MongoDB."""
    if records:
        result = collection.insert_many(records)
        print(f"[INFO] Saved {len(result.inserted_ids)} records to MongoDB.")
    return len(records)


# ─── Main Execution ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Epidemic Outbreak Early Warning Radar - Data Collector")
    print("=" * 60)

    # Collect simulated + real posts
    print("\n[1] Collecting simulated posts...")
    sim_posts = scrape_simulated_health_news(num_posts=80)

    print("[2] Attempting real news scrape (WHO RSS)...")
    real_posts = scrape_real_health_news()

    all_posts = sim_posts + real_posts
    print(f"[3] Total posts collected: {len(all_posts)}")

    # NLP Processing
    print("[4] Processing posts with NLP...")
    processed = process_posts(all_posts)
    high_risk = [p for p in processed if p["label"] == 1]
    print(f"    → High-risk posts: {len(high_risk)}")

    # Save to MongoDB
    print("[5] Saving to MongoDB...")
    try:
        col = get_mongo_collection()
        saved = save_to_mongodb(processed, col)
        print(f"    → {saved} records saved successfully.")
    except Exception as e:
        print(f"    [WARN] MongoDB not available: {e}. Skipping storage.")

    print("\n[✓] Data collection complete!")
