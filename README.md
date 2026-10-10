# Epidemic Outbreak Early Warning Radar
# README

## 🦠 Project Overview
**Team:** CODE CRAFTERS | **Event:** Hackathon 2K26

A real-time public health surveillance system that detects early signs of disease outbreaks (COVID-19, Dengue, Influenza) by:
- Scraping and analysing simulated news / social media posts
- Extracting symptom keywords using NLP
- Predicting outbreak probability using a Random Forest classifier
- Storing data in MongoDB
- Serving a Flask API with live stats and prediction endpoints
- Displaying everything on a rich Streamlit dashboard with interactive maps

---

## 📁 Project Structure

```
hathon2k26/
├── data_collection/
│   └── scraper.py             ← BeautifulSoup scraper + NLP processor + MongoDB storage
├── ml_model/
│   ├── train_model.py         ← Random Forest training script
│   ├── predictor.py           ← Inference / prediction module
│   └── saved_model/           ← Auto-created after training (joblib files)
├── backend/
│   └── app.py                 ← Flask REST API
├── frontend/
│   └── dashboard.py           ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Start MongoDB
Make sure MongoDB is running locally on port 27017.
If MongoDB is not available, the system runs fine in **memory-only mode**.

```bash
# Windows
net start MongoDB
```

### 3. Train the ML Model
```bash
cd ml_model
python train_model.py
```
This generates `saved_model/rf_outbreak_model.joblib` and related files.

### 4. Start the Flask Backend
Open a new terminal:
```bash
cd backend
python app.py
```
API will be live at: `http://localhost:5000`

### 5. Launch the Streamlit Dashboard
Open another terminal:
```bash
cd frontend
streamlit run dashboard.py
```
Dashboard opens at: `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/data` | Latest processed posts |
| POST | `/api/predict` | Predict outbreak for input text |
| GET | `/api/stats` | Aggregate statistics |
| POST | `/api/refresh` | Trigger new data collection |
| GET | `/api/locations` | City-level risk data for map |
| GET | `/api/timeline` | Hourly symptom score trend |

### Example `/api/predict` Call:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Dengue and fever cases rising rapidly in Chennai hospitals!"}'
```

---

## 🚨 Alert System
- Symptom score is computed by matching weighted keywords in each post.
- Any post with `symptom_score >= 6` is flagged with:
  > **🚨 ALERT: Possible Outbreak Detected!**
- If >30% of posts are flagged, a **Global Alert** is raised in the dashboard.

---

## 🦠 Detected Diseases
- COVID-19 / Coronavirus
- Dengue Fever
- Influenza / Flu
- Malaria

---

## 📦 Tech Stack
| Layer | Technology |
|-------|-----------|
| Scraping | BeautifulSoup4, Requests |
| NLP | NLTK, Regex |
| ML | Scikit-learn (Random Forest) |
| Storage | MongoDB (pymongo) |
| Backend | Flask, Flask-CORS |
| Dashboard | Streamlit, Plotly |
| Data Gen | Faker |
