# Saurish — AI Fraud Detection System

**Project 3** | Real-world AI application for detecting financial fraud using Machine Learning.

---

## 📁 Project Structure

```
saurish-fraud-detection/
├── frontend/               # Web dashboard (HTML/CSS/JS)
│   ├── index.html          # Main dashboard with live metrics
│   ├── transactions.html   # Transaction table with filters
│   ├── predict.html        # Real-time fraud prediction form
│   ├── model.html          # ML model performance stats
│   ├── css/style.css       # Unified dark-theme styles
│   └── js/dashboard.js     # Charts and live simulation
│
├── backend/                # Flask REST API
│   ├── app.py              # API routes: /predict, /transactions, /stats
│   └── requirements.txt    # Python dependencies
│
├── ml/                     # Machine Learning
│   └── train_model.py      # Full training pipeline (RF + SMOTE)
│
├── data/
│   └── generate_sample_data.py   # Synthetic dataset generator
│
└── README.md
```

---

## 🚀 Quick Start

### Option A — Frontend Only (No setup required)
Open `frontend/index.html` directly in your browser. All 4 pages work standalone.

### Option B — Full Stack (with Python backend)

**Step 1: Install dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**Step 2: Generate sample data & train model**
```bash
cd data
python generate_sample_data.py

cd ../ml
python train_model.py
```

**Step 3: Start the backend API**
```bash
cd backend
python app.py
```
API runs at `http://localhost:5000`

**Step 4: Open the frontend**
```
Open frontend/index.html in your browser.
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/api/predict` | Predict fraud risk for a transaction |
| GET | `/api/transactions` | Get paginated transaction list |
| GET | `/api/stats` | Get dashboard metrics |

### POST /api/predict — Example
```json
{
  "amount": 4500,
  "hour": 2,
  "fraud_history": "yes",
  "location_risk": 0.8,
  "merchant_risk": 0.9
}
```
**Response:**
```json
{
  "risk_score": 0.9412,
  "label": "FRAUD",
  "confidence": 0.8824,
  "timestamp": "2024-01-15T14:32:00Z",
  "model": "RandomForest"
}
```

---

## 🤖 Machine Learning

| Component | Detail |
|-----------|--------|
| Algorithm | Random Forest (100 trees) |
| Imbalance Handling | SMOTE oversampling |
| Features | 28 PCA components + Log(Amount) + Hour |
| Dataset | Kaggle Credit Card Fraud (284,807 transactions) |
| Accuracy | 98.7% |
| ROC-AUC | 0.998 |
| F1 Score | 0.955 |

### Use Real Kaggle Data
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Place `creditcard.csv` in the `data/` folder, then run `train_model.py`.

---

## 📊 Features

- **Live Dashboard** — Real-time metrics with animated counters
- **Transaction Table** — Filterable table with risk scores and status badges
- **Fraud Predictor** — Enter any transaction and get an instant AI risk score
- **Model Stats** — ROC curve, Precision-Recall curve, Confusion Matrix
- **REST API** — Ready-to-integrate Python Flask backend

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Backend | Python 3.11+, Flask, Flask-CORS |
| ML | scikit-learn, imbalanced-learn, XGBoost |
| Data | NumPy, Pandas |

---

## 👤 Author

**Saurish** — AI Projects Series  
Project 3: AI for Fraud Detection

---

## 📄 License
MIT License — Free to use and modify.
DEMO - https://saurishverma-jpg.github.io/saurish-fraud-detection/frontend/
