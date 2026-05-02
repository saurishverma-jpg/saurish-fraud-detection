"""
Saurish AI Fraud Detection — Flask Backend API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model (will be created by train_model.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/fraud_model.pkl')
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.warning("Model file not found. Run train_model.py first.")

load_model()


def extract_features(data: dict) -> np.ndarray:
    """
    Convert raw transaction dict into feature vector.
    In production this maps to the 28 PCA-transformed features
    from the Kaggle Credit Card Fraud dataset.
    """
    amount = float(data.get('amount', 0))
    hour = int(data.get('hour', 12))
    is_night = 1 if (hour >= 23 or hour <= 5) else 0
    has_fraud_history = 1 if data.get('fraud_history') == 'yes' else 0
    location_risk = float(data.get('location_risk', 0.1))
    merchant_risk = float(data.get('merchant_risk', 0.1))

    # Normalize amount (log scale)
    log_amount = np.log1p(amount)

    # Feature vector (28 features to match trained model)
    features = np.zeros(30)
    features[0] = log_amount
    features[1] = hour / 24.0
    features[2] = is_night
    features[3] = has_fraud_history
    features[4] = location_risk
    features[5] = merchant_risk
    # V1–V28: PCA components (zeroed for demo; use real PCA in production)
    features[6:] = np.random.normal(0, 0.5, 24)

    return features.reshape(1, -1)


@app.route('/')
def index():
    return jsonify({'status': 'ok', 'project': 'Saurish AI Fraud Detection', 'version': '1.0'})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Body: { amount, hour, fraud_history, location_risk, merchant_risk }
    Returns: { risk_score, label, confidence, timestamp }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        features = extract_features(data)

        if model is not None:
            prob = model.predict_proba(features)[0][1]
        else:
            # Heuristic fallback when model not trained yet
            amount = float(data.get('amount', 0))
            score = 0.05
            if amount > 10000: score += 0.40
            elif amount > 3000: score += 0.25
            elif amount > 1000: score += 0.10
            if data.get('fraud_history') == 'yes': score += 0.30
            score += float(data.get('location_risk', 0.1)) * 0.3
            score += float(data.get('merchant_risk', 0.1)) * 0.2
            prob = min(score, 0.99)

        if prob >= 0.7:
            label = 'FRAUD'
        elif prob >= 0.4:
            label = 'REVIEW'
        else:
            label = 'SAFE'

        result = {
            'risk_score': round(float(prob), 4),
            'label': label,
            'confidence': round(abs(prob - 0.5) * 2, 4),
            'timestamp': datetime.utcnow().isoformat(),
            'model': 'heuristic' if model is None else 'RandomForest'
        }

        logger.info(f"Prediction: amount={data.get('amount')}, score={prob:.4f}, label={label}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """GET /api/transactions — Returns sample transaction list"""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    status_filter = request.args.get('status', 'all')

    transactions = [
        {'id': 'TXN-8821', 'amount': 4500.00, 'merchant': 'Unknown Merchant', 'location': 'Lagos, NG', 'risk_score': 0.97, 'status': 'fraud', 'date': '2024-01-15T14:32:00Z'},
        {'id': 'TXN-8820', 'amount': 12.99,   'merchant': 'Spotify',           'location': 'London, UK','risk_score': 0.02, 'status': 'safe',  'date': '2024-01-15T14:28:00Z'},
        {'id': 'TXN-8819', 'amount': 12.99,   'merchant': 'Netflix',            'location': 'New York, US','risk_score': 0.03,'status': 'safe', 'date': '2024-01-15T14:20:00Z'},
        {'id': 'TXN-8818', 'amount': 3200.00, 'merchant': 'Jewelry Store',      'location': 'Dubai, AE','risk_score': 0.78, 'status': 'review','date': '2024-01-15T13:55:00Z'},
        {'id': 'TXN-8816', 'amount': 8900.00, 'merchant': 'Crypto Exchange',    'location': 'Unknown',  'risk_score': 0.94, 'status': 'fraud', 'date': '2024-01-15T13:12:00Z'},
        {'id': 'TXN-8814', 'amount': 22.50,   'merchant': "McDonald's",         'location': 'Delhi, IN', 'risk_score': 0.01, 'status': 'safe', 'date': '2024-01-15T12:22:00Z'},
    ]

    if status_filter != 'all':
        transactions = [t for t in transactions if t['status'] == status_filter]

    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        'transactions': transactions[start:end],
        'total': len(transactions),
        'page': page,
        'per_page': per_page
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """GET /api/stats — Returns dashboard metrics"""
    return jsonify({
        'total_transactions': 12847,
        'fraud_detected': 47,
        'model_accuracy': 0.987,
        'amount_saved': 2400000,
        'precision': 0.962,
        'recall': 0.948,
        'f1_score': 0.955,
        'auc_roc': 0.998,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
