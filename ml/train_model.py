"""
Saurish AI Fraud Detection — Model Training Script
Dataset: Kaggle Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Run: python train_model.py
"""

import numpy as np
import pandas as pd
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), '../data/creditcard.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_model.pkl')
REPORT_PATH = os.path.join(os.path.dirname(__file__), 'model_report.json')

RANDOM_STATE = 42
TEST_SIZE    = 0.2

print("=" * 60)
print("  Saurish AI Fraud Detection — Model Training")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        print("\n[INFO] Dataset not found. Generating synthetic data...")
        return generate_synthetic_data()

    print(f"\n[INFO] Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Fraud rate: {df['Class'].mean():.4%}")
    return df


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic fraud data for demo purposes."""
    np.random.seed(RANDOM_STATE)
    n_fraud = int(n_samples * 0.003)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = np.random.randn(n_legit, 28) * 0.5
    legit_amount = np.abs(np.random.lognormal(4, 1, n_legit))
    legit_time   = np.random.uniform(0, 172800, n_legit)

    # Fraudulent transactions (slightly different distribution)
    fraud = np.random.randn(n_fraud, 28) * 1.5 + 0.8
    fraud_amount = np.abs(np.random.lognormal(5, 2, n_fraud))
    fraud_time   = np.random.uniform(0, 172800, n_fraud)

    cols = [f'V{i}' for i in range(1, 29)]

    df_legit = pd.DataFrame(legit, columns=cols)
    df_legit['Amount'] = legit_amount
    df_legit['Time']   = legit_time
    df_legit['Class']  = 0

    df_fraud = pd.DataFrame(fraud, columns=cols)
    df_fraud['Amount'] = fraud_amount
    df_fraud['Time']   = fraud_time
    df_fraud['Class']  = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    print(f"[INFO] Synthetic data shape: {df.shape}, fraud rate: {df['Class'].mean():.4%}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    # Log-transform Amount (highly skewed)
    df = df.copy()
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] % 86400) // 3600

    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Log_Amount', 'Hour']
    X = df[feature_cols].values
    y = df['Class'].values

    print(f"\n[INFO] Features: {X.shape[1]}")
    print(f"[INFO] Class distribution: {np.bincount(y)}")
    return X, y, feature_cols


# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\n[INFO] Applying SMOTE to handle class imbalance ...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE: {np.bincount(y_res)}")

    # Model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    print("\n[INFO] Training Random Forest ...")
    pipeline.fit(X_res, y_res)

    # Evaluate
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    auc    = roc_auc_score(y_test, y_proba)
    ap     = average_precision_score(y_test, y_proba)
    f1     = f1_score(y_test, y_pred)

    print("\n" + "─" * 50)
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Avg Prec : {ap:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{np.array(cm)}")

    results = {
        'accuracy':  report['accuracy'],
        'precision': report['1']['precision'],
        'recall':    report['1']['recall'],
        'f1_score':  report['1']['f1-score'],
        'roc_auc':   auc,
        'avg_precision': ap,
        'confusion_matrix': cm,
        'model': 'RandomForestClassifier',
        'n_estimators': 100,
        'smote': True
    }

    return pipeline, results


# ─────────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────────
def save_artifacts(pipeline, results):
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[OK] Model saved to {MODEL_PATH}")

    with open(REPORT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Report saved to {REPORT_PATH}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    df = load_data()
    X, y, feature_cols = preprocess(df)
    pipeline, results = train(X, y)
    save_artifacts(pipeline, results)
    print("\n[DONE] Training complete!")
