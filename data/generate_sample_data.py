"""
Saurish Fraud Detection — Sample Data Generator
Generates a sample creditcard.csv if you don't have the Kaggle dataset.
Run: python generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os

OUTPUT = os.path.join(os.path.dirname(__file__), 'creditcard.csv')
np.random.seed(42)

N_LEGIT = 9700
N_FRAUD = 300

print("Generating synthetic credit card transaction data...")

legit_features = np.random.randn(N_LEGIT, 28) * 0.5
fraud_features = np.random.randn(N_FRAUD, 28) * 1.8 + np.random.choice([-1, 1], (N_FRAUD, 28)) * 1.2

cols = [f'V{i}' for i in range(1, 29)]

df_legit = pd.DataFrame(legit_features, columns=cols)
df_legit['Time']   = np.random.uniform(0, 172800, N_LEGIT)
df_legit['Amount'] = np.abs(np.random.lognormal(3.5, 1.2, N_LEGIT))
df_legit['Class']  = 0

df_fraud = pd.DataFrame(fraud_features, columns=cols)
df_fraud['Time']   = np.random.uniform(0, 172800, N_FRAUD)
df_fraud['Amount'] = np.abs(np.random.lognormal(5.5, 1.8, N_FRAUD))
df_fraud['Class']  = 1

df = pd.concat([df_legit, df_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUTPUT, index=False)

print(f"Saved {len(df)} rows to {OUTPUT}")
print(f"Fraud rate: {df['Class'].mean():.2%}")
print("Done!")
