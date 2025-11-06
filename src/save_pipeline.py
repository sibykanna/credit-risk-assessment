# src/save_pipeline.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "raw", "credit_data.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
PIPE_PATH = os.path.join(MODEL_DIR, "credit_xgb_pipeline.joblib")

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def preprocess_and_train():
    df = load_data()
    if "SeriousDlqin2yrs" not in df.columns:
        raise ValueError("Target column not found.")
    # Basic numeric fill
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_res, y_res)

    # Save pipeline dict
    pipeline = {
        "scaler": scaler,
        "model": model,
        "feature_columns": X.columns.tolist()
    }
    joblib.dump(pipeline, PIPE_PATH)
    print(f"Saved pipeline to {PIPE_PATH}")
    return PIPE_PATH

if __name__ == "__main__":
    preprocess_and_train()
