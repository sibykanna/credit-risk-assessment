import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# === Paths ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "raw", "credit_data.csv")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df

def clean_data(df):
    # Drop rows with missing target
    if "SeriousDlqin2yrs" in df.columns:
        df = df.dropna(subset=["SeriousDlqin2yrs"])
    else:
        raise ValueError("Target column 'SeriousDlqin2yrs' not found!")

    # Fill numeric missing values with median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def split_data(df):
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_logistic_regression(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
