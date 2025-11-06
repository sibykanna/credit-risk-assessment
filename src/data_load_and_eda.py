# src/data_load_and_eda.py
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA = os.path.join(ROOT, "data", "raw", "credit_data.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data(path=RAW_DATA):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please put the CSV in data/raw/ and name it credit_data.csv")
    df = pd.read_csv(path)
    return df

def basic_info(df):
    print("=== HEAD ===")
    print(df.head(5).to_string(index=False))
    print("\n=== SHAPE ===")
    print(df.shape)
    print("\n=== INFO ===")
    print(df.info())
    print("\n=== DESCRIBE ===")
    print(df.describe(include='all').T)

def missing_and_unique(df):
    print("\n=== MISSING VALUES ===")
    mv = df.isnull().sum().sort_values(ascending=False)
    print(mv[mv>0])
    print("\n=== UNIQUE COUNTS (top 20 cols) ===")
    print(df.nunique().sort_values(ascending=False).head(20))

def target_distribution(df, target_col):
    print(f"\n=== TARGET DISTRIBUTION: {target_col} ===")
    print(df[target_col].value_counts(dropna=False))
    # Quick pie / bar plot
    counts = df[target_col].value_counts(normalize=False)
    ax = counts.plot(kind="bar")
    ax.set_title(f"Distribution of {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

def save_clean_sample(df, out_path=os.path.join(PROCESSED_DIR, "clean_sample.csv")):
    # simple minimal cleaning: drop duplicates, keep first N rows as sample
    df2 = df.drop_duplicates().reset_index(drop=True)
    df2.to_csv(out_path, index=False)
    print(f"Saved sample cleaned CSV to {out_path}")

def main():
    df = load_data()
    basic_info(df)
    missing_and_unique(df)

    # Guess target column: look for common names
    possible_targets = [c for c in df.columns if c.lower() in ("target", "default", "loan_status", "bad_loan", "serious_dlqin2yrs")]
    if possible_targets:
        target_col = possible_targets[0]
        target_distribution(df, target_col)
    else:
        print("\nNo obvious target column found automatically. Check column names and tell me which column is the label (e.g., 'default' or 'loan_status')\n")
        print("Columns:", df.columns.tolist())

    save_clean_sample(df)

if __name__ == "__main__":
    main()
