import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
import shap

sns.set(style="whitegrid")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "raw", "credit_data.csv")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df


def clean_data(df):
    # Fill missing numeric values with median and drop duplicates
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df):
    if "SeriousDlqin2yrs" not in df.columns:
        raise ValueError("Target column 'SeriousDlqin2yrs' not found in the dataset.")
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def scale_and_balance(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Before SMOTE:", np.bincount(y_train))
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    print("After SMOTE:", np.bincount(y_res))

    return X_res, X_test_scaled, y_res, y_test, scaler


def train_xgboost(X_res, y_res):
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
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Some wrappers might not have predict_proba; handle gracefully
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        # fallback: use decision_function if available, else predicted labels
        if hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {auc:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - XGBoost")
        plt.legend(loc="lower right")
        plt.show()
    except Exception as e:
        print("Could not compute ROC-AUC / ROC Curve:", e)


def explain_model(model, X_train):
    """
    Try to create a SHAP feature importance plot. If SHAP fails
    due to XGBoost/SHAP compatibility issues, fall back to
    XGBoost's built-in feature importance plot.
    """
    # Ensure X_train is a DataFrame with column names
    if not isinstance(X_train, pd.DataFrame):
        try:
            # try to get columns from model or leave as generic columns
            cols = getattr(X_train, "columns", None)
            X_train = pd.DataFrame(X_train, columns=cols)
        except Exception:
            X_train = pd.DataFrame(X_train)

    # Try the newer SHAP Explainer API first
    try:
        print("Trying shap.Explainer(...) (recommended)...")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar")
        return
    except Exception as e:
        print("shap.Explainer failed:", repr(e))
        print("Trying TreeExplainer on model.get_booster() ...")

    # Try TreeExplainer with the underlying booster (common workaround)
    try:
        booster = getattr(model, "get_booster", lambda: model)()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar")
        return
    except Exception as e:
        print("TreeExplainer on booster failed:", repr(e))
        print("Falling back to XGBoost built-in feature importance plot...")

    # Final fallback: xgboost.plot_importance (feature gain)
    try:
        ax = plot_importance(model, max_num_features=20)
        ax.set_title("XGBoost Feature Importance (gain)")
        plt.show()
    except Exception as e:
        print("xgboost.plot_importance also failed:", repr(e))
        print("No feature importance plot available.")


def main():
    df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_res, X_test_scaled, y_res, y_test, scaler = scale_and_balance(X_train, X_test, y_train, y_test)

    # Train XGBoost on resampled (balanced) training data
    model = train_xgboost(X_res, y_res)

    # Evaluate on the scaled test set
    evaluate(model, X_test_scaled, y_test)

    # Try to explain model: pass a DataFrame of training data (resampled)
    print("\nGenerating SHAP feature importance plot (with fallbacks)...")
    try:
        # If X_train has columns, use them; otherwise we'll create generic names in the function
        X_res_df = pd.DataFrame(X_res, columns=X_train.columns)
    except Exception:
        X_res_df = pd.DataFrame(X_res)
    explain_model(model, X_res_df)


if __name__ == "__main__":
    main()
