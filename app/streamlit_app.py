# app/streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "credit_xgb_pipeline.joblib")

@st.cache_data
def load_pipeline(path=MODEL_PATH):
    return joblib.load(path)

def main():
    st.set_page_config(page_title="Credit Risk Demo", layout="centered")
    st.title("Credit Risk Assessment — Demo")

    pipeline = load_pipeline()
    scaler = pipeline["scaler"]
    model = pipeline["model"]
    cols = pipeline["feature_columns"]

    st.write("Enter customer features below (use realistic values).")

    # Build form inputs dynamically using feature names
    inputs = {}
    with st.form("input_form"):
        for c in cols:
            # choose numeric input; default 0
            val = st.number_input(label=c, value=0.0, format="%.3f")
            inputs[c] = val
        submitted = st.form_submit_button("Predict")

    if submitted:
        X = pd.DataFrame([inputs], columns=cols)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[:, 1][0]
        pred = model.predict(X_scaled)[0]
        st.metric("Default probability", f"{prob:.4f}")
        st.write("Predicted label (1 = default / 0 = non-default):", int(pred))

        st.write("**Interpretation:** higher probability → higher credit risk.")
        st.balloons()

if __name__ == "__main__":
    main()
