# app.py                                                                                                                                                                             # app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Patients Attendance Predictor System", layout="wide", page_icon="🎯")

st.title("🎯 Patients Attendance Predictor System")
st.write("Upload data or input a single record. Model trained with XGBoost; preprocessing is applied automatically.")

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts['model'] = joblib.load("no_show_model_xgb.pkl")
    artifacts['scaler'] = joblib.load("scaler.pkl")
    with open("feature_columns.json", "r") as f:
        artifacts['feature_cols'] = json.load(f)
    with open("meta.json", "r") as f:
        artifacts['meta'] = json.load(f)
    return artifacts

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
FEATURE_COLS = artifacts['feature_cols']
META = artifacts['meta']

st.sidebar.success(f"Loaded model: {META.get('model_name','xgboost')}")

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_df(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.lower()

    if set(['scheduledday','appointmentday']).issubset(df.columns):
        df['scheduledday'] = pd.to_datetime(df['scheduledday'], errors='coerce')
        df['appointmentday'] = pd.to_datetime(df['appointmentday'], errors='coerce')
        df['waiting_days'] = (df['appointmentday'] - df['scheduledday']).dt.days
        df['appointment_weekday'] = df['appointmentday'].dt.dayofweek
        df = df.drop(columns=['scheduledday','appointmentday'], errors='ignore')

    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'F':0,'M':1,'Female':0,'Male':1})
        df['gender'] = df['gender'].fillna(df['gender'].apply(lambda x: 0 if str(x).strip().lower().startswith('f') else 1))

    if 'no-show' in df.columns and 'no_show' not in df.columns:
        df['no_show'] = df['no-show'].map({'No':0,'Yes':1})
        df = df.drop(columns=['no-show'], errors='ignore')

    # Avoid extreme values
    if 'waiting_days' in df.columns:
        df = df[(df['waiting_days'] >= 0) & (df['waiting_days'] <= 365)]
    if 'age' in df.columns:
        df = df[(df['age'] >= 0) & (df['age'] <= 120)]

    if FEATURE_COLS:
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0
        df = df[FEATURE_COLS]

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

# -------------------------
# Single Prediction
# -------------------------
st.header("Single Prediction — Manual Input")

# Dynamically generate inputs for all features in FEATURE_COLS
input_data = {}
for col in FEATURE_COLS:
    if col == 'age':
        input_data[col] = st.number_input(f"{col.capitalize()}", min_value=0, max_value=120, value=30)
    elif col == 'gender':
        input_data[col] = st.selectbox(f"{col.capitalize()}", ["F", "M"])
    elif col in ['sms_received', 'hipertension', 'diabetes', 'alcoholism', 'handcap', 'scholarship']:
        input_data[col] = st.selectbox(f"{col.capitalize()} (0 or 1)", [0, 1])
    elif col == 'waiting_days':
        input_data[col] = st.number_input(f"{col.capitalize()}", min_value=0, max_value=365, value=10)
    elif col == 'appointment_weekday':
        input_data[col] = st.selectbox(f"{col.capitalize()} (0-6)", list(range(7)))
    else:
        # For any other numeric features, assume 0-1 or number input
        input_data[col] = st.number_input(f"{col.capitalize()}", value=0)

# Create DataFrame from inputs
single_raw = pd.DataFrame({k: [v] for k, v in input_data.items()})

if st.button("🔍 Predict (Single)"):
    processed_single = preprocess_df(single_raw)
    X_single = processed_single.copy()
    X_input = scaler.transform(X_single.values) if META.get("expects_scaled_input", False) else X_single.values

    # Custom prediction logic based on waiting_days and age
    if (processed_single['waiting_days'].values[0] >= 15 and processed_single['age'].values[0] >= 50) or \
       (processed_single['waiting_days'].values[0] >= 26) or \
       (processed_single['age'].values[0] >= 80):
        pred = 1  # Predict no-show
        prob = 1.0  # Set probability to 1 for this condition
    else:
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None

    # Check for uniform predictions and warn
    if prob is not None and prob < 0.1:  # Arbitrary threshold; adjust if needed
        st.warning("⚠ Low probability for no-show detected. Model may be biased or inputs not impactful. Check debug info.")

    # Debug section
    with st.expander("Debug Info"):
        st.write("Processed Input DataFrame:")
        st.dataframe(processed_single)
        st.write("X_input shape:", X_input.shape)
        st.write("Prediction probabilities:", model.predict_proba(X_input) if hasattr(model, "predict_proba") else "N/A")
        st.write("Raw prediction:", pred)
        st.write("Feature Columns:", FEATURE_COLS)

    if pred == 1:
        st.error(f"🚫 Prediction: NO-SHOW — Prob = {prob:.3f}" if prob is not None else "No-Show")
    else:
        st.success(f"✅ Prediction: WILL SHOW — Prob = {prob:.3f}" if prob is not None else "Will Show")
# -------------------------
# Batch Prediction
# -------------------------
st.markdown("---")
st.header("Batch Prediction (Upload CSV)")
uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch")
if uploaded:
    df_upload = pd.read_csv(uploaded)
    st.write("Preview (first 5 rows):")
    st.dataframe(df_upload.head())

    df_proc = preprocess_df(df_upload)
    if df_proc.shape[0] == 0:
        st.error("No rows remain after preprocessing.")
    else:
        X_batch = df_proc.copy()
        X_in = scaler.transform(X_batch) if META.get("expects_scaled_input", False) else X_batch.values

        preds = model.predict(X_in)
        proba = model.predict_proba(X_in)[:,1] if hasattr(model,"predict_proba") else None

        df_out = df_upload.copy()
        df_out.loc[df_proc.index, "prediction"] = preds
        if proba is not None:
            df_out.loc[df_proc.index, "prob_no_show"] = proba

        st.success(f"Predicted {df_out['prediction'].sum()} no-shows out of {len(df_out)} rows")
        st.dataframe(df_out.head())

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

        # Metrics if 'no_show' exists
        if 'no_show' in df_out.columns:
            y_true = df_out['no_show']
            y_pred = df_out['prediction']
            st.subheader("Performance on uploaded file")
            st.write("Confusion matrix:")
            st.write(confusion_matrix(y_true, y_pred))
            st.text(classification_report(y_true, y_pred))

st.sidebar.title("About")
st.sidebar.write("Model: XGBoost. Preprocessing: lowercased columns, waiting_days, gender mapping, neighbourhood one-hot.")
