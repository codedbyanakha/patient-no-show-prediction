# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="No-Show Prediction", layout="wide", page_icon="🎯")

st.title("🎯 No-Show Prediction Dashboard")
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
    with open("neighbourhood_columns.json", "r") as f:
        artifacts['neigh_cols'] = json.load(f)
    with open("meta.json", "r") as f:
        artifacts['meta'] = json.load(f)
    return artifacts

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
FEATURE_COLS = artifacts['feature_cols']
NEIGH_COLS = artifacts['neigh_cols']
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

    if 'neighbourhood' in df.columns and NEIGH_COLS:
        dummies = pd.get_dummies(df['neighbourhood'], prefix='neighbourhood', drop_first=True)
        df = pd.concat([df.drop(columns=['neighbourhood']), dummies], axis=1)
    for col in (NEIGH_COLS or []):
        if col not in df.columns:
            df[col] = 0

    if 'waiting_days' in df.columns:
        df = df[df['waiting_days'] >= 0]
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
c1, c2, c3, c4 = st.columns(4)
with c1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
with c2:
    gender_sel = st.selectbox("Gender", ["F","M"])
with c3:
    sms_received = st.selectbox("SMS Received (0 or 1)", [0,1])
with c4:
    waiting_days = st.number_input("Waiting days", min_value=0, max_value=365, value=10)

single_raw = pd.DataFrame({
    "age": [age],
    "gender": [gender_sel],
    "sms_received": [sms_received],
    "waiting_days": [waiting_days]
})

if st.button("🔍 Predict (Single)"):
    processed_single = preprocess_df(single_raw)
    X_single = processed_single.copy()
    X_input = scaler.transform(X_single) if META.get("expects_scaled_input", False) else X_single.values

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None

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

