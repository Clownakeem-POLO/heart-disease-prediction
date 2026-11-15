"""
Streamlit demo for the heart disease classifier.

Loads the model trained in analysis.ipynb (via build_notebook.py) and lets you enter
a patient's clinical readings to get a risk estimate. The threshold is the recall
focused one chosen in the notebook, not the default 0.5.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk", page_icon="heart", layout="centered")

MODEL_PATH = Path("model/heart_model.joblib")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

st.title("Heart disease risk estimate")
st.caption(
    "Trained on the UCI Cleveland dataset (297 patients). This is a portfolio "
    "demonstration, not a medical device. Do not use it for real diagnosis."
)

if bundle is None:
    st.error("Model file not found. Run `python build_notebook.py` first to train it.")
    st.stop()

scaler = bundle["scaler"]
model = bundle["model"]
columns = bundle["columns"]
threshold = bundle["threshold"]

st.subheader("Patient readings")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 54)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    cp = st.selectbox(
        "Chest pain type",
        [1, 2, 3, 4],
        format_func=lambda x: {1: "Typical angina", 2: "Atypical angina",
                               3: "Non-anginal", 4: "Asymptomatic"}[x],
    )
    trestbps = st.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Serum cholesterol (mg/dl)", 120, 570, 245)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl", ["No", "Yes"], horizontal=True)
    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T abnormality",
                               2: "LV hypertrophy"}[x],
    )

with col2:
    thalach = st.slider("Max heart rate achieved", 70, 210, 150)
    exang = st.radio("Exercise induced angina", ["No", "Yes"], horizontal=True)
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox(
        "Slope of peak exercise ST",
        [1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
    )
    ca = st.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox(
        "Thalassemia",
        [3, 6, 7],
        format_func=lambda x: {3: "Normal", 6: "Fixed defect",
                               7: "Reversible defect"}[x],
    )

row = {
    "age": age, "sex": 1 if sex == "Male" else 0, "cp": cp,
    "trestbps": trestbps, "chol": chol, "fbs": 1 if fbs == "Yes" else 0,
    "restecg": restecg, "thalach": thalach, "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
}
X = pd.DataFrame([row])[columns]

if st.button("Estimate risk", type="primary"):
    prob = float(model.predict_proba(scaler.transform(X))[:, 1][0])
    flagged = prob >= threshold
    st.metric("Estimated probability of disease", f"{prob:.0%}")
    if flagged:
        st.error(
            f"Flagged as elevated risk (probability {prob:.0%} is at or above the "
            f"{threshold:.0%} recall-focused threshold). Recommend clinical follow-up."
        )
    else:
        st.success(
            f"Below the {threshold:.0%} threshold (probability {prob:.0%}). "
            "Lower modelled risk."
        )
    st.caption(
        "The threshold is deliberately below 50% because, clinically, missing a case "
        "is worse than a false alarm. See the notebook for the reasoning."
    )
