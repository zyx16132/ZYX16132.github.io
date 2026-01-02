import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =============================
# 1️⃣ Load model pipeline
# =============================
@st.cache_resource
def load_pipeline():
    bundle = joblib.load("xgb_pipeline_no_class.joblib")
    return bundle

bundle = load_pipeline()
model = bundle["model"]
encoder_mapping = bundle["encoder_mapping"]
feature_cols = bundle["feature_cols"]  # numeric features
cat_cols = bundle["cat_cols"]          # categorical features, e.g., ['Antibiotic']

# =============================
# 2️⃣ Page setup
# =============================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

# =============================
# 3️⃣ Sidebar input order (can be arbitrary)
# =============================
sidebar_order = [
    "Antibiotic", "pH", "Water content(%)", "m(g)", "T(°C)",
    "V(L)", "t(min)", "HCL Conc(mol/L)", "NaOH Conc(mol/L)"
]

feature_ranges = {
    'pH': (2.0, 12.0, 6.08),
    'Water content(%)': (5.35, 98.1, 69.9),
    'm(g)': (1.0, 500.0, 79.36),
    'T(°C)': (0.0, 340.0, 117.8),
    'V(L)': (0.05, 1.0, 0.23),
    't(min)': (0.0, 480.0, 64.59),
    'HCL Conc(mol/L)': (0.0, 0.6, 0.06),
    'NaOH Conc(mol/L)': (0.0, 0.6, 0.01)
}

inputs = {}

# =============================
# 4️⃣ Categorical feature inputs
# =============================
for col in sidebar_order:
    if col in cat_cols:
        options = list(encoder_mapping[col].keys())
        inputs[col] = st.sidebar.selectbox(col, options)

# =============================
# 5️⃣ Numeric feature inputs
# =============================
for col in sidebar_order:
    if col in feature_cols:
        min_val, max_val, default = feature_ranges[col]
        inputs[col] = st.sidebar.number_input(
            label=col,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            format="%.3f"
        )

# =============================
# 6️⃣ Predict button
# =============================
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# =============================
# 7️⃣ Prediction logic
# =============================
if predict_btn:
    # Convert inputs to DataFrame
    X_user = pd.DataFrame([inputs])

    # Map categorical features
    for cat in cat_cols:
        X_user[cat] = X_user[cat].map(encoder_mapping[cat])
        if X_user[cat].isna().any():
            X_user[cat] = X_user[cat].fillna(np.mean(list(encoder_mapping[cat].values())))

    # Ensure strict column order for the model
    X_user_final = X_user[feature_cols + cat_cols]

    # Prediction
    pred = model.predict(X_user_final)[0]

    # Display result
    st.markdown(f"### ✅ Predicted Degradation rate: **{pred:.2f}%**")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"suffix": "%"},
        title={"text": "Degradation rate"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkgreen"},
            "steps": [
                {"range": [0, 50], "color": "#f2f2f2"},
                {"range": [50, 100], "color": "#c7e9c0"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter the parameters on the left and click Predict.")

st.markdown("---")
st.markdown(
    "*This application uses the final trained XGBoost model "
    "and the same target encoding as the training pipeline.*"
)
