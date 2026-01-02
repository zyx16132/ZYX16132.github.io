import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ======================================================
# 1️⃣ 加载训练好的 pipeline
# ======================================================
bundle = joblib.load("xgb_pipeline.joblib")
best_xgb = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

# 特征顺序（保持训练时顺序 + 分类列）
MODEL_FEATURES = feature_cols.tolist() + ["Antibiotic"]

# ======================================================
# 2️⃣ Streamlit 页面
# ======================================================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

LABELS = {
    'Antibiotic': 'Type of Antibiotic',
    'pH': 'Initial environmental pH [2–12]',
    'Water content(%)': 'Water content (%) [5.35–98.1]',
    'm(g)': 'Quality (g) [1–500]',
    'T(°C)': 'Reaction temperature (°C) [0–340]',
    'V(L)': 'Reactor volume (L) [0.05–1]',
    't(min)': 'Reaction time (min) [0–480]',
    'HCL Conc(mol/L)': 'HCL concentration (mol/L) [0–0.6]',
    'NaOH Conc(mol/L)': 'NaOH concentration (mol/L) [0–0.6]'
}

# 左侧输入
st.sidebar.header("Please enter parameters")
inputs = {}

# 抗生素选择
ANTIBIOTIC_LIST = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox(LABELS['Antibiotic'], ANTIBIOTIC_LIST)

# 数值输入默认值
defaults = {
    'pH': 6.08,
    'Water content(%)': 69.9,
    'm(g)': 79.36,
    'T(°C)': 117.8,
    'V(L)': 0.23,
    't(min)': 64.59,
    'HCL Conc(mol/L)': 0.06,
    'NaOH Conc(mol/L)': 0.01
}

for k, v in defaults.items():
    inputs[k] = st.sidebar.number_input(LABELS[k], value=float(v), format="%.3f")

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# ======================================================
# 3️⃣ 预测逻辑
# ======================================================
if predict_btn:
    X_user = pd.DataFrame([inputs])
    X_user = X_user[MODEL_FEATURES]  # 保持列顺序
    X_user_enc = encoder.transform(X_user)  # 使用训练时的编码器
    pred = best_xgb.predict(X_user_enc)[0]

    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Degradation rate"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please enter parameters on the left and click Predict.")

st.markdown("---")
st.markdown("*This system uses a unified machine learning pipeline to ensure consistent preprocessing and prediction.*")
