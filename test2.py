# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =============================
# 1️⃣ 加载模型和 encoder（唯一来源）
# =============================
@st.cache_resource
def load_pipeline():
    bundle = joblib.load("xgb_pipeline.joblib")
    return bundle["model"], bundle["encoder"], bundle["feature_cols"], bundle["cat_cols"]

model, encoder, feature_cols, cat_cols = load_pipeline()

# =============================
# 2️⃣ 页面布局
# =============================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

st.sidebar.header("Please enter parameters")

# =============================
# 3️⃣ 特征范围和默认值（与你训练集一致）
# =============================
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
# 4️⃣ 分类特征（严格来自 encoder）
# =============================
antibiotic_list = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox(
    "Type of Antibiotic",
    antibiotic_list
)

# =============================
# 5️⃣ 数值特征输入
# =============================
for feat in feature_cols:
    min_val, max_val, default = feature_ranges[feat]
    inputs[feat] = st.sidebar.number_input(
        feat,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        format="%.3f"
    )

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# =============================
# 6️⃣ 预测逻辑（完全对齐训练）
# =============================
if predict_btn:
    # ---------- 构造 DataFrame ----------
    X_user = pd.DataFrame([inputs])

    # ---------- Target Encoding ----------
    X_user_enc = encoder.transform(X_user)

    # ---------- 严格列顺序 ----------
    final_cols = feature_cols + cat_cols
    X_user_enc = X_user_enc[final_cols]

    # ---------- 预测 ----------
    pred = model.predict(X_user_enc)[0]

    # =============================
    # 7️⃣ 展示结果
    # =============================
    st.markdown(f"### ✅ Predicted degradation rate: **{pred:.2f}%**")

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
    "*This application uses the final trained XGBoost model and the same "
    "target encoding strategy as the training pipeline to ensure full reproducibility.*"
)
