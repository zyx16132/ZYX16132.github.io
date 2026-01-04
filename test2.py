# app.py（最终可部署版本）
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# -------------------- 1. 加载模型和文件 --------------------
@st.cache_resource
def load_pipeline():
    model = joblib.load("xgb_best.pkl")
    with open("antibiotic_onehot_map.json", "r", encoding="utf-8") as f:
        antibiotic_map = json.load(f)
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    return model, antibiotic_map, feature_columns

model, antibiotic_map, feature_columns = load_pipeline()

# 自动识别抗生素 one-hot 列
antibiotic_onehot_cols = [c for c in feature_columns if c.startswith("Antibiotic_")]

# -------------------- 2. 页面布局 --------------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

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

# -------------------- 3. 分类特征（Antibiotic） --------------------
inputs["Antibiotic"] = st.sidebar.selectbox(
    "Antibiotic",
    options=sorted(antibiotic_map.keys())
)

# -------------------- 4. 数值特征 --------------------
for col in sidebar_order:
    if col != "Antibiotic":
        min_val, max_val, default = feature_ranges[col]
        inputs[col] = st.sidebar.number_input(
            label=col,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.001,
            format="%.3f"
        )

# -------------------- 5. Predict 按钮 --------------------
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# -------------------- 6. 预测逻辑 --------------------
if predict_btn:
    X_user = pd.DataFrame(index=[0])

    # Antibiotic one-hot 展开
    onehot_str = antibiotic_map[inputs["Antibiotic"]]  # "1000000000" 类型
    for col, val in zip(antibiotic_onehot_cols, onehot_str):
        X_user[col] = int(val)

    # 数值特征
    X_user["pH"] = inputs["pH"]
    X_user["Water content (%)"] = inputs["Water content(%)"]
    X_user["m (g)"] = inputs["m(g)"]
    X_user["T (°C)"] = inputs["T(°C)"]
    X_user["V (L)"] = inputs["V(L)"]
    X_user["t (min)"] = inputs["t(min)"]
    X_user["Acid Conc (mol/L)"] = inputs["HCL Conc(mol/L)"]
    X_user["Alkali Conc (mol/L)"] = inputs["NaOH Conc(mol/L)"]

    # 列顺序对齐训练
    X_user_final = X_user[feature_columns]

    # 预测
    pred = model.predict(X_user_final)[0]

    # 显示结果
    st.markdown(f"### ✅ Predicted Degradation rate: **{pred:.2f}%**")

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
    "and the exact one-hot encoding scheme from the training pipeline.*"
)
