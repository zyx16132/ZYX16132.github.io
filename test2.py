# test2.py（最终修正版）
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -------------------- 1. 加载 3 个独立文件（无 bundle） --------------------
@st.cache_resource
def load_pipeline():
    model   = joblib.load("final_model_only.joblib")
    mapping = joblib.load("encoder_mapping.json")
    columns = joblib.load("train_columns.json")
    return model, mapping, columns

model, encoder_mapping, train_columns = load_pipeline()
feature_cols = [c for c in train_columns if c != 'Antibiotic']
cat_cols     = ['Antibiotic']

# -------------------- 2. 页面布局（同原文件） --------------------
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

# -------------------- 3. 分类特征（动态全部抗生素） --------------------
for col in sidebar_order:
    if col in cat_cols:
        options = sorted(encoder_mapping[col].keys())
        inputs[col] = st.sidebar.selectbox(col, options)

# -------------------- 4. 数值特征（保留 3 位小数） --------------------
for col in sidebar_order:
    if col in feature_cols:
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

# -------------------- 6. 预测逻辑（对齐 train_columns） --------------------
if predict_btn:
    X_user = pd.DataFrame(columns=train_columns)
    for col, val in inputs.items():
        X_user.loc[0, col] = val
    # 直接 map，永无除零
    for cat in cat_cols:
        mapping = encoder_mapping[cat]
        X_user[cat] = X_user[cat].map(mapping).fillna(np.mean(list(mapping.values())))
    X_user = X_user.astype(float)
    X_user_final = X_user[train_columns]
    pred = model.predict(X_user_final.values)[0]

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
    "and the same target encoding as the training pipeline.*"
)
