# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =============================
# 1️⃣ 加载模型 bundle（无需自定义类）
# =============================
@st.cache_resource
def load_pipeline():
    bundle = joblib.load("xgb_pipeline_no_class.joblib")
    return bundle

bundle = load_pipeline()
model = bundle["model"]
encoder_mapping = bundle["encoder_mapping"]
feature_cols = bundle["feature_cols"]  # 数值列
cat_cols = bundle["cat_cols"]          # 分类列，如 ['Antibiotic']

# =============================
# 2️⃣ 页面布局
# =============================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

# =============================
# 3️⃣ 数值特征范围与默认值
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
# 4️⃣ 分类特征输入（selectbox）
# =============================
for cat in cat_cols:
    options = list(encoder_mapping[cat].keys())
    options.sort()
    inputs[cat] = st.sidebar.selectbox(f"{cat}", options)

# =============================
# 5️⃣ 数值特征输入
# =============================
for feat in feature_cols:
    if feat not in feature_ranges:
        st.warning(f"Feature '{feat}' not found in feature_ranges, using default 0.0")
        min_val, max_val, default = 0.0, 100.0, 0.0
    else:
        min_val, max_val, default = feature_ranges[feat]

    inputs[feat] = st.sidebar.number_input(
        label=feat,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        format="%.3f"
    )

# =============================
# 6️⃣ Predict 按钮
# =============================
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# =============================
# 7️⃣ 预测逻辑
# =============================
if predict_btn:
    # 构造用户输入 DataFrame
    X_user = pd.DataFrame([inputs])

    # 分类列映射（安全处理未知类别）
    for cat in cat_cols:
        mapping = encoder_mapping.get(cat, {})
        X_user[cat] = X_user[cat].map(mapping)
        # 如果输入的类别不在训练集中，用均值填充
        if X_user[cat].isna().any():
            X_user[cat] = X_user[cat].fillna(np.mean(list(mapping.values())) if mapping else 0.0)

    # 确保列顺序与训练一致
    final_cols = feature_cols + cat_cols
    X_user = X_user[final_cols]

    # 预测
    pred = model.predict(X_user)[0]

    # 显示结果
    st.markdown(f"### ✅ Predicted Degradation rate: **{pred:.2f}%**")

    # 仪表盘
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
