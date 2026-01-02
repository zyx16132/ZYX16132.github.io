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
# 3️⃣ 左侧输入：严格匹配训练列名
# =============================

# 左侧显示顺序可以调整，但 key 必须和 feature_cols + cat_cols 一致
inputs = {}

# 1️⃣ 分类列
for cat in cat_cols:
    options = list(encoder_mapping[cat].keys())
    inputs[cat] = st.sidebar.selectbox(f"{cat}", options)

# 2️⃣ 数值列
# 给用户默认值和范围，可以美观排列
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

for feat in feature_cols:
    min_val, max_val, default = feature_ranges.get(feat, (0.0, 100.0, 0.0))
    inputs[feat] = st.sidebar.number_input(
        label=feat,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        format="%.3f"
    )

# =============================
# 4️⃣ Predict 按钮
# =============================
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# =============================
# 5️⃣ 预测逻辑
# =============================
if predict_btn:
    # 构造 DataFrame
    X_user = pd.DataFrame([inputs])

    # 分类列映射
    for cat in cat_cols:
        X_user[cat] = X_user[cat].map(encoder_mapping[cat])
        if X_user[cat].isna().any():
            # 若映射为空则用平均值填充
            X_user[cat] = X_user[cat].fillna(np.mean(list(encoder_mapping[cat].values())))

    # 严格按照模型训练列顺序排列
    X_user = X_user[feature_cols + cat_cols]

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
