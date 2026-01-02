# test2.py
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
    return joblib.load("xgb_pipeline_no_class.joblib")

bundle = load_pipeline()

# 取出各个组件
model   = bundle["model"]
encoder_mapping = bundle["encoder_mapping"]
feature_cols    = bundle["feature_cols"]   # 数值列
cat_cols        = bundle["cat_cols"]       # 分类列
train_columns   = bundle["train_columns"]  # ✅ 关键：训练时的完整列顺序

# =============================
# 2️⃣ 页面布局
# =============================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

# =============================
# 3️⃣ 侧边栏输入顺序与范围
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
# 4️⃣ 分类特征输入（selectbox）
# =============================
for col in sidebar_order:
    if col in cat_cols:
        options = list(encoder_mapping[col].keys())
        inputs[col] = st.sidebar.selectbox(col, options)

# =============================
# 5️⃣ 数值特征输入（number_input）
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
# 6️⃣ Predict 按钮
# =============================
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# =============================
# 7️⃣ 预测逻辑（完全对齐 train_columns）
# =============================
if predict_btn:
    # 1. 按训练列顺序建空表
    X_user = pd.DataFrame(columns=train_columns)

    # 2. 填值
    for col, val in inputs.items():
        X_user.loc[0, col] = val

    # 3. 分类映射
    for cat in cat_cols:
        mapping = encoder_mapping[cat]
        X_user[cat] = X_user[cat].map(mapping)
        X_user[cat] = X_user[cat].fillna(np.mean(list(mapping.values())))

    # 4. 转数值
    X_user = X_user.astype(float)

    # 5. 按训练顺序切片 → 列数/顺序 100% 一致
    X_user_final = X_user[train_columns]

    # 6. 预测（不会再报 feature mismatch）
    pred = model.predict(X_user_final.values)[0]

    # 7. 显示
    st.markdown(f"### ✅ Predicted Degradation rate: **{pred:.2f}%**")

    # 8. 仪表盘
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
