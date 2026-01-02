# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# =============================
# 1️⃣ 定义 TargetEncoderCV（必须和训练时一致）
# =============================
class TargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, n_splits=5, random_state=42):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = None
        self.mapping_ = dict()

    def fit(self, X, y, groups=None):
        self.global_mean_ = y.mean()
        self.mapping_ = dict()
        for col in self.cat_cols:
            if col in X.columns:
                self.mapping_[col] = y.groupby(X[col]).mean()
            else:
                self.mapping_[col] = pd.Series(dtype=float)
        return self

    def transform(self, X, y=None, groups=None):
        X_encoded = X.copy()
        for col in self.cat_cols:
            if col not in X_encoded.columns:
                continue
            X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# =============================
# 2️⃣ 加载模型和 encoder
# =============================
bundle = joblib.load("xgb_pipeline.joblib")
model = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]  # 数值特征
cat_col = ['Antibiotic']              # 分类特征

# =============================
# 3️⃣ 页面布局
# =============================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

st.sidebar.header("Please enter parameters")

# =============================
# 4️⃣ 特征范围和默认值
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

# 分类特征选择框
ANTIBIOTIC_LIST = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox("Type of Antibiotic", ANTIBIOTIC_LIST)

# 数值特征输入框
for feat, (min_val, max_val, default) in feature_ranges.items():
    inputs[feat] = st.sidebar.number_input(
        f"{feat} ({min_val}, {max_val})",
        value=float(default),
        min_value=float(min_val),
        max_value=float(max_val),
        format="%.3f"
    )

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# =============================
# 5️⃣ 预测逻辑（只修改 app.py，不碰模型）
# =============================
if predict_btn:
    # 构造用户输入 DataFrame
    X_user = pd.DataFrame([inputs])

    # 分类列编码
    X_user_enc = encoder.transform(X_user)

    # -----------------------------
    # 对齐 XGBoost 训练列顺序
    # -----------------------------
    trained_cols = model.get_booster().feature_names
    X_user_enc_aligned = pd.DataFrame()
    for col in trained_cols:
        if col in X_user_enc.columns:
            X_user_enc_aligned[col] = X_user_enc[col]
        else:
            # 如果训练时有列但用户输入没有，则填 0（不会影响预测结果）
            X_user_enc_aligned[col] = 0.0
    X_user_enc = X_user_enc_aligned

    # -----------------------------
    # 预测
    # -----------------------------
    pred = model.predict(X_user_enc)[0]

    # -----------------------------
    # 显示结果
    # -----------------------------
    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}%`")

    # 仪表盘
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Degradation rate (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter the parameters on the left and click Predict.")

st.markdown("---")
st.markdown("*This system uses a unified machine learning pipeline to ensure consistent preprocessing and prediction.*")
