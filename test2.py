# ================== app.py ==================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold

# ================== 1️⃣ 定义 TargetEncoderCV（必须） ==================
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
            if y is not None and groups is not None:
                # CV 安全编码（训练集）
                X_encoded[col] = np.nan
                gkf = GroupKFold(n_splits=self.n_splits)
                X_temp, y_temp, groups_temp = X.copy(), y.copy(), groups.copy()
                for train_idx, val_idx in gkf.split(X_temp, y_temp, groups_temp):
                    mapping = y_temp.iloc[train_idx].groupby(X_temp.iloc[train_idx][col]).mean()
                    X_encoded.iloc[val_idx, X_encoded.columns.get_loc(col)] = X_temp.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:
                # 测试集或单样本
                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# ================== 2️⃣ 加载模型 ==================
bundle = joblib.load("xgb_pipeline.joblib")  # 放在项目根目录
best_xgb = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

# ================== 3️⃣ Streamlit 页面 ==================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ------------------ 左侧输入 ------------------
st.sidebar.header("Please enter parameters")
inputs = {}

# 抗生素选择
ANTIBIOTIC_LIST = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox("Type of Antibiotic", ANTIBIOTIC_LIST)

# 数值输入
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
    inputs[k] = st.sidebar.number_input(k, value=float(v), format="%.3f")

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# ------------------ 预测逻辑 ------------------
MODEL_FEATURES = [
    'pH', 'Water content(%)', 'm(g)', 'T(°C)',
    'V(L)', 't(min)', 'HCL Conc(mol/L)', 'NaOH Conc(mol/L)',
    'Degradation', 'Antibiotic'
]

if predict_btn:
    # 构造单样本 DataFrame
    X_user = pd.DataFrame([inputs])
    X_user['Degradation'] = 0.0
    X_user = X_user[MODEL_FEATURES]

    # 编码 & 预测
    X_user_enc = encoder.transform(X_user)
    pred = best_xgb.predict(X_user_enc)[0]

    # 显示结果
    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

    # 仪表盘可视化
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
