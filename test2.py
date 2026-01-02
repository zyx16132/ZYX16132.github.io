# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# ===== 1️⃣ 定义 TargetEncoderCV（和训练时完全一致） =====
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
                X_encoded[col] = pd.Series(index=X_encoded.index, dtype=float)
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=self.n_splits)
                for train_idx, val_idx in gkf.split(X, y, groups):
                    mapping = y.iloc[train_idx].groupby(X.iloc[train_idx][col]).mean()
                    X_encoded.iloc[val_idx] = X.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:
                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# 加载模型
bundle = joblib.load("xgb_pipeline.joblib")
model = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

st.title("🧪 Degradation rate prediction system")

# 用户输入
antibiotic = st.selectbox("Type of Antibiotic", ["CEP", "AMP", "其他"])
ph = st.number_input("pH", value=5.0)
water_content = st.number_input("Water content(%)", value=70.0)
m = st.number_input("m(g)", value=80.0)
T = st.number_input("T(°C)", value=120.0)
V = st.number_input("V(L)", value=0.23)
t = st.number_input("t(min)", value=64.0)
HCL = st.number_input("HCL Conc(mol/L)", value=0.05)
NaOH = st.number_input("NaOH Conc(mol/L)", value=0.01)

# 构建 DataFrame（只包含模型特征，不要 Degradation）
X_user = pd.DataFrame({
    "pH": [ph],
    "Water content(%)": [water_content],
    "m(g)": [m],
    "T(°C)": [T],
    "V(L)": [V],
    "t(min)": [t],
    "HCL Conc(mol/L)": [HCL],
    "NaOH Conc(mol/L)": [NaOH],
    "Antibiotic": [antibiotic]
})

# 使用训练时的 encoder 转换
X_user_enc = encoder.transform(X_user)

# 预测
pred = model.predict(X_user_enc)[0]

st.write(f"Predicted Degradation: {pred:.2f}")


