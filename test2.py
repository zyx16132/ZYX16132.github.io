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

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -------------------------
# 1️⃣ 加载模型
# -------------------------
bundle = joblib.load("xgb_pipeline.joblib")
model = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

# -------------------------
# 2️⃣ 页面布局
# -------------------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

st.sidebar.header("Please enter parameters")

# -------------------------
# 3️⃣ 特征范围 & 默认值
# -------------------------
feature_ranges = {
    'pH': (2, 12, 6.08),
    'Water content(%)': (5.35, 98.1, 69.9),
    'm(g)': (1, 500, 79.36),
    'T(°C)': (0, 340, 117.8),
    'V(L)': (0.05, 1, 0.23),
    't(min)': (0, 480, 64.59),
    'HCL Conc(mol/L)': (0, 0.6, 0.06),
    'NaOH Conc(mol/L)': (0, 0.6, 0.01)
}

inputs = {}

# 分类特征
ANTIBIOTIC_LIST = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox("Type of Antibiotic", ANTIBIOTIC_LIST)

# 数值特征
for feat, (min_val, max_val, default) in feature_ranges.items():
    inputs[feat] = st.sidebar.number_input(f"{feat} ({min_val}, {max_val})", 
                                           value=float(default), 
                                           min_value=min_val, 
                                           max_value=max_val, 
                                           format="%.3f")

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# -------------------------
# 4️⃣ 预测逻辑
# -------------------------
if predict_btn:
    X_user = pd.DataFrame([inputs])
    # 按训练特征顺序
    X_user = X_user[feature_cols + ['Antibiotic']]
    # 编码
    X_user_enc = encoder.transform(X_user)
    # 预测
    pred = model.predict(X_user_enc)[0]

    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

    # 仪表盘
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
