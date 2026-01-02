# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# =======================
# 1️⃣ 定义 TargetEncoderCV
# =======================
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
                import numpy as np
                X_encoded[col] = pd.Series(index=X_encoded.index, dtype=float)
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=self.n_splits)
                for train_idx, val_idx in gkf.split(X, y, groups):
                    mapping = y.iloc[train_idx].groupby(X.iloc[train_idx][col]).mean()
                    X_encoded.iloc[val_idx, X_encoded.columns.get_loc(col)] = X.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:
                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# =======================
# 2️⃣ 加载模型
# =======================
bundle = joblib.load("xgb_pipeline.joblib")
model = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

# =======================
# 3️⃣ 页面布局
# =======================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

# =======================
# 4️⃣ 特征范围与默认值（训练平均值）
# =======================
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

# 分类特征
ANTIBIOTIC_LIST = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox("Type of Antibiotic", ANTIBIOTIC_LIST)

# 数值特征
for feat, (min_val, max_val, default) in feature_ranges.items():
    # 确保 float 类型一致，避免 MixedNumericTypesError
    inputs[feat] = st.sidebar.number_input(
        f"{feat} ({min_val}, {max_val})",
        value=float(default),
        min_value=float(min_val),
        max_value=float(max_val),
        format="%.3f"
    )

# =======================
# 5️⃣ 预测按钮
# =======================
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# =======================
# 6️⃣ 预测逻辑
# =======================
if predict_btn:
    # 构造 DataFrame
    X_user = pd.DataFrame([inputs])

    # 保证列顺序一致，并过滤不存在的列
    all_cols = [col for col in feature_cols + ['Antibiotic'] if col in X_user.columns]
    X_user = X_user[all_cols]

    # 编码
    X_user_enc = encoder.transform(X_user)

    # 预测
    pred = model.predict(X_user_enc)[0]
    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}%`")

    # 仪表盘
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Degradation rate (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkgreen"},
               'steps': [{'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 100], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75,
                             'value': pred}}
    ))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please enter the parameters on the left and click Predict.")

st.markdown("---")
st.markdown("*This system uses a unified machine learning pipeline to ensure consistent preprocessing and prediction.*")
