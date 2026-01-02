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

# ======================================================
# 加载模型、编码器和特征列
# ======================================================
bundle = joblib.load("xgb_pipeline.joblib")
model = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

# ======================================================
# 读取原始数据用于计算 min/max/mean
# ======================================================
df = pd.read_excel("data.xlsx")

# 数值特征
num_features = feature_cols.copy()
categorical_features = ['Antibiotic']

# ======================================================
# Streamlit 页面设置
# ======================================================
st.set_page_config(
    page_title="Degradation rate prediction",
    layout="centered"
)

st.title("🧪 Degradation rate prediction system")

# ======================================================
# Sidebar 输入
# ======================================================
st.sidebar.header("Please enter parameters")

inputs = {}

for feat in num_features:
    min_val = df[feat].min()
    max_val = df[feat].max()
    default = df[feat].mean()
    inputs[feat] = st.sidebar.number_input(
        f"{feat} ({min_val:.3f}, {max_val:.3f})",
        value=float(default),
        min_value=float(min_val),
        max_value=float(max_val),
        format="%.3f"
    )

# 分类特征选择
for feat in categorical_features:
    unique_vals = df[feat].unique().tolist()
    default = unique_vals[0]
    inputs[feat] = st.sidebar.selectbox(
        f"Type of {feat}",
        options=unique_vals,
        index=0
    )

# ======================================================
# 准备单样本预测
# ======================================================
X_user = pd.DataFrame([inputs])

# 只对数值+Antibiotic列进行编码
X_user_enc = encoder.transform(X_user)

# 预测
pred = model.predict(X_user_enc)[0]

# ======================================================
# 显示结果（居中 + 仪表盘）
# ======================================================
st.subheader("Predicted Degradation Rate (%)")
st.metric(label="Degradation Rate", value=f"{pred:.2f}")

# 可选仪表盘显示
st.write("### Gauge-style visualization")
st.markdown(
    f"""
    <div style="display:flex; justify-content:center;">
        <progress value="{pred}" max="100" style="width:60%; height:30px;"></progress>
    </div>
    """,
    unsafe_allow_html=True
)
