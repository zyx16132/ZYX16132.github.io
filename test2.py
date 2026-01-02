# ======================================================
# 0️⃣ 反序列化占位：TargetEncoderCV（必须最先定义）
# ======================================================
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """
    ⚠️ 这是一个“反序列化占位类”
    作用：仅用于 joblib.load 时让 pickle 找到类定义
    ⚠️ 不会重新 fit，不会改变任何预测结果
    """

    def __init__(self, cat_cols=None, n_splits=5, random_state=42):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = None
        self.mapping_ = {}

    def fit(self, X, y=None, groups=None):
        return self

    def transform(self, X, y=None, groups=None):
        X_out = X.copy()
        for col, mapping in self.mapping_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].map(mapping).fillna(self.global_mean_)
        return X_out


# ======================================================
# 1️⃣ 正常 imports
# ======================================================
import streamlit as st
import plotly.graph_objects as go
import joblib

# ======================================================
# 2️⃣ 页面配置
# ======================================================
st.set_page_config(
    page_title="Degradation rate prediction",
    layout="centered"
)

st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ======================================================
# 3️⃣ 加载模型 Pipeline
# ======================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("xgb_pipeline_groupCV.pkl")

try:
    pipe = load_pipeline()
except Exception as e:
    st.error("❌ Model pipeline loading failed")
    st.exception(e)
    st.stop()

st.success("✅ Model pipeline loaded successfully")

# ======================================================
# 4️⃣ 特征定义（名称必须与训练一致）
# ======================================================
FEATURES = [
    'pH',
    'Water content(%)',
    'm(g)',
    'T(°C)',
    'V(L)',
    't(min)',
    'HCL Conc(mol/L)',
    'NaOH Conc(mol/L)',
    'Antibiotic'
]

LABELS = {
    'Antibiotic': 'Type of Antibiotic',
    'pH': 'Initial environmental pH [2–12]',
    'Water content(%)': 'Water content (%) [5.35–98.1]',
    'm(g)': 'Quality (g) [1–500]',
    'T(°C)': 'Reaction temperature (°C) [0–340]',
    'V(L)': 'Reactor volume (L) [0.05–1]',
    't(min)': 'Reaction time (min) [0–480]',
    'HCL Conc(mol/L)': 'HCL concentration (mol/L) [0–0.6]',
    'NaOH Conc(mol/L)': 'NaOH concentration (mol/L) [0–0.6]'
}

# ======================================================
# 5️⃣ 侧边栏输入
# ======================================================
st.sidebar.header("Please enter parameters")

inputs = {}

inputs['Antibiotic'] = st.sidebar.text_input(
    LABELS['Antibiotic'],
    value="TC"
)

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
    inputs[k] = st.sidebar.number_input(
        LABELS[k],
        value=float(v),
        format="%.3f"
    )

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# ======================================================
# 6️⃣ 预测
# ======================================================
if predict_btn:
    try:
        X_user = pd.DataFrame([inputs])

        # Pipeline 自动完成：TargetEncoding → XGB → 预测
        pred = pipe.predict(X_user)[0]

        st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Degradation rate"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

else:
    st.info("Please enter parameters on the left and click Predict.")

# ======================================================
# 7️⃣ 页脚
# ======================================================
st.markdown("---")
st.markdown(
    "*This system uses a unified machine learning pipeline to ensure consistent preprocessing and prediction.*"
)
