# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================
# ✅ 关键：必须重新定义 TargetEncoderCV（用于反序列化）
# ======================================================
class TargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None, n_splits=5, random_state=42):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X, y=None, groups=None):
        return self

    def transform(self, X):
        X_enc = X.copy()
        for col in self.mapping_:
            if col in X_enc.columns:
                X_enc[col] = (
                    X_enc[col]
                    .map(self.mapping_[col])
                    .fillna(self.global_mean_)
                )
        return X_enc


# ======================================================
# 页面配置
# ======================================================
st.set_page_config(
    page_title="Degradation rate prediction",
    layout="centered"
)

st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ======================================================
# 加载模型 & encoder
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load("xgb_best.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("encoder.pkl")

try:
    model = load_model()
    encoder = load_encoder()
except Exception as e:
    st.error(f"❌ Model or encoder loading failed:\n\n{e}")
    st.stop()

# ======================================================
# 特征列（必须与训练一致）
# ======================================================
FEATURE_COLS = [
    'Antibiotic',
    'pH',
    'Water content(%)',
    'm(g)',
    'T(°C)',
    'V(L)',
    't(min)',
    'HCL Conc(mol/L)',
    'NaOH Conc(mol/L)'
]

FEATURE_LABELS = [
    'Type of Antibiotic',
    'Initial environmental pH [2,12]',
    'Water content (%) [5.35,98.1]',
    'Quality (g) [1,500]',
    'Reaction temperature (°C) [0,340]',
    'Reactor volume (L) [0.05,1]',
    'Reaction time (min) [0,480]',
    'HCL concentration (mol/L) [0,0.6]',
    'NaOH concentration (mol/L) [0,0.6]'
]

# ======================================================
# 侧边栏输入
# ======================================================
st.sidebar.header("Please enter parameters")
inputs = {}

# Antibiotic 下拉框
antibiotic_options = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox(
    FEATURE_LABELS[0],
    antibiotic_options
)

default_values = {
    'pH': 6.08,
    'Water content(%)': 69.9,
    'm(g)': 79.36,
    'T(°C)': 117.8,
    'V(L)': 0.23,
    't(min)': 64.59,
    'HCL Conc(mol/L)': 0.06,
    'NaOH Conc(mol/L)': 0.01
}

for col, label in zip(FEATURE_COLS[1:], FEATURE_LABELS[1:]):
    inputs[col] = st.sidebar.number_input(
        label,
        value=float(default_values[col]),
        format="%.3f"
    )

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# ======================================================
# 预测
# ======================================================
if predict_btn:
    try:
        X_user = pd.DataFrame([inputs], columns=FEATURE_COLS)

        # ✅ 编码 Antibiotic
        X_user_enc = encoder.transform(X_user)

        # ✅ 预测
        pred = model.predict(X_user_enc)[0]

        st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Degradation rate"},
            gauge={'axis': {'range': [0, 1]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed:\n\n{e}")
else:
    st.info("Please enter the parameters on the left and click the prediction button.")
