# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PipelineTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None, n_splits=5, random_state=42):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = None
        self.mapping_ = dict()

    def fit(self, X, y=None, groups=None):
        # ⚠️ 预测阶段不会调用 fit，这里只为 pickle 兼容
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in self.mapping_:
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].map(
                    self.mapping_[col]
                ).fillna(self.global_mean_)
        return X_encoded

# ======================================================
# Streamlit 页面配置
# ======================================================
st.set_page_config(
    page_title="Degradation rate prediction",
    layout="centered"
)

st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ======================================================
# 加载 pipeline（encoder + XGB 已全部包含在内）
# ======================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("xgb_pipeline_groupCV.pkl")

try:
    pipe = load_pipeline()
except Exception as e:
    st.error(f"❌ Pipeline loading failed:\n\n{e}")
    st.stop()

# ======================================================
# 特征列（⚠️ 必须与训练时完全一致）
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

# ---------- Antibiotic 下拉框 ----------
# ⚠️ encoder 已在 pipeline 内，这里只是为了给用户选项
encoder = pipe.named_steps['encoder']
antibiotic_options = list(encoder.mapping_['Antibiotic'].index)

inputs['Antibiotic'] = st.sidebar.selectbox(
    FEATURE_LABELS[0],
    antibiotic_options
)

# ---------- 数值输入 ----------
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
# 主界面预测
# ======================================================
if predict_btn:
    try:
        # 构建严格匹配训练特征顺序的 DataFrame
        X_user = pd.DataFrame([inputs], columns=FEATURE_COLS)

        # ⚠️ 直接用 pipeline.predict
        # encoder + XGB 会自动完成
        pred = pipe.predict(X_user)[0]

        st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

        # ---------- 仪表盘 ----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Degradation rate", 'font': {'size': 22}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': pred
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(
            f"❌ Prediction failed:\n\n{e}\n\n"
            "⚠️ Please make sure inputs match the training features."
        )
else:
    st.info("Please enter the parameters on the left and click the prediction button.")
