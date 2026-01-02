# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# ======================================================
# 页面配置
# ======================================================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ======================================================
# 加载 Pipeline（唯一需要的模型文件）
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
# ⚠️ 必须与训练时特征名一致（但不需要顺序）
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
    'pH': 'Initial environmental pH [2,12]',
    'Water content(%)': 'Water content (%) [5.35,98.1]',
    'm(g)': 'Quality (g) [1,500]',
    'T(°C)': 'Reaction temperature (°C) [0,340]',
    'V(L)': 'Reactor volume (L) [0.05,1]',
    't(min)': 'Reaction time (min) [0,480]',
    'HCL Conc(mol/L)': 'HCL concentration (mol/L) [0,0.6]',
    'NaOH Conc(mol/L)': 'NaOH concentration (mol/L) [0,0.6]'
}

# ======================================================
# 侧边栏输入
# ======================================================
st.sidebar.header("Please enter parameters")
inputs = {}

# ⚠️ Antibiotic 直接文本输入（最稳）
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
# 预测
# ======================================================
if predict_btn:
    try:
        X_user = pd.DataFrame([inputs])

        # 🔑 Pipeline 会自动完成编码 + 预测
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
