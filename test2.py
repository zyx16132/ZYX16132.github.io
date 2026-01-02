# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# ---------------- Streamlit 页面 ----------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ---------- 加载 pipeline ----------
@st.cache_resource
def load_pipeline():
    return joblib.load("xgb_pipeline_groupCV.pkl")

try:
    pipe = load_pipeline()
except Exception as e:
    st.error(f"Pipeline loading failed: {e}")
    st.stop()

# ---------- 特征名（⚠ 必须与训练时一致） ----------
feat_cols = [
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

feat_cols_cn = [
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

# ---------- 侧边栏输入 ----------
st.sidebar.header("Please enter parameters")

inputs = {}

# ✅ Antibiotic 直接从 pipeline encoder 中读取
encoder = pipe.named_steps['encoder']
antibiotics_list = sorted(encoder.mapping_['Antibiotic'].index.tolist())

inputs['Antibiotic'] = st.sidebar.selectbox(
    feat_cols_cn[0],
    antibiotics_list
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

for col, col_cn in zip(feat_cols[1:], feat_cols_cn[1:]):
    inputs[col] = st.sidebar.number_input(
        col_cn,
        value=float(default_values[col]),
        format="%.3f"
    )

btn = st.sidebar.button("🔍 Predict degradation rate")

# ---------- 主界面 ----------
if btn:
    try:
        # ✅ 严格按训练顺序构建 DataFrame
        X_user = pd.DataFrame([[inputs[c] for c in feat_cols]], columns=feat_cols)

        # 预测
        pred = pipe.predict(X_user)[0]

        st.markdown(f"### Predicted Degradation rate: `{pred:.3f}`")

        # 仪表盘
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Degradation rate"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 1], 'color': "lightgreen"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Please enter the parameters in the left column and click the prediction button")
