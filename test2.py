import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ---------- 加载 pipeline ----------
@st.cache_resource
def load_pipeline():
    pipe = joblib.load("xgb_pipeline_groupCV.pkl")
    return pipe

pipe = load_pipeline()

# ---------- 特征名 ----------
feat_cols = ['Antibiotic', 'pH', 'Water content(%)', 'm(g)', 'T(°C)',
             'V(L)', 't(min)', 'HCL Conc(mol/L)', 'NaOH Conc(mol/L)']
feat_cols_cn = ['Type of Antibiotic', 'Initial environmental pH', 'Water content(%)',
                'Quality(g)', 'Reaction temperature(°C)', 'Reactor volume(L)',
                'Reaction time(min)', 'HCL concentration(mol/L)', 'NaOH concentration(mol/L)']

# ---------- 侧边栏输入 ----------
st.sidebar.header("Please enter parameters")
inputs = {}

# 自动获取 Antibiotic 类别
antibiotics_list = list(pipe.named_steps['encoder'].mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox(feat_cols_cn[0], antibiotics_list)

# 数值默认值
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
    inputs[col] = st.sidebar.number_input(col_cn, value=default_values[col], format="%.3f")

btn = st.sidebar.button("🔍 Predict degradation rate")

# ---------- 主界面 ----------
if btn:
    X_user = pd.DataFrame([inputs])
    pred = pipe.predict(X_user)[0]
    st.markdown(f"### Predicted Degradation rate: `{pred:.3f}`")

    # 仪表盘显示
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Degradation rate", 'font': {'size': 24}},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "darkgreen"},
               'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                         {'range': [0.5, 1], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': pred}}))
    st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("Please enter the parameters in the left column and click the prediction button")
