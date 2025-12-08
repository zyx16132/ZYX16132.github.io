# app.py
import streamlit as st
import pandas as pd
import shap
import plotly.graph_objects as go
import joblib

# ---------- 页面配置 ----------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ---------- 加载模型和 SHAP Explainer ----------
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("xgb_best.pkl")
    # 这里暂时不传 X 给 explainer，预测时再用 shap.Explainer
    return model

model = load_model_and_explainer()

# ---------- 特征名 ----------
feat_cols = ['Class', 'pH', 'Water content(%)', 'm(g)', 'T(°C)',
             'HR(°C/min)', 'V(L)', 't(min)', 'Conc(mol/L)']
feat_cols_cn = ['Types of antibiotics（Take an integer from [0, 9]）', 'Initial environmental pH [4.6,7.5]', 'Water content(%) [5.35,95.93]',
                'Quality(g) [1,300]', 'Reaction temperature(°C) [22,250]', 'Heating rate(°C/min) [0.19,14]',
                'Reactor volume(L) [0.05,1]', 'Reaction time(min) [0,180]', 'Acid concentration(mol/L) [0,0.6]']

# ---------- 侧边栏输入 ----------
st.sidebar.header("Please enter parameters")
inputs = {}
for col, col_cn in zip(feat_cols, feat_cols_cn):
    inputs[col] = st.sidebar.number_input(col_cn, value=0.0, format="%.3f")

btn = st.sidebar.button("🔍 Predict degradation rate")

# ---------- 主界面 ----------
if btn:
    X_user = pd.DataFrame([inputs])
    pred = model.predict(X_user)[0]
    st.markdown(f"### Predict degradation rate： `{pred:.3f}`")

    # 仪表盘
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "degradation rate", 'font': {'size': 24}},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "darkgreen"},
               'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                         {'range': [0.5, 1], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': pred}}))
    st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("Please enter the parameters in the left column and click the prediction button")
