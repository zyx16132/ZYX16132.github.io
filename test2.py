# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import plotly.graph_objects as go
import seaborn as sns

plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# ---------- 页面配置 ----------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ---------- 加载模型 ----------
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("xgb_pen.json")  # 确保模型文件在同一目录
    return model

model = load_model()
explainer = shap.TreeExplainer(model)

# ---------- 中文特征名 ----------
feat_cols = ['Class', 'pH', 'Water content(%)', 'm(g)', 'T(°C)',
             'HR(°C/min)', 'V(L)', 't(min)', 'Conc(mol/L)']
feat_cols_cn = ['Types of antibiotics', 'Initial environmental pH', 'Water content(%)', 'Quality(g)', 'Reaction temperature(°C)',
                'Heating rate(°C/min)', 'Reactor volume(L)', 'Reaction time(min)', 'Acid concentration(mol/L)']

# ---------- 侧边栏输入 ----------
st.sidebar.header("Please enter parameters")
inputs = {}
for col, col_cn in zip(feat_cols, feat_cols_cn):
    inputs[col] = st.sidebar.number_input(col_cn, value=0.0, format="%.3f")

btn = st.sidebar.button("🔍 Predict degradation rate")

# ---------- 主界面 ----------
if btn:
    # 构造 DataFrame
    X_user = pd.DataFrame([inputs])
    pred = model.predict(X_user)[0]

    # 显示结果
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
