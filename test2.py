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
st.set_page_config(page_title="降解率预测", layout="centered")
st.title("🧪 降解率预测系统")
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
feat_cols_cn = ['抗生素类型', 'pH', '含水率(%)', '菌渣质量(g)', '反应温度(°C)',
                '升温速率(°C/min)', '反应器体积(L)', '反应时间(min)', '酸浓度(uM)']

# ---------- 侧边栏输入 ----------
st.sidebar.header("请输入参数")
inputs = {}
for col, col_cn in zip(feat_cols, feat_cols_cn):
    inputs[col] = st.sidebar.number_input(col_cn, value=0.0, format="%.3f")

btn = st.sidebar.button("🔍 预测降解率")

# ---------- 主界面 ----------
if btn:
    # 构造 DataFrame
    X_user = pd.DataFrame([inputs])
    pred = model.predict(X_user)[0]

    # 显示结果
    st.markdown(f"### 预测降解率： `{pred:.3f}`")

    # 仪表盘
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "降解率", 'font': {'size': 24}},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "darkgreen"},
               'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                         {'range': [0.5, 1], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': pred}}))
    st.plotly_chart(fig_gauge, use_container_width=True)


else:
    st.info("请在左侧栏输入参数后点击预测按钮")