# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 🔥 只从 model.py 读取“已训练好的”对象
import model


# ======================================================
# 1️⃣ 页面配置
# ======================================================
st.set_page_config(
    page_title="Degradation rate prediction",
    layout="centered"
)

st.title("🧪 Degradation rate prediction system")
st.markdown("---")


# ======================================================
# 2️⃣ 特征顺序（⚠️ 必须与 model.py 训练时完全一致）
# ======================================================
MODEL_FEATURES = [
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


# ======================================================
# 3️⃣ 左侧输入栏
# ======================================================
st.sidebar.header("Please enter parameters")

# —— 抗生素（来自 encoder，不允许手写）
ANTIBIOTIC_LIST = list(model.encoder.mapping_['Antibiotic'].index)

antibiotic = st.sidebar.selectbox(
    "Type of Antibiotic",
    ANTIBIOTIC_LIST
)

# —— 数值参数（完全不改你的默认值）
pH = st.sidebar.number_input(
    "Initial environmental pH [2–12]",
    value=6.080,
    format="%.3f"
)

water = st.sidebar.number_input(
    "Water content (%) [5.35–98.1]",
    value=69.900,
    format="%.3f"
)

m = st.sidebar.number_input(
    "Quality (g) [1–500]",
    value=79.360,
    format="%.3f"
)

T = st.sidebar.number_input(
    "Reaction temperature (°C) [0–340]",
    value=117.800,
    format="%.3f"
)

V = st.sidebar.number_input(
    "Reactor volume (L) [0.05–1]",
    value=0.230,
    format="%.3f"
)

t = st.sidebar.number_input(
    "Reaction time (min) [0–480]",
    value=64.590,
    format="%.3f"
)

hcl = st.sidebar.number_input(
    "HCL concentration (mol/L) [0–0.6]",
    value=0.060,
    format="%.3f"
)

naoh = st.sidebar.number_input(
    "NaOH concentration (mol/L) [0–0.6]",
    value=0.010,
    format="%.3f"
)

predict_btn = st.sidebar.button("🔍 Predict degradation rate")


# ======================================================
# 4️⃣ 预测逻辑（⚠️ 只 transform + predict）
# ======================================================
if predict_btn:

    # —— 构造用户输入（⚠️ 不包含 Degradation）
    X_user = pd.DataFrame([{
        'pH': pH,
        'Water content(%)': water,
        'm(g)': m,
        'T(°C)': T,
        'V(L)': V,
        't(min)': t,
        'HCL Conc(mol/L)': hcl,
        'NaOH Conc(mol/L)': naoh,
        'Antibiotic': antibiotic
    }])

    # —— 确保顺序一致
    X_user = X_user[MODEL_FEATURES]

    # —— 使用训练阶段的 encoder
    X_user_enc = model.encoder.transform(X_user)

    # —— 使用训练阶段的 best_xgb
    pred = model.best_xgb.predict(X_user_enc)[0]

    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

    # 仪表盘
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Degradation rate"},
            gauge={'axis': {'range': [0, 100]}}
        )
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter parameters on the left and click Predict.")


st.markdown("---")
st.markdown(
    "*This system uses a unified machine learning pipeline to ensure consistent preprocessing and prediction.*"
)
