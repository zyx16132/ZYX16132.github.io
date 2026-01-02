# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

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
# 特征列（必须与训练完全一致）
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

# ---------- Antibiotic ----------
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
# 预测
# ======================================================
if predict_btn:
    try:
        # 构建 DataFrame
        X_user = pd.DataFrame([inputs], columns=FEATURE_COLS)

        # === 关键一步：Antibiotic 编码 ===
        X_user_encoded = encoder.transform(X_user)

        # 预测
        pred = model.predict(X_user_encoded)[0]

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
