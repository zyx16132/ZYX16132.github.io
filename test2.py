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
# 加载模型 + encoder（已训练好）
# ======================================================
@st.cache_resource
def load_model_and_encoder():
    model = joblib.load("xgb_best.pkl")      # 你的 XGB 模型
    encoder = joblib.load("encoder.pkl")     # TargetEncoderCV
    return model, encoder

try:
    model, encoder = load_model_and_encoder()
except Exception as e:
    st.error(f"❌ Model or encoder loading failed:\n\n{e}")
    st.stop()

# ======================================================
# ⚠️ 必须与模型训练时完全一致的特征顺序
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
    'Degradation',   # ⚠️ 占位列（必须）
    'Antibiotic'
]

# 页面显示名称
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

# Antibiotic 下拉框
inputs['Antibiotic'] = st.sidebar.selectbox(
    LABELS['Antibiotic'],
    list(encoder.mapping_['Antibiotic'].index)
)

# 数值输入
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
        LABELS[k], value=float(v), format="%.3f"
    )

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# ======================================================
# 预测
# ======================================================
if predict_btn:
    try:
        # 构建 DataFrame
        X_user = pd.DataFrame([inputs])

        # 🔑 补占位 Degradation
        X_user['Degradation'] = 0.0

        # 🔑 按训练顺序重排
        X_user = X_user[MODEL_FEATURES]

        # 编码 + 预测
        X_user_enc = encoder.transform(X_user)
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
    st.info("Please enter parameters on the left and click Predict.")
