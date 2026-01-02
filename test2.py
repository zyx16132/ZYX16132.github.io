# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ======================================================
# 1️⃣ 加载训练好的模型 pipeline
# ======================================================
bundle = joblib.load("xgb_pipeline.joblib")
best_xgb = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

# ======================================================
# 2️⃣ 页面设置
# ======================================================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ======================================================
# 3️⃣ 输入界面
# ======================================================
# 抗生素下拉
antibiotic_list = list(encoder.mapping_['Antibiotic'].index)
antibiotic = st.sidebar.selectbox("Type of Antibiotic", antibiotic_list)

# 数值输入（默认值可按你的数据修改）
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

inputs = {}
for k, v in defaults.items():
    inputs[k] = st.sidebar.number_input(k, value=float(v), format="%.3f")

# ======================================================
# 4️⃣ 预测按钮
# ======================================================
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

if predict_btn:
    # 构造输入 DataFrame
    df_input = pd.DataFrame([inputs])
    df_input['Antibiotic'] = antibiotic
    df_input['Degradation'] = 0.0  # 占位
    df_input = df_input[feature_cols]  # 保证特征顺序
    
    # TargetEncoder 编码
    df_input_enc = encoder.transform(df_input)
    
    # 模型预测
    pred = best_xgb.predict(df_input_enc)[0]

    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")
    
    # 可视化
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Degradation rate"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter parameters on the left and click Predict.")

st.markdown("---")
st.markdown("*This system uses a pre-trained machine learning model for consistent prediction.*")
