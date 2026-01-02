import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------
# 1️⃣ 加载 pipeline
# ---------------------------
bundle = joblib.load("xgb_pipeline_no_class.joblib")
model = bundle["model"]
encoder_mapping = bundle["encoder_mapping"]
feature_cols = bundle["feature_cols"]
cat_cols = bundle["cat_cols"]

# ---------------------------
# 2️⃣ 页面配置 & 输入
# ---------------------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.sidebar.header("Please enter parameters")

inputs = {}

# 分类特征选择
inputs['Antibiotic'] = st.sidebar.selectbox("Type of Antibiotic", list(encoder_mapping.keys()))

# 数值特征输入
feature_ranges = {
    'pH': (2.0, 12.0, 6.08),
    'Water content(%)': (5.35, 98.1, 69.9),
    'm(g)': (1.0, 500.0, 79.36),
    'T(°C)': (0.0, 340.0, 117.8),
    'V(L)': (0.05, 1.0, 0.23),
    't(min)': (0.0, 480.0, 64.59),
    'HCL Conc(mol/L)': (0.0, 0.6, 0.06),
    'NaOH Conc(mol/L)': (0.0, 0.6, 0.01)
}

for feat in feature_cols:
    min_val, max_val, default = feature_ranges[feat]
    inputs[feat] = st.sidebar.number_input(
        feat, min_value=float(min_val), max_value=float(max_val), value=float(default), format="%.3f"
    )

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# ---------------------------
# 3️⃣ 预测逻辑
# ---------------------------
if predict_btn:
    X_user = pd.DataFrame([inputs])
    # 手动 Target Encoding：用保存的 mapping
    for col in cat_cols:
        X_user[col] = X_user[col].map(encoder_mapping[col]).fillna(np.mean(list(encoder_mapping[col].values())))
    # 确保列顺序一致
    X_user = X_user[feature_cols + cat_cols]
    
    pred = model.predict(X_user)[0]

    st.markdown(f"### ✅ Predicted degradation rate: **{pred:.2f}%**")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"suffix": "%"},
        title={"text": "Degradation rate"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkgreen"}}
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter the parameters on the left and click Predict.")
