import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==============================
# 1️⃣ 加载训练好的 pipeline
# ==============================
@st.cache_resource
def load_pipeline():
    bundle = joblib.load("xgb_pipeline_no_class.joblib")
    return bundle

bundle = load_pipeline()
model = bundle["model"]
encoder_mapping = bundle["encoder_mapping"]
feature_cols = bundle["feature_cols"]  # 原始数值特征列
cat_cols = bundle["cat_cols"]          # 分类列，如 ['Antibiotic']

# ==============================
# 2️⃣ 页面布局
# ==============================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("请输入参数")

# ==============================
# 3️⃣ 用户友好的显示名映射到训练列名
# ==============================
display_name_map = {
    "pH值": "pH",
    "水分(%)": "Water content(%)",
    "物质质量(g)": "m(g)",
    "温度(°C)": "T(°C)",
    "体积(L)": "V(L)",
    "时间(min)": "t(min)",
    "盐酸浓度(mol/L)": "HCL Conc(mol/L)",
    "氢氧化钠浓度(mol/L)": "NaOH Conc(mol/L)",
    "抗生素类型": "Antibiotic"
}

# 可以随意顺序展示
user_input_order = [
    "抗生素类型", "pH值", "水分(%)", "物质质量(g)", "温度(°C)",
    "体积(L)", "时间(min)", "盐酸浓度(mol/L)", "氢氧化钠浓度(mol/L)"
]

# ==============================
# 4️⃣ 数值输入范围
# ==============================
feature_ranges = {
    "pH值": (2.0, 12.0, 6.08),
    "水分(%)": (5.35, 98.1, 69.9),
    "物质质量(g)": (1.0, 500.0, 79.36),
    "温度(°C)": (0.0, 340.0, 117.8),
    "体积(L)": (0.05, 1.0, 0.23),
    "时间(min)": (0.0, 480.0, 64.59),
    "盐酸浓度(mol/L)": (0.0, 0.6, 0.06),
    "氢氧化钠浓度(mol/L)": (0.0, 0.6, 0.01)
}

inputs = {}

# ==============================
# 5️⃣ 分类输入
# ==============================
for disp_name, col_name in display_name_map.items():
    if col_name in cat_cols:
        options = list(encoder_mapping[col_name].keys())
        inputs[col_name] = st.sidebar.selectbox(disp_name, options)

# ==============================
# 6️⃣ 数值输入
# ==============================
for disp_name in user_input_order:
    col_name = display_name_map[disp_name]
    if col_name not in cat_cols:  # 数值列
        min_val, max_val, default = feature_ranges[disp_name]
        inputs[col_name] = st.sidebar.number_input(
            label=disp_name,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            format="%.3f"
        )

# ==============================
# 7️⃣ Predict 按钮
# ==============================
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

if predict_btn:
    # 构建 DataFrame
    X_user = pd.DataFrame([inputs])

    # 分类映射
    for cat in cat_cols:
        X_user[cat] = X_user[cat].map(encoder_mapping[cat])
        if X_user[cat].isna().any():
            X_user[cat] = X_user[cat].fillna(np.mean(list(encoder_mapping[cat].values())))

    # 严格按训练列顺序排列
    X_user_final = X_user[feature_cols + cat_cols]

    # 预测
    pred = model.predict(X_user_final)[0]

    # 显示
    st.markdown(f"### ✅ Predicted Degradation rate: **{pred:.2f}%**")

    # 仪表盘
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"suffix": "%"},
        title={"text": "Degradation rate"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "darkgreen"},
               "steps": [
                   {"range": [0, 50], "color": "#f2f2f2"},
                   {"range": [50, 100], "color": "#c7e9c0"}
               ]},
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("请在左侧输入参数并点击 Predict.")

st.markdown("---")
st.markdown("*该应用使用训练好的 XGBoost 模型及相同的 Target Encoding。*")
