import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go

# -------------------------------
# Load model and metadata
# -------------------------------
@st.cache_resource
def load_pipeline():
    # 加载训练好的模型
    model = joblib.load("xgb_best.pkl")

    # 读取抗生素映射（缩写 -> 编码整数）
    with open("antibiotic_encoded_map.json", "r", encoding="utf-8") as f:
        antibiotic_map = json.load(f)

    # 读取训练时的特征列
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, antibiotic_map, feature_columns

model, antibiotic_map, feature_columns = load_pipeline()

# -------------------------------
# 页面布局
# -------------------------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

# -------------------------------
# Sidebar 输入
# -------------------------------
inputs = {}

# 抗生素缩写选择
inputs["Antibiotic"] = st.sidebar.selectbox(
    "Antibiotic",
    options=sorted(antibiotic_map.keys())
)

# 数值输入函数
def num_input(label, vmin, vmax, default):
    label_with_range = f"{label} ({vmin}–{vmax})"
    return st.sidebar.number_input(
        label=label_with_range,
        min_value=float(vmin),
        max_value=float(vmax),
        value=float(default),
        step=0.001,
        format="%.3f"
    )

# 其他特征
inputs["pH"]                = num_input("pH", 2.0, 12.0, 6.08)
inputs["Water content(%)"]  = num_input("Water content(%)", 5.35, 98.1, 69.9)
inputs["m(g)"]              = num_input("m(g)", 1.0, 500.0, 79.36)
inputs["T(°C)"]             = num_input("T(°C)", 0.0, 340.0, 117.8)
inputs["V(L)"]              = num_input("V(L)", 0.05, 1.0, 0.23)
inputs["t(min)"]            = num_input("t(min)", 0.0, 480.0, 64.59)
inputs["HCL Conc (mol/L)"]  = num_input("HCL Conc (mol/L)", 0.0, 0.6, 0.06)
inputs["NaOH Conc (mol/L)"] = num_input("NaOH Conc (mol/L)", 0.0, 0.6, 0.01)

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# -------------------------------
# Prediction
# -------------------------------
if predict_btn:
    # 初始化特征矩阵
    X = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    # 填充抗生素编码
    X.loc[0, "Antibiotic_encoded"] = antibiotic_map[inputs["Antibiotic"]]

    # 填充数值特征
    X.loc[0, "pH"]                  = inputs["pH"]
    X.loc[0, "Water content(%)"]   = inputs["Water content(%)"]
    X.loc[0, "m(g)"]               = inputs["m(g)"]
    X.loc[0, "T(°C)"]              = inputs["T(°C)"]
    X.loc[0, "V(L)"]               = inputs["V(L)"]
    X.loc[0, "t(min)"]             = inputs["t(min)"]
    X.loc[0, "Acid Conc(mol/L)"]   = inputs["HCL Conc (mol/L)"]
    X.loc[0, "Alkali Conc(mol/L)"] = inputs["NaOH Conc (mol/L)"]

    # 模型预测
    pred = model.predict(X)[0]
    pred_percent = pred * 100

    # 显示结果
    st.markdown(f"### ✅ Predicted Degradation rate: **{pred_percent:.2f}%**")

    # Gauge 图
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_percent,
        number={"suffix": "%"},
        title={"text": "Degradation rate"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkgreen"},
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 说明文字
    st.caption(
        "*This model is applicable only to the experimental systems covered by the present database. "
        "For predictions in other independent systems, retraining the model using data from the corresponding system "
        "is recommended to achieve optimal performance.*"
    )
