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
    model = joblib.load("xgb_best.pkl")

    # 读取抗生素 one-hot 映射（缩写 -> "1000000000"）
    with open("antibiotic_onehot_map.json", "r", encoding="utf-8") as f:
        raw_map = json.load(f)

    # 兼容是否包了一层 "encoded"
    if "encoded" in raw_map:
        antibiotic_map = raw_map["encoded"]
    else:
        antibiotic_map = raw_map

    # 读取训练时使用的特征列
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, antibiotic_map, feature_columns


model, antibiotic_map, feature_columns = load_pipeline()

# 抗生素 one-hot 列（顺序必须与训练一致）
antibiotic_onehot_cols = [
    c for c in feature_columns if c.startswith("Antibiotic_")
]

# -------------------------------
# Page layout
# -------------------------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

# -------------------------------
# Sidebar inputs
# -------------------------------
inputs = {}

# 显示抗生素缩写（ERY, SM, ...）
inputs["Antibiotic"] = st.sidebar.selectbox(
    "Antibiotic",
    options=sorted(antibiotic_map.keys())
)

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

inputs["pH"]                  = num_input("pH", 2.0, 12.0, 6.08)
inputs["Water content(%)"]    = num_input("Water content(%)", 5.35, 98.1, 69.9)
inputs["m(g)"]                = num_input("m(g)", 1.0, 500.0, 79.36)
inputs["T(°C)"]               = num_input("T(°C)", 0.0, 340.0, 117.8)
inputs["V(L)"]                = num_input("V(L)", 0.05, 1.0, 0.23)
inputs["t(min)"]              = num_input("t(min)", 0.0, 480.0, 64.59)
inputs["HCL Conc (mol/L)"]    = num_input("HCL Conc (mol/L)", 0.0, 0.6, 0.06)
inputs["NaOH Conc (mol/L)"]   = num_input("NaOH Conc (mol/L)", 0.0, 0.6, 0.01)

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# -------------------------------
# Prediction
# -------------------------------
if predict_btn:
    # 初始化特征矩阵（与训练完全一致）
    X = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    # 抗生素 one-hot 填充
    onehot_str = antibiotic_map[inputs["Antibiotic"]]  # e.g. "1000000000"
    for col, bit in zip(antibiotic_onehot_cols, onehot_str):
        X.loc[0, col] = int(bit)

    # 连续变量
    X.loc[0, "pH"]                  = inputs["pH"]
    X.loc[0, "Water content (%)"]   = inputs["Water content(%)"]
    X.loc[0, "m (g)"]               = inputs["m(g)"]
    X.loc[0, "T (°C)"]              = inputs["T(°C)"]
    X.loc[0, "V (L)"]               = inputs["V(L)"]
    X.loc[0, "t (min)"]             = inputs["t(min)"]
    X.loc[0, "Acid Conc (mol/L)"]   = inputs["HCL Conc (mol/L)"]
    X.loc[0, "Alkali Conc (mol/L)"] = inputs["NaOH Conc (mol/L)"]

    # 预测
    pred = model.predict(X.values)[0]
    pred_percent = pred * 100

    # 展示结果
    st.markdown(
        f"### ✅ Predicted Degradation rate: **{pred_percent:.2f}%**"
    )

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

    st.caption(
        "*This model is applicable only to the experimental systems covered by the present database. "
        "For predictions in other independent systems, retraining the model using data from the corresponding system "
        "is recommended to achieve optimal performance.*"
    )
