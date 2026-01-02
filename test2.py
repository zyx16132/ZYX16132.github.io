# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------- 自定义 TargetEncoderCV ----------------
class TargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, n_splits=5, random_state=42):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = None
        self.mapping_ = dict()

    def fit(self, X, y, groups=None):
        self.global_mean_ = y.mean()
        self.mapping_ = dict()
        for col in self.cat_cols:
            if col in X.columns:
                self.mapping_[col] = y.groupby(X[col]).mean()
            else:
                self.mapping_[col] = pd.Series(dtype=float)
        return self

    def transform(self, X, y=None, groups=None):
        X_encoded = X.copy()
        for col in self.cat_cols:
            if col not in X_encoded.columns:
                continue
            if y is not None and groups is not None:
                # 训练集 CV 安全编码
                X_encoded[col] = pd.NA
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=self.n_splits)
                X_temp, y_temp, groups_temp = X.copy(), y.copy(), groups.copy()
                for train_idx, val_idx in gkf.split(X_temp, y_temp, groups_temp):
                    mapping = y_temp.iloc[train_idx].groupby(X_temp.iloc[train_idx][col]).mean()
                    X_encoded.iloc[val_idx, X_encoded.columns.get_loc(col)] = X_temp.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:
                # 测试集 / 新样本
                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# ---------------- Streamlit 页面配置 ----------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# ---------------- 加载 pipeline ----------------
@st.cache_resource
def load_pipeline():
    # 确保 PipelineTargetEncoder 在此文件中定义，否则 joblib 会报错
    class PipelineTargetEncoder(TargetEncoderCV):
        def transform(self, X, y=None, groups=None):
            X_encoded = super().transform(X, y=y, groups=groups)
            feature_cols = ['Antibiotic', 'pH', 'Water content(%)', 'm(g)',
                            'T(°C)', 'V(L)', 't(min)', 'HCL Conc(mol/L)', 'NaOH Conc(mol/L)']
            return X_encoded[feature_cols]
    pipe = joblib.load("xgb_pipeline_groupCV.pkl")
    return pipe

pipe = load_pipeline()

# ---------------- 特征名 ----------------
feat_cols = ['Antibiotic', 'pH', 'Water content(%)', 'm(g)', 'T(°C)',
             'V(L)', 't(min)', 'HCL Conc(mol/L)', 'NaOH Conc(mol/L)']

feat_cols_cn = ['Type of Antibiotic',
                'Initial environmental pH [2,12]',
                'Water content (%) [5.35,98.1]',
                'Quality (g) [1,500]',
                'Reaction temperature (°C) [0,340]',
                'Reactor volume (L) [0.05,1]',
                'Reaction time (min) [0,480]',
                'HCL concentration (mol/L) [0,0.6]',
                'NaOH concentration (mol/L) [0,0.6]']

# ---------------- 侧边栏输入 ----------------
st.sidebar.header("Please enter parameters")
inputs = {}

encoder = pipe.named_steps['encoder']
antibiotics_list = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox(feat_cols_cn[0], antibiotics_list)

# 默认数值
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

for col, col_cn in zip(feat_cols[1:], feat_cols_cn[1:]):
    inputs[col] = st.sidebar.number_input(col_cn, value=default_values[col], format="%.3f")

btn = st.sidebar.button("🔍 Predict degradation rate")

# ---------------- 主界面 ----------------
if btn:
    try:
        # 构建 DataFrame
        X_user = pd.DataFrame([inputs], columns=feat_cols)

        # 使用 pipeline predict
        pred = pipe.predict(X_user)[0]

        st.markdown(f"### Predicted Degradation rate: `{pred:.3f}`")

        # 仪表盘显示
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Degradation rate", 'font': {'size': 24}},
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "darkgreen"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                             {'range': [0.5, 1], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': pred}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}\n\n⚠️ Please make sure the inputs match the features used in training.")
else:
    st.info("Please enter the parameters in the left column and click the prediction button")
