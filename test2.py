# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# ===== 1️⃣ 定义 TargetEncoderCV（和训练时完全一致） =====
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
                X_encoded[col] = pd.Series(index=X_encoded.index, dtype=float)
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=self.n_splits)
                for train_idx, val_idx in gkf.split(X, y, groups):
                    mapping = y.iloc[train_idx].groupby(X.iloc[train_idx][col]).mean()
                    X_encoded.iloc[val_idx] = X.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:
                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# 加载模型、编码器和特征列
bundle = joblib.load("xgb_pipeline.joblib")
model = bundle["model"]
encoder = bundle["encoder"]
feature_cols = bundle["feature_cols"]

# 加载训练数据用于获取特征范围
df = pd.read_excel("data.xlsx")
X_train = df[feature_cols].copy()
X_train['Antibiotic'] = df['Antibiotic']

# 获取每列最小值和最大值（数值列）
num_cols = [c for c in feature_cols if c != 'Antibiotic']
feature_ranges = {col: (X_train[col].min(), X_train[col].max()) for col in num_cols}

st.title("🧪 Degradation rate prediction system")

# 用户输入
antibiotic = st.selectbox(
    "Type of Antibiotic",
    ["CEP", "AMP", "其他"]  # 可根据训练数据修改
)

ph = st.number_input(
    f"pH ({feature_ranges['pH'][0]} ~ {feature_ranges['pH'][1]})",
    min_value=float(feature_ranges['pH'][0]),
    max_value=float(feature_ranges['pH'][1]),
    value=float((feature_ranges['pH'][0]+feature_ranges['pH'][1])/2)
)

water_content = st.number_input(
    f"Water content(%) ({feature_ranges['Water content(%)'][0]} ~ {feature_ranges['Water content(%)'][1]})",
    min_value=float(feature_ranges['Water content(%)'][0]),
    max_value=float(feature_ranges['Water content(%)'][1]),
    value=float((feature_ranges['Water content(%)'][0]+feature_ranges['Water content(%)'][1])/2)
)

m = st.number_input(
    f"m(g) ({feature_ranges['m(g)'][0]} ~ {feature_ranges['m(g)'][1]})",
    min_value=float(feature_ranges['m(g)'][0]),
    max_value=float(feature_ranges['m(g)'][1]),
    value=float((feature_ranges['m(g)'][0]+feature_ranges['m(g)'][1])/2)
)

T = st.number_input(
    f"T(°C) ({feature_ranges['T(°C)'][0]} ~ {feature_ranges['T(°C)'][1]})",
    min_value=float(feature_ranges['T(°C)'][0]),
    max_value=float(feature_ranges['T(°C)'][1]),
    value=float((feature_ranges['T(°C)'][0]+feature_ranges['T(°C)'][1])/2)
)

V = st.number_input(
    f"V(L) ({feature_ranges['V(L)'][0]} ~ {feature_ranges['V(L)'][1]})",
    min_value=float(feature_ranges['V(L)'][0]),
    max_value=float(feature_ranges['V(L)'][1]),
    value=float((feature_ranges['V(L)'][0]+feature_ranges['V(L)'][1])/2)
)

t = st.number_input(
    f"t(min) ({feature_ranges['t(min)'][0]} ~ {feature_ranges['t(min)'][1]})",
    min_value=float(feature_ranges['t(min)'][0]),
    max_value=float(feature_ranges['t(min)'][1]),
    value=float((feature_ranges['t(min)'][0]+feature_ranges['t(min)'][1])/2)
)

HCL = st.number_input(
    f"HCL Conc(mol/L) ({feature_ranges['HCL Conc(mol/L)'][0]} ~ {feature_ranges['HCL Conc(mol/L)'][1]})",
    min_value=float(feature_ranges['HCL Conc(mol/L)'][0]),
    max_value=float(feature_ranges['HCL Conc(mol/L)'][1]),
    value=float((feature_ranges['HCL Conc(mol/L)'][0]+feature_ranges['HCL Conc(mol/L)'][1])/2)
)

NaOH = st.number_input(
    f"NaOH Conc(mol/L) ({feature_ranges['NaOH Conc(mol/L)'][0]} ~ {feature_ranges['NaOH Conc(mol/L)'][1]})",
    min_value=float(feature_ranges['NaOH Conc(mol/L)'][0]),
    max_value=float(feature_ranges['NaOH Conc(mol/L)'][1]),
    value=float((feature_ranges['NaOH Conc(mol/L)'][0]+feature_ranges['NaOH Conc(mol/L)'][1])/2)
)

# 构建用户输入 DataFrame
X_user = pd.DataFrame({
    "pH": [ph],
    "Water content(%)": [water_content],
    "m(g)": [m],
    "T(°C)": [T],
    "V(L)": [V],
    "t(min)": [t],
    "HCL Conc(mol/L)": [HCL],
    "NaOH Conc(mol/L)": [NaOH],
    "Antibiotic": [antibiotic]
})

# 编码分类变量
X_user_enc = encoder.transform(X_user)

# 预测
pred = model.predict(X_user_enc)[0]

st.write(f"Predicted Degradation: {pred:.2f}")

