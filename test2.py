# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.graph_objects as go

plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# ======================================================
# 1️⃣ TargetEncoderCV
# ======================================================
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
                # 分组 CV 编码
                X_encoded[col] = np.nan
                gkf = GroupKFold(n_splits=self.n_splits)
                X_temp, y_temp, groups_temp = X.copy(), y.copy(), groups.copy()
                for train_idx, val_idx in gkf.split(X_temp, y_temp, groups_temp):
                    mapping = y_temp.iloc[train_idx].groupby(X_temp.iloc[train_idx][col]).mean()
                    X_encoded.iloc[val_idx, X_encoded.columns.get_loc(col)] = X_temp.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:
                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# ======================================================
# 2️⃣ 数据加载
# ======================================================
df = pd.read_excel("data.xlsx")  # 替换为你的数据路径

feature_cols = df.columns[1:10]
categorical_cols = ['Antibiotic']

X = df[feature_cols].copy()
X['Antibiotic'] = df['Antibiotic']
y = df['Degradation']
groups = df['Group']

test_groups = {4, 5, 8, 12, 13, 15, 16, 17}
all_groups = set(df['Group'].unique())
train_groups = all_groups - test_groups

train_mask = groups.isin(train_groups)
test_mask = groups.isin(test_groups)

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]
groups_train = groups.loc[train_mask]

# ======================================================
# 3️⃣ 编码
# ======================================================
encoder = TargetEncoderCV(cat_cols=categorical_cols, n_splits=5, random_state=42)
X_train_encoded = encoder.fit_transform(X_train, y_train, groups=groups_train)
X_test_encoded = encoder.transform(X_test)

# ======================================================
# 4️⃣ XGB 模型训练
# ======================================================
param_dist = {
    'n_estimators': [100, 150, 200, 300, 400, 500],
    'max_depth': [6, 7, 8, 9],
    'learning_rate': [0.15, 0.2],
    'subsample': [0.5, 0.6],
    'colsample_bytree': [0.4, 0.5],
    'reg_alpha': [1.0, 5.0],
    'reg_lambda': [10, 30, 50]
}

xgb_base = XGBRegressor(random_state=42, objective="reg:squarederror")
group_kfold = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=30,
    scoring='r2',
    cv=group_kfold,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train_encoded, y_train, groups=groups_train)
best_xgb = search.best_estimator_

# ======================================================
# 5️⃣ Streamlit 网页
# ======================================================
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")

# 特征顺序
MODEL_FEATURES = [
    'pH', 'Water content(%)', 'm(g)', 'T(°C)',
    'V(L)', 't(min)', 'HCL Conc(mol/L)', 'NaOH Conc(mol/L)',
    'Degradation', 'Antibiotic'
]

LABELS = {
    'Antibiotic': 'Type of Antibiotic',
    'pH': 'Initial environmental pH [2–12]',
    'Water content(%)': 'Water content (%) [5.35–98.1]',
    'm(g)': 'Quality (g) [1–500]',
    'T(°C)': 'Reaction temperature (°C) [0–340]',
    'V(L)': 'Reactor volume (L) [0.05–1]',
    't(min)': 'Reaction time (min) [0–480]',
    'HCL Conc(mol/L)': 'HCL concentration (mol/L) [0–0.6]',
    'NaOH Conc(mol/L)': 'NaOH concentration (mol/L) [0–0.6]'
}

st.sidebar.header("Please enter parameters")
inputs = {}

# 抗生素选择
ANTIBIOTIC_LIST = list(encoder.mapping_['Antibiotic'].index)
inputs['Antibiotic'] = st.sidebar.selectbox(
    LABELS['Antibiotic'], ANTIBIOTIC_LIST
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
    inputs[k] = st.sidebar.number_input(LABELS[k], value=float(v), format="%.3f")

predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# 预测逻辑
if predict_btn:
    X_user = pd.DataFrame([inputs])
    X_user['Degradation'] = 0.0
    X_user = X_user[MODEL_FEATURES]

    X_user_enc = encoder.transform(X_user)
    pred = best_xgb.predict(X_user_enc)[0]

    st.markdown(f"### ✅ Predicted Degradation rate: `{pred:.3f}`")

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
st.markdown("*This system uses a unified machine learning pipeline to ensure consistent preprocessing and prediction.*")
