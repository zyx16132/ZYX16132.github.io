# test2.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import io
import sys

warnings.filterwarnings("ignore")

# -------------------- 1. 训练控制台输出到网页 --------------------
class StreamlitLogger:
    def write(self, buf):
        st.text(buf)
    def flush(self):
        pass
sys.stdout = StreamlitLogger()

# -------------------- 2. TargetEncoderCV（同前） --------------------
class TargetEncoderCV(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.global_mean_ = None
        self.mapping_ = {}

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        for col in self.cat_cols:
            self.mapping_[col] = y.groupby(X[col]).mean()
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in self.cat_cols:
            X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

# -------------------- 3. 训练函数（只跑一次，缓存） --------------------
@st.cache_resource   # 训练结果缓存，重启才重跑
def train_and_embed():
    st.info("🚀 正在训练最终模型，首次打开需 30-60 秒，请稍候...")
    df = pd.read_excel(r'data.xlsx')
    feature_cols = df.columns[1:10]
    categorical_cols = ['Antibiotic']
    X = df[feature_cols].copy()
    X['Antibiotic'] = df['Antibiotic']
    y = df['Degradation']

    # 划分训练/测试
    test_groups = {4, 5, 8, 12, 13, 15, 16, 17}
    all_groups = set(df['Group'].unique())
    train_groups = all_groups - test_groups
    train_mask = df['Group'].isin(train_groups)
    test_mask  = df['Group'].isin(test_groups)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    # 一次性编码
    encoder = TargetEncoderCV(cat_cols=categorical_cols)
    encoder.fit(X_train, y_train)
    X_train_enc = encoder.transform(X_train)
    X_test_enc  = encoder.transform(X_test)

    # 随机搜索（纯数值）
    param_dist = {
        'n_estimators': [100, 150, 200, 300, 400, 500],
        'max_depth': [6, 7, 8, 9],
        'learning_rate': [0.15, 0.2],
        'subsample': [0.5, 0.6],
        'colsample_bytree': [0.4, 0.5],
        'reg_alpha': [1.0, 5.0],
        'reg_lambda': [10, 30, 50]
    }
    xgb_base = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=30,
        scoring='r2',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train_enc, y_train)
    best_params = search.best_params_
    print("最佳参数:", best_params)

    # 总模型（最佳参数 + 全训练集）
    final_model = xgb.XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
    final_model.fit(X_train_enc, y_train)

    # 性能打印
    pred_test = final_model.predict(X_test_enc)
    print("\n===== 总模型性能 =====")
    print(f"Test R² : {r2_score(y_test, pred_test):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")
    print(f"Test MAE : {mean_absolute_error(y_test, pred_test):.4f}")

    # 返回硬编码素材
    encoder_dict = {cat: encoder.mapping_[cat].to_dict() for cat in categorical_cols}
    columns      = list(X_train_enc.columns)

    return {
        "model": final_model,
        "mapping": encoder_dict,
        "columns": columns
    }

# -------------------- 4. 硬编码素材（训练结果） --------------------
EMBED = train_and_embed()

model           = EMBED["model"]
encoder_mapping = EMBED["mapping"]
train_columns   = EMBED["columns"]
feature_cols    = [c for c in train_columns if c != 'Antibiotic']
cat_cols        = ['Antibiotic']

# -------------------- 5. 页面布局（同原文件） --------------------
st.set_page_config(page_title="Degradation rate prediction", layout="centered")
st.title("🧪 Degradation rate prediction system")
st.markdown("---")
st.sidebar.header("Please enter parameters")

sidebar_order = [
    "Antibiotic", "pH", "Water content(%)", "m(g)", "T(°C)",
    "V(L)", "t(min)", "HCL Conc(mol/L)", "NaOH Conc(mol/L)"
]

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

inputs = {}

# -------------------- 6. 分类特征（动态全部抗生素） --------------------
for col in sidebar_order:
    if col in cat_cols:
        options = sorted(encoder_mapping.keys())
        inputs[col] = st.sidebar.selectbox(col, options)

# -------------------- 7. 数值特征（保留 3 位小数） --------------------
for col in sidebar_order:
    if col in feature_cols:
        min_val, max_val, default = feature_ranges[col]
        inputs[col] = st.sidebar.number_input(
            label=col,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.001,
            format="%.3f"
        )

# -------------------- 8. Predict 按钮 --------------------
predict_btn = st.sidebar.button("🔍 Predict degradation rate")

# -------------------- 9. 预测逻辑（对齐 train_columns） --------------------
if predict_btn:
    X_user = pd.DataFrame(columns=train_columns)
    for col, val in inputs.items():
        X_user.loc[0, col] = val
    # 直接 map，永无除零
    for cat in cat_cols:
        mapping = encoder_mapping
        X_user[cat] = X_user[cat].map(mapping).fillna(np.mean(list(mapping.values())))
    X_user = X_user.astype(float)
    X_user_final = X_user[train_columns]
    pred = model.predict(X_user_final.values)[0]

    st.markdown(f"### ✅ Predicted Degradation rate: **{pred:.2f}%**")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={"suffix": "%"},
        title={"text": "Degradation rate"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkgreen"},
            "steps": [
                {"range": [0, 50], "color": "#f2f2f2"},
                {"range": [50, 100], "color": "#c7e9c0"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter the parameters on the left and click Predict.")

st.markdown("---")
st.markdown(
    "*This application uses the final trained XGBoost model "
    "and the same target encoding as the training pipeline.*"
)
