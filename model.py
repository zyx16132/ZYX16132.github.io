import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# 1. TargetEncoderCV（完全不动）
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
        self.mapping_ = {}
        for col in self.cat_cols:
            if col in X.columns:
                self.mapping_[col] = y.groupby(X[col]).mean()
        return self

    def transform(self, X, y=None, groups=None):
        X_encoded = X.copy()
        for col in self.cat_cols:
            if col in X_encoded.columns:
                X_encoded[col] = (
                    X_encoded[col]
                    .map(self.mapping_[col])
                    .fillna(self.global_mean_)
                )
        return X_encoded


# ======================================================
# 2. 训练函数（只负责训练）
# ======================================================
def train_xgb_model(data_path="文献数据.xlsx"):
    df = pd.read_excel(data_path)

    feature_cols = df.columns[1:10]
    categorical_cols = ["Antibiotic"]

    X = df[feature_cols].copy()
    X["Antibiotic"] = df["Antibiotic"]
    y = df["Degradation"]
    groups = df["Group"]

    test_groups = {4, 5, 8, 12, 13, 15, 16, 17}
    train_mask = ~groups.isin(test_groups)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    groups_train = groups.loc[train_mask]

    encoder = TargetEncoderCV(categorical_cols, 5, 42)
    encoder.fit(X_train, y_train, groups_train)
    X_train_enc = encoder.transform(X_train)

    param_dist = {
        "n_estimators": [100, 150, 200, 300, 400, 500],
        "max_depth": [6, 7, 8, 9],
        "learning_rate": [0.15, 0.2],
        "subsample": [0.5, 0.6],
        "colsample_bytree": [0.4, 0.5],
        "reg_alpha": [1.0, 5.0],
        "reg_lambda": [10, 30, 50]
    }

    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, objective="reg:squarederror"),
        param_dist,
        n_iter=30,
        scoring="r2",
        cv=GroupKFold(5),
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train_enc, y_train, groups=groups_train)

    return search.best_estimator_, encoder, feature_cols


# ======================================================
# 3. ⚠️ 只训练一次（模块加载时）
# ======================================================
_best_xgb, _encoder, _feature_cols = train_xgb_model()

# ======================================================
# 4. 🔒 对网页只读暴露
# ======================================================
best_xgb = _best_xgb
encoder = _encoder
feature_cols = _feature_cols

__all__ = ["best_xgb", "encoder", "feature_cols"]
