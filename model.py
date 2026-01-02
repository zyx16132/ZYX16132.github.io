import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")


# ======================================================
# 1. TargetEncoderCV 定义（完全保留）
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
                X_encoded[col] = np.nan
                gkf = GroupKFold(n_splits=self.n_splits)
                X_temp, y_temp, groups_temp = X.copy(), y.copy(), groups.copy()
                for train_idx, val_idx in gkf.split(X_temp, y_temp, groups_temp):
                    mapping = y_temp.iloc[train_idx].groupby(X_temp.iloc[train_idx][col]).mean()
                    X_encoded.iloc[val_idx, X_encoded.columns.get_loc(col)] = \
                        X_temp.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:
                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded


# ======================================================
# 2. 核心训练函数（新增，只是“包起来”）
# ======================================================
def train_xgb_model(data_path=r'文献数据.xlsx'):
    # ---------- 读取数据 ----------
    df = pd.read_excel(data_path)

    feature_cols = df.columns[1:10]
    categorical_cols = ['Antibiotic']

    X = df[feature_cols].copy()
    X['Antibiotic'] = df['Antibiotic']
    y = df['Degradation']
    groups = df['Group']

    # ---------- Group 划分 ----------
    test_groups = {4, 5, 8, 12, 13, 15, 16, 17}
    all_groups = set(df['Group'].unique())
    train_groups = all_groups - test_groups

    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    groups_train = groups.loc[train_mask]

    print("训练集样本数:", X_train.shape[0])
    print("测试集样本数:", X_test.shape[0])
    print("训练集文献数:", len(train_groups))
    print("测试集文献数:", len(test_groups))
    print("测试集 Group:", sorted(test_groups))

    # ---------- Target Encoding ----------
    encoder = TargetEncoderCV(cat_cols=categorical_cols, n_splits=5, random_state=42)
    X_train_encoded = encoder.fit_transform(X_train, y_train, groups=groups_train)
    X_test_encoded = encoder.transform(X_test)

    # ---------- XGB 参数搜索（完全不改） ----------
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

    print("\n最佳参数组合:")
    print(search.best_params_)

    # ---------- 训练集 CV ----------
    mae_list, rmse_list, r2_list = [], [], []

    print("\n训练集 5折 Group CV 每折指标:")
    for i, (tr_idx, val_idx) in enumerate(
            group_kfold.split(X_train_encoded, y_train, groups_train), 1):
        model = XGBRegressor(**best_xgb.get_params())
        model.fit(X_train_encoded.iloc[tr_idx], y_train.iloc[tr_idx])
        y_pred = model.predict(X_train_encoded.iloc[val_idx])

        r2_list.append(r2_score(y_train.iloc[val_idx], y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], y_pred)))
        mae_list.append(mean_absolute_error(y_train.iloc[val_idx], y_pred))

        print(f" Fold {i}: R²={r2_list[-1]:.4f}, RMSE={rmse_list[-1]:.4f}, MAE={mae_list[-1]:.4f}")

    print("\n训练集 5折 Group CV 平均指标:")
    print(f"R²   : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
    print(f"RMSE : {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"MAE  : {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")

    # ---------- 测试集 ----------
    pred_test = best_xgb.predict(X_test_encoded)
    print("\n测试集结果:")
    print("MAE :", mean_absolute_error(y_test, pred_test))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_test)))
    print("R²  :", r2_score(y_test, pred_test))

    # ---------- 返回给 app.py 用 ----------
    return {
        "model": best_xgb,
        "encoder": encoder,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols
    }


# ======================================================
# 3. 本地运行入口（必须保留）
# ======================================================
if __name__ == "__main__":
    train_xgb_model()

# ===============================
# 🔒 供网页调用的“只读接口”
# ===============================
__all__ = [
    "best_xgb",
    "encoder",
    "feature_cols"
]


