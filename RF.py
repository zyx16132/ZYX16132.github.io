import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

df = pd.read_excel(r'data.xlsx')

num_cols = df.columns[1:10].drop('Antibiotic', errors='ignore')
categorical_cols = ['Antibiotic']

X = df[num_cols.tolist()].copy()
X['Antibiotic'] = df['Antibiotic']
y = df['Degradation']
groups = df['Group']

test_groups = {4, 5, 8, 12, 13, 15, 16, 17}
all_groups = set(groups.unique())
train_groups = all_groups - test_groups

train_mask = groups.isin(train_groups)
test_mask = groups.isin(test_groups)

X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
y_train, y_test = y.loc[train_mask], y.loc[test_mask]
groups_train = groups.loc[train_mask]

print("训练集样本数:", X_train.shape[0])
print("测试集样本数:", X_test.shape[0])
print("训练集文献数:", len(train_groups))
print("测试集文献数:", len(test_groups))

from sklearn.model_selection import GroupKFold

class TargetEncoderCV:
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
                    X_encoded.iloc[val_idx, X_encoded.columns.get_loc(col)] = X_temp.iloc[val_idx][col].map(mapping)
                X_encoded[col] = X_encoded[col].fillna(y.mean())
            else:

                X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

encoder = TargetEncoderCV(cat_cols=categorical_cols, n_splits=5, random_state=42)

encoder.fit(X_train[categorical_cols], y_train, groups=groups_train)
X_train_encoded = encoder.transform(X_train[categorical_cols], y=y_train, groups=groups_train)

X_test_encoded = encoder.transform(X_test[categorical_cols])

X_train_final = pd.concat([X_train_encoded, X_train[num_cols]], axis=1)
X_test_final = pd.concat([X_test_encoded, X_test[num_cols]], axis=1)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'max_features': ['sqrt']
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
group_kfold = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring='r2',
    cv=group_kfold,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_final, y_train, groups=groups_train)
best_rf = search.best_estimator_
print("\n最佳参数组合:")
print(search.best_params_)

joblib.dump(best_rf, "rf_groupCV.pkl")

mae_list, rmse_list, r2_list = [], [], []
y_train_cv_pred = np.zeros(len(y_train))

print("\n训练集 5折 Group CV 每折指标:")
for i, (tr_idx, val_idx) in enumerate(group_kfold.split(X_train_final, y_train, groups_train), 1):
    X_tr, X_val = X_train_final.iloc[tr_idx], X_train_final.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model = RandomForestRegressor(**best_rf.get_params())
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)

    y_train_cv_pred[val_idx] = y_pred

    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f" Fold {i}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

r2_cv_mean = np.mean(r2_list)
r2_cv_std = np.std(r2_list)
rmse_cv_mean = np.mean(rmse_list)
rmse_cv_std = np.std(rmse_list)
mae_cv_mean = np.mean(mae_list)
mae_cv_std = np.std(mae_list)

print("\n训练集 5折 Group CV 平均性能:")
print(f"R²   : {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")
print(f"RMSE : {rmse_cv_mean:.4f} ± {rmse_cv_std:.4f}")
print(f"MAE  : {mae_cv_mean:.4f} ± {mae_cv_std:.4f}")

y_test_pred = best_rf.predict(X_test_final)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("\n独立 Group 测试集性能:")
print(f"R²   : {r2_test:.4f}")
print(f"RMSE : {rmse_test:.4f}")
print(f"MAE  : {mae_test:.4f}")

idx = np.random.choice(X_test_final.index)
single_x = X_test_final.loc[[idx]]
single_true = y_test.loc[idx]
single_pred = best_rf.predict(single_x)[0]

print(f"\n选中测试样本 index = {idx}")
print("真实降解率:", single_true)
print("模型预测值:", single_pred)