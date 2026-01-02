import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
train_groups = set(groups.unique()) - test_groups

train_mask = groups.isin(train_groups)
test_mask  = groups.isin(test_groups)

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]
groups_train = groups.loc[train_mask]

print("训练集样本数:", X_train.shape[0])
print("测试集样本数:", X_test.shape[0])
print("训练集文献数:", len(train_groups))
print("测试集文献数:", len(test_groups))

class TargetEncoder:
    def __init__(self):
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        for col in X.columns:
            self.mapping_[col] = y.groupby(X[col]).mean().to_dict()
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = X_encoded[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_encoded

encoder = TargetEncoder()
encoder.fit(X_train[categorical_cols], y_train)
X_train_cat = encoder.transform(X_train[categorical_cols])
X_test_cat  = encoder.transform(X_test[categorical_cols])

X_train_final = pd.concat([X_train_cat, X_train[num_cols]], axis=1)
X_test_final  = pd.concat([X_test_cat, X_test[num_cols]], axis=1)

knn_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

param_dist = {
    'knn__n_neighbors': [10, 12, 15, 18, 20],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

group_cv = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=knn_pipe,
    param_distributions=param_dist,
    n_iter=20,
    scoring='r2',
    cv=group_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_final, y_train, groups=groups_train)
best_knn = search.best_estimator_

print("\n最佳参数组合:")
print(search.best_params_)

joblib.dump(best_knn, "knn_best.pkl")

r2_list, mae_list, rmse_list = [], [], []
y_train_cv_pred = np.zeros(len(y_train))

print("\n训练集 5折 Group CV 每折性能:")
for i, (train_idx, val_idx) in enumerate(group_cv.split(X_train_final, y_train, groups_train), 1):
    X_tr, X_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    best_knn.fit(X_tr, y_tr)
    y_pred = best_knn.predict(X_val)
    y_train_cv_pred[val_idx] = y_pred

    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    r2_list.append(r2)
    mae_list.append(mae)
    rmse_list.append(rmse)

    print(f" Fold {i}: R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")

r2_cv_mean   = np.mean(r2_list)
r2_cv_std    = np.std(r2_list)
mae_cv_mean  = np.mean(mae_list)
mae_cv_std   = np.std(mae_list)
rmse_cv_mean = np.mean(rmse_list)
rmse_cv_std  = np.std(rmse_list)

print("\n训练集 5折 Group CV 平均性能:")
print(f"R²   : {r2_cv_mean:.4f} ± {r2_cv_std:.4f}")
print(f"MAE  : {mae_cv_mean:.4f} ± {mae_cv_std:.4f}")
print(f"RMSE : {rmse_cv_mean:.4f} ± {rmse_cv_std:.4f}")

best_knn.fit(X_train_final, y_train)
y_test_pred = best_knn.predict(X_test_final)

mae_test  = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test   = r2_score(y_test, y_test_pred)

print("\n独立 Group 测试集性能:")
print(f"MAE : {mae_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"R²  : {r2_test:.4f}")

idx = np.random.choice(X_test_final.index)
single_x = X_test_final.loc[[idx]]
single_true = y_test.loc[idx]
single_pred = best_knn.predict(single_x)[0]

print(f"\n选中测试样本 index = {idx}")
print("真实降解率:", single_true)
print("模型预测值:", single_pred)