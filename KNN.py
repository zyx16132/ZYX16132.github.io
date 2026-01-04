import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from joblib import dump

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_excel('data.xlsx')

train_indices = []
test_indices = []

for group_id, g_df in df.groupby(df.iloc[:, 0]):
    idx = g_df.index.to_numpy()
    np.random.shuffle(idx)

    n_train = int(len(idx) * 0.8)
    train_indices.extend(idx[:n_train])
    test_indices.extend(idx[n_train:])

df_train = df.loc[train_indices].reset_index(drop=True)
df_test  = df.loc[test_indices].reset_index(drop=True)

print('========== 数据划分 ==========')
print('训练集样本数:', df_train.shape[0])
print('测试集样本数:', df_test.shape[0])
print('Group 总数:', df.iloc[:, 0].nunique())
print('训练集 Group 数:', df_train.iloc[:, 0].nunique())
print('测试集 Group 数:', df_test.iloc[:, 0].nunique())

X_train = df_train.iloc[:, 2:11]
y_train = df_train.iloc[:, 11]

X_test  = df_test.iloc[:, 2:11]
y_test  = df_test.iloc[:, 11]

param_dist = {
    'n_neighbors': list(range(3, 31)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'leaf_size': [10, 20, 30, 40, 50]
}

knn_base = KNeighborsRegressor()

search = RandomizedSearchCV(
    estimator=knn_base,
    param_distributions=param_dist,
    n_iter=40,
    scoring='r2',
    cv=30,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)
best_knn = search.best_estimator_

print('\n========== 最佳参数组合（KNN） ==========')
for k, v in search.best_params_.items():
    print(f'{k}: {v}')

def calc_metrics(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred)
    )

r2_tr, rmse_tr, mae_tr = calc_metrics(y_train, best_knn.predict(X_train))
r2_te, rmse_te, mae_te = calc_metrics(y_test,  best_knn.predict(X_test))

print('\n========== 模型性能汇总（KNN） ==========')
print(f'Train R²   : {r2_tr:.4f}')
print(f'Train RMSE : {rmse_tr:.4f}')
print(f'Train MAE  : {mae_tr:.4f}')
print('---------------------------------')
print(f'Test  R²   : {r2_te:.4f}')
print(f'Test  RMSE : {rmse_te:.4f}')
print(f'Test  MAE  : {mae_te:.4f}')

dump(best_knn, 'knn_best.pkl')
print('\n模型已保存为 knn_best.pkl')

print('\n========== 5 折 Group 分层交叉验证（KNN） ==========')

K = 5
df_cv = df.copy()
df_cv['fold'] = -1

for gid, g_df in df_cv.groupby(df_cv.iloc[:, 0]):
    idx = g_df.index.to_numpy()
    np.random.shuffle(idx)
    folds = np.array_split(idx, K)
    for k in range(K):
        df_cv.loc[folds[k], 'fold'] = k

cv_metrics = []

for k in range(K):
    df_train_cv = df_cv[df_cv['fold'] != k]
    df_test_cv  = df_cv[df_cv['fold'] == k]

    X_tr = df_train_cv.iloc[:, 2:11]
    y_tr = df_train_cv.iloc[:, 11]
    X_te = df_test_cv.iloc[:, 2:11]
    y_te = df_test_cv.iloc[:, 11]

    model = clone(best_knn)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2, rmse, mae = calc_metrics(y_te, y_pred)
    cv_metrics.append([r2, rmse, mae])

    print(f'Fold {k+1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}')

cv_metrics = np.array(cv_metrics)

print('\n========== 5 折交叉验证统计（KNN） ==========')
print(f'R²   : {cv_metrics[:,0].mean():.4f} ± {cv_metrics[:,0].std():.4f}')
print(f'RMSE : {cv_metrics[:,1].mean():.4f} ± {cv_metrics[:,1].std():.4f}')
print(f'MAE  : {cv_metrics[:,2].mean():.4f} ± {cv_metrics[:,2].std():.4f}')