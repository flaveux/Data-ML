import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('log_returns_dataset.xlsx', sheet_name='Log_Returns')

all_regressors = [col for col in df.columns if col != 'IMOEX']
y = df['IMOEX']
X = df[all_regressors]

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)

print("=" * 80)
print("РЕГУЛЯРИЗОВАННЫЕ МОДЕЛИ: LASSO RIDGE ELASTIC NET")
print("=" * 80)
print(f"Форма данных: {df.shape}")
print(f"Всего регрессоров: {X.shape[1]}")
print(f"Обучающая выборка: {len(X_train)}")
print(f"Тестовая выборка: {len(X_test)}")
print(f"Кросс-валидация: TimeSeriesSplit(n_splits=5)")
print("\nСхема TimeSeriesSplit (5 фолдов на train данных):")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    print(f"  Фолд {fold}: train [{train_idx[0]}..{train_idx[-1]}] ({len(train_idx)} набл.) → val [{val_idx[0]}..{val_idx[-1]}] ({len(val_idx)} набл.)")

results = []

# LASSO

print("\n" + "=" * 80)
print("1. LASSO (L1) - с автоматическим подбором alpha (TimeSeriesSplit)")
print("=" * 80)

lasso_cv = LassoCV(cv=tscv, random_state=42,
                   alphas=np.logspace(-5, 0, 100),
                   max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Оптимальный alpha: {lasso_cv.alpha_:.8f}")

lasso = Lasso(alpha=lasso_cv.alpha_, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Прогноз и метрики
y_pred_lasso = lasso.predict(X_test_scaled)
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print(f"\nМЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
print(f"  R²   = {r2_lasso:.4f}")
print(f"  RMSE = {rmse_lasso:.6f}")
print(f"  MAE  = {mae_lasso:.6f}")

# Коэффициенты
lasso_selected = X.columns[lasso.coef_ != 0].tolist()
print(f"\nОтобрано признаков: {len(lasso_selected)} из {X.shape[1]}")
print(f"Отобранные: {lasso_selected}")

# Таблица коэффициентов с константой
lasso_const = lasso.intercept_
for i, col in enumerate(X.columns):
    print(f"  {col}: {lasso.coef_[i]:.6f}")
print(f"  const: {lasso_const:.6f}")

results.append({
    'Модель': 'LASSO (L1)',
    'R²': r2_lasso,
    'RMSE': rmse_lasso,
    'MAE': mae_lasso,
    'Отобрано признаков': len(lasso_selected),
    'alpha': lasso_cv.alpha_
})

# RIDGE

print("\n" + "=" * 80)
print("2. RIDGE (L2) - с автоматическим подбором alpha (TimeSeriesSplit)")
print("=" * 80)

ridge_cv = RidgeCV(alphas=np.logspace(-5, 2, 100), cv=tscv)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Оптимальный alpha: {ridge_cv.alpha_:.8f}")

ridge = Ridge(alpha=ridge_cv.alpha_, random_state=42)
ridge.fit(X_train_scaled, y_train)

# Прогноз и метрики
y_pred_ridge = ridge.predict(X_test_scaled)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print(f"\nМЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
print(f"  R²   = {r2_ridge:.4f}")
print(f"  RMSE = {rmse_ridge:.6f}")
print(f"  MAE  = {mae_ridge:.6f}")

# Коэффициенты (Ridge не обнуляет)
print(f"\nКОЭФФИЦИЕНТЫ RIDGE (все признаки сохранены):")
ridge_const = ridge.intercept_
for i, col in enumerate(X.columns):
    print(f"  {col}: {ridge.coef_[i]:.6f}")
print(f"  const: {ridge_const:.6f}")

results.append({
    'Модель': 'RIDGE (L2)',
    'R²': r2_ridge,
    'RMSE': rmse_ridge,
    'MAE': mae_ridge,
    'Отобрано признаков': X.shape[1],
    'alpha': ridge_cv.alpha_
})

# ELASTIC NET (L1+L2)

print("\n" + "=" * 80)
print("3. ELASTIC NET (L1+L2) - с автоматическим подбором alpha и l1_ratio (TimeSeriesSplit)")
print("=" * 80)

elastic_cv = ElasticNetCV(cv=tscv, random_state=42,
                          l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1],
                          alphas=np.logspace(-5, 0, 100),
                          max_iter=10000)
elastic_cv.fit(X_train_scaled, y_train)

print(f"Оптимальный alpha: {elastic_cv.alpha_:.8f}")
print(f"Оптимальный l1_ratio: {elastic_cv.l1_ratio_:.4f}")

elastic = ElasticNet(alpha=elastic_cv.alpha_, l1_ratio=elastic_cv.l1_ratio_,
                     random_state=42, max_iter=10000)
elastic.fit(X_train_scaled, y_train)

# Прогноз и метрики
y_pred_elastic = elastic.predict(X_test_scaled)
r2_elastic = r2_score(y_test, y_pred_elastic)
rmse_elastic = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
mae_elastic = mean_absolute_error(y_test, y_pred_elastic)

print(f"\nМЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
print(f"  R²   = {r2_elastic:.4f}")
print(f"  RMSE = {rmse_elastic:.6f}")
print(f"  MAE  = {mae_elastic:.6f}")

# Коэффициенты
elastic_selected = X.columns[elastic.coef_ != 0].tolist()
print(f"\nОтобрано признаков: {len(elastic_selected)} из {X.shape[1]}")
print(f"Отобранные: {elastic_selected}")

print(f"\nКОЭФФИЦИЕНТЫ ELASTIC NET (только ненулевые):")
elastic_const = elastic.intercept_
for i, col in enumerate(X.columns):
    if elastic.coef_[i] != 0:
        print(f"  {col}: {elastic.coef_[i]:.6f}")
print(f"  const: {elastic_const:.6f}")

results.append({
    'Модель': 'Elastic Net (L1+L2)',
    'R²': r2_elastic,
    'RMSE': rmse_elastic,
    'MAE': mae_elastic,
    'Отобрано признаков': len(elastic_selected),
    'alpha': elastic_cv.alpha_,
    'l1_ratio': elastic_cv.l1_ratio_
})


print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА МЕТРИК (тестовая выборка)")
print("=" * 80)

metrics_df = pd.DataFrame(results)
print(metrics_df.round(6).to_string(index=False))

print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ (с константой)")
print("=" * 80)

coef_df = pd.DataFrame({
    'Признак': X.columns.tolist() + ['const'],
    'LASSO (L1)': list(lasso.coef_) + [lasso.intercept_],
    'RIDGE (L2)': list(ridge.coef_) + [ridge.intercept_],
    'Elastic Net': list(elastic.coef_) + [elastic.intercept_]
})
print(coef_df.round(6).to_string(index=False))

metrics_df.to_excel('regularization_metrics.xlsx', index=False)
coef_df.to_excel('regularization_coefficients.xlsx', index=False)
