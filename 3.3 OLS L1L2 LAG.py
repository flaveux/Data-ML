import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

df = pd.read_excel('DATASET.xlsx')

TARGET = 'IMOEX'

FEATURE_COLS = [
    'S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000',
    'Bovespa', 'Euro Stoxx 50', 'DAX', 'CAC 40',
    'FTSE 100', 'AEX', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225'
]

LAG_PERIODS = [1, 2, 3]

df_feat = pd.DataFrame(index=df.index)

for col in FEATURE_COLS + [TARGET]:
    for lag in LAG_PERIODS:
        df_feat[f'{col}_lag{lag}'] = df[col].shift(lag)

df_feat[TARGET] = df[TARGET]
df_feat = df_feat.dropna()

X = df_feat.drop(columns=[TARGET])
y = df_feat[TARGET]

print(f"\nПризнаков : {X.shape[1]}  (13 регрессоров × 3 лага + IMOEX × 3 лага = 42)")
print(f"Наблюдений: {len(y)}")

TRAIN_RATIO = 0.8
split_idx = int(len(y) * TRAIN_RATIO)

X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n  R²   : {r2:.4f}  → объясняет {round(r2 * 100, 1)}% дисперсии уровня IMOEX")
    print(f"  RMSE : {rmse:.2f} пунктов индекса")
    print(f"  MAE  : {mae:.2f} пунктов индекса")
    return {'Модель': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae}

def print_significant_coefs(model, feature_names, top_n=5):
    coefs = pd.Series(model.coef_, index=feature_names)
    nonzero = coefs[coefs != 0].sort_values(key=abs, ascending=False)
    print(f"\n  Ненулевых коэффициентов: {len(nonzero)} из {len(coefs)}")
    print(f"  Топ-{top_n} по абсолютной величине:")
    for feat, val in nonzero.head(top_n).items():
        print(f"    {feat:<35} {val:.4f}")


results = []

# LASSO

print("\n" + "=" * 80)
print("LASSO (L1-регуляризация)")
print("=" * 80)

lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
results.append(evaluate('Lasso', y_test, y_pred_lasso))
print_significant_coefs(lasso, X.columns)

# RIDGE

print("\n" + "=" * 80)
print("RIDGE (L2-регуляризация)")
print("=" * 80)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
results.append(evaluate('Ridge', y_test, y_pred_ridge))
print_significant_coefs(ridge, X.columns)

# ELASTIC NET

print("\n" + "=" * 80)
print("ELASTIC NET (L1 + L2)")
print("=" * 80)

enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
enet.fit(X_train, y_train)
y_pred_enet = enet.predict(X_test)
results.append(evaluate('ElasticNet', y_test, y_pred_enet))
print_significant_coefs(enet, X.columns)


print("\n" + "=" * 80)
print("СРАВНЕНИЕ МНК-МОДЕЛЕЙ (тестовая выборка)")
print("=" * 80)

results_df = pd.DataFrame(results).set_index('Модель')
results_df = results_df.sort_values('R²', ascending=False)

print(f"\n  {'Модель':<15} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
print(f"  {'-' * 45}")
for name, row in results_df.iterrows():
    print(f"  {name:<15} {row['R²']:>8.4f} {row['RMSE']:>10.2f} {row['MAE']:>10.2f}")

# (TimeSeriesSplit, 5 фолдов)

print("\n" + "=" * 80)
print("КРОСС-ВАЛИДАЦИЯ (TimeSeriesSplit, 5 фолдов)")
print("=" * 80)

tscv = TimeSeriesSplit(n_splits=5)
models_cv = {
    'Lasso': Lasso(alpha=0.01, max_iter=10000, random_state=42),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
}

cv_results = []
for name, model in models_cv.items():
    cv_r2, cv_rmse, cv_mae = [], [], []
    for train_idx, test_idx in tscv.split(X):
        Xtr_raw = X.iloc[train_idx]
        Xte_raw = X.iloc[test_idx]
        ytr = y.iloc[train_idx]
        yte = y.iloc[test_idx]

        # Масштабируем внутри каждого фолда отдельно
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_raw)
        Xte_s = sc.transform(Xte_raw)

        model.fit(Xtr_s, ytr)
        ypred = model.predict(Xte_s)

        cv_r2.append(r2_score(yte, ypred))
        cv_rmse.append(np.sqrt(mean_squared_error(yte, ypred)))
        cv_mae.append(mean_absolute_error(yte, ypred))

    print(f"\n  {name}:")
    print(f"    R²   = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
    print(f"    RMSE = {np.mean(cv_rmse):.2f} ± {np.std(cv_rmse):.2f} пунктов")
    print(f"    MAE  = {np.mean(cv_mae):.2f} ± {np.std(cv_mae):.2f} пунктов")
    cv_results.append({
        'Модель': name,
        'R² mean': np.mean(cv_r2),
        'R² std': np.std(cv_r2),
        'RMSE mean': np.mean(cv_rmse),
        'MAE mean': np.mean(cv_mae)
    })


print("\n" + "=" * 80)
print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
print("=" * 80)
print(f"\n  {'Модель':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10}  Тип")
print(f"  {'-' * 60}")

# МНК результаты
mnk_models = [
    ('Lasso', r2_score(y_test, y_pred_lasso),
     np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
     mean_absolute_error(y_test, y_pred_lasso), 'МНК'),
    ('Ridge', r2_score(y_test, y_pred_ridge),
     np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
     mean_absolute_error(y_test, y_pred_ridge), 'МНК'),
    ('ElasticNet', r2_score(y_test, y_pred_enet),
     np.sqrt(mean_squared_error(y_test, y_pred_enet)),
     mean_absolute_error(y_test, y_pred_enet), 'МНК'),
]
for name, r2, rmse, mae, typ in sorted(mnk_models, key=lambda x: -x[1]):
    print(f"  {name:<20} {r2:>8.4f} {rmse:>10.2f} {mae:>10.2f}  {typ}")

