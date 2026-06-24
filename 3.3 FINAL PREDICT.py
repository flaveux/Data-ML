import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

df_raw  = pd.read_excel('DATASET.xlsx')   # уровни (157 набл.)
df_pred = pd.read_excel('PREDICT.xlsx')   # прогнозный период (8 набл.)

TARGET      = 'IMOEX'
ALL_REGS    = ['S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000', 'Bovespa',
               'Euro Stoxx 50', 'DAX', 'CAC 40', 'FTSE 100', 'AEX',
               'Nifty 50', 'Shanghai Composite', 'Nikkei 225']
SEL_REGS    = ['S&P 500', 'NASDAQ', 'Euro Stoxx 50',
               'Nifty 50', 'Shanghai Composite', 'Nikkei 225']
LAG_PERIODS = [1, 2, 3]
DATES       = ['04.01.2026','11.01.2026','18.01.2026','25.01.2026',
               '01.02.2026','08.02.2026','15.02.2026','22.02.2026']

last_price  = df_raw[TARGET].iloc[-1]
y_real      = df_pred[TARGET].values

tscv = TimeSeriesSplit(n_splits=5)

print("=" * 80)
print("ML-МОДЕЛИ: 4 СПЕЦИФИКАЦИИ, ПРОГНОЗ 8 НЕДЕЛЬ")
print("=" * 80)
print(f"Последний уровень IMOEX (28.12.2025): {last_price:.2f}")
print(f"Реальные значения PREDICT: {y_real.tolist()}")


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ


def add_lags(df, cols, lags):
    d = df.copy()
    for col in cols:
        for lag in lags:
            d[f'{col}_lag{lag}'] = d[col].shift(lag)
    orig_regs = [c for c in cols if c != TARGET]
    d = d.drop(columns=orig_regs)
    return d.dropna()

def calc_metrics(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                           / np.array(y_true))) * 100
    return {
        'Модель': name,
        'R²':     round(r2,   4),
        'RMSE':   round(rmse, 2),
        'MAE':    round(mae,  2),
        'MAPE%':  round(mape, 2)
    }

def forecast_ml(model, feat_cols, regs, df_raw, df_pred, lags=False):
    if not lags:
        X_pred = df_pred[regs].copy()
        return [round(p, 2) for p in model.predict(X_pred)]
    else:
        tail = df_raw[regs + [TARGET]].iloc[-3:].reset_index(drop=True)
        combined = pd.concat(
            [tail, df_pred[regs + [TARGET]].reset_index(drop=True)],
            ignore_index=True
        )
        rows = []
        for step in range(8):
            row = {}
            for col in regs + [TARGET]:
                for lag in LAG_PERIODS:
                    row[f'{col}_lag{lag}'] = combined[col].iloc[step + 3 - lag]
            rows.append(row)
        X_pred = pd.DataFrame(rows)[feat_cols]
        return [round(p, 2) for p in model.predict(X_pred)]

# ГИПЕРПАРАМЕТРЫ
param_rf = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [3, 4, 5, 6, None],
    'min_samples_leaf':[2, 5, 10, 15],
    'max_features':    ['sqrt', 'log2', 0.5, 0.8],
}
param_xgb = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [3, 4, 5, 6],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.6, 0.8, 1.0],
    'min_child_weight':[1, 3, 5],
}
param_lgbm = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [3, 4, 5, 6],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.6, 0.8, 1.0],
    'num_leaves':      [15, 31, 63],
}

def fit_ml(estimator, params, X, y):
    search = RandomizedSearchCV(
        estimator, param_distributions=params,
        n_iter=30, scoring='r2', cv=tscv,
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

# СПЕЦИФИКАЦИИ

SPECS = {
    'Спец 1: все регрессоры (13 признаков, без лагов)':
        {'regs': ALL_REGS, 'lags': False},
    'Спец 2: все регрессоры + лаги 1-3 (42 признака)':
        {'regs': ALL_REGS, 'lags': True},
    'Спец 3: S&P/NASDAQ/Euro Stoxx 50/Nifty/Shanghai/Nikkei (6 признаков, без лагов)':
        {'regs': SEL_REGS, 'lags': False},
    'Спец 4: S&P/NASDAQ/Euro Stoxx 50/Nifty/Shanghai/Nikkei + лаги 1-3 (21 признак)':
        {'regs': SEL_REGS, 'lags': True},
}

all_results   = []
all_forecasts = {}

# ГЛАВНЫЙ ЦИКЛ

for spec_name, spec in SPECS.items():
    regs     = spec['regs']
    use_lags = spec['lags']

    print("\n" + "=" * 80)
    print(f"СПЕЦИФИКАЦИЯ: {spec_name}")
    print("=" * 80)

    if not use_lags:
        X_ml        = df_raw[regs]
        y_ml        = df_raw[TARGET]
        feat_cols   = regs
    else:
        df_lag      = add_lags(df_raw[regs + [TARGET]], regs + [TARGET], LAG_PERIODS)
        feat_cols   = [c for c in df_lag.columns if c != TARGET]
        X_ml        = df_lag[feat_cols]
        y_ml        = df_lag[TARGET]

    split       = int(len(X_ml) * 0.7)
    X_train     = X_ml.iloc[:split]
    X_test      = X_ml.iloc[split:]
    y_train     = y_ml.iloc[:split]
    y_test      = y_ml.iloc[split:]

    print(f"  Признаков: {X_ml.shape[1]} | Train: {len(y_train)} | Test: {len(y_test)}")

    for mname, estimator, params in [
        ('Random Forest', RandomForestRegressor(random_state=42, n_jobs=-1), param_rf),
        ('XGBoost',       XGBRegressor(random_state=42, verbosity=0),        param_xgb),
        ('LightGBM',      LGBMRegressor(random_state=42, verbose=-1),        param_lgbm),
    ]:
        model, best_p = fit_ml(estimator, params, X_train, y_train)
        print(f"\n  {mname}: {best_p}")

        y_pred = model.predict(X_test)
        res    = calc_metrics(mname, y_test.values, y_pred)
        res['Спецификация'] = spec_name
        all_results.append(res)

        # Переобучаем на полной выборке для прогноза
        model.fit(X_ml, y_ml)
        fc = forecast_ml(model, feat_cols, regs, df_raw, df_pred, lags=use_lags)
        all_forecasts[f'{spec_name} | {mname}'] = fc

# ИТОГОВЫЕ ТАБЛИЦЫ

print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА МЕТРИК (тестовая выборка 30%)")
print("=" * 80)

res_df = pd.DataFrame(all_results)
for spec_name in SPECS:
    print(f"\n{spec_name}")
    sub = res_df[res_df['Спецификация'] == spec_name][
        ['Модель', 'R²', 'RMSE', 'MAE', 'MAPE%']]
    print(sub.to_string(index=False))

print("\n" + "=" * 80)
print("ПРОГНОЗЫ НА 8 НЕДЕЛЬ (уровни IMOEX)")
print("=" * 80)

fc_df = pd.DataFrame(all_forecasts, index=DATES)
fc_df.index.name = 'Дата'
fc_df['Факт'] = y_real
print(fc_df.to_string())