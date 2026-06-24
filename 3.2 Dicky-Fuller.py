import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

file_path = 'DATASET.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Если первый столбец (C0) — нумерация, удаляем его
if df.columns[0] == 'C0' or df.iloc[:, 0].dtype in ['int64', 'float64']:
    # Проверяем, похож ли первый столбец на нумерацию
    first_col = df.iloc[:, 0]
    if first_col.is_monotonic_increasing and first_col.iloc[0] == 1:
        df = df.drop(columns=[df.columns[0]])

columns_check = ['IMOEX', 'S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000',
                 'Bovespa', 'Euro Stoxx 50', 'DAX', 'CAC 40', 'FTSE 100',
                 'AEX', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225']
existing_columns = [col for col in columns_check if col in df.columns]

existing_columns = ['IMOEX'] + [col for col in existing_columns if col != 'IMOEX']
df_clean = df[existing_columns].dropna()

print("=" * 80)
print("ИСХОДНЫЕ ДАННЫЕ")
print("=" * 80)
print(f"Первый столбец: {df_clean.columns[0]}")
print(f"Форма данных: {df_clean.shape}")
print("\nПервые 5 строк:")
print(df_clean.head())


# РАСЧЁТ ЛОГАРИФМИЧЕСКИХ ДОХОДНОСТЕЙ

log_returns = pd.DataFrame()
for col in existing_columns:
    prices = df_clean[col].dropna()
    log_returns[col] = np.log(prices / prices.shift(1))
log_returns = log_returns.dropna()

print("\n" + "=" * 80)
print("ПРЕОБРАЗОВАННЫЕ ДАННЫЕ (ЛОГАРИФМИЧЕСКИЕ ДОХОДНОСТИ)")
print("=" * 80)
print(f"Первый столбец: {log_returns.columns[0]}")
print(f"Форма данных: {log_returns.shape}")
print("\nПервые 5 строк:")
print(log_returns.head())

transformed_file_path = 'log_returns_dataset.xlsx'
log_returns.to_excel(transformed_file_path, sheet_name='Log_Returns', index=False)

# ADF-ТЕСТ

def adf_analysis(series, name):
    result = adfuller(series.dropna(), regression='c', autolag='AIC')
    p_value = result[1]
    n_lags = result[2]
    stationary = '+' if p_value < 0.05 else '-'
    return {
        'Индекс': name,
        'p-value': p_value,
        'Количество лагов': n_lags,
        'Стационарность': stationary
    }

results_returns = []
for col in log_returns.columns:
    results_returns.append(adf_analysis(log_returns[col], col))

df_returns = pd.DataFrame(results_returns)
df_returns['p-value'] = df_returns['p-value'].apply(lambda x: f"{x:.10e}")

adf_results_file = 'adf_test_results.xlsx'
df_returns.to_excel(adf_results_file, index=False)
df = df.iloc[:, 1:]  # удаляет первый столбец (индекс 0)
print("\n" + "=" * 80)
print(df_returns[['Индекс', 'p-value', 'Количество лагов', 'Стационарность']].to_string(index=False))
