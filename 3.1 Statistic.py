import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, mode

file_path = 'DATASET.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

columns_check = ['IMOEX', 'S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000',
                 'Bovespa', 'Euro Stoxx 50', 'DAX', 'CAC 40', 'FTSE 100',
                 'AEX', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225']

existing_columns = [col for col in columns_check if col in df.columns]
df_clean = df[existing_columns].dropna()

stats_data = []

for col in existing_columns:
    data = df_clean[col].dropna()

    mode_val = mode(data, keepdims=True)[0][0] if len(mode(data, keepdims=True)[0]) > 0 else np.nan

    stats_data.append({
        'Индекс': col,
        'Среднее': data.mean(),
        'Мода': mode_val,
        'Медиана': data.median(),
        'Q1 (25%)': data.quantile(0.25),
        'Q3 (75%)': data.quantile(0.75),
        'Станд. отклонение': data.std(),
        'Дисперсия': data.var(),
        'Асимметрия (Skewness)': skew(data),
        'Эксцесс (Kurtosis)': kurtosis(data),
        'Коэф. вариации (CV), %': (data.std() / data.mean()) * 100
    })


df_stats = pd.DataFrame(stats_data)

print("\n" + "=" * 120)
print("ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ ПО ИНДЕКСАМ (без min/max)")
print("=" * 120)
print(df_stats.round(4).to_string(index=False))

df_stats_transposed = df_stats.set_index('Индекс').T
print("\n" + "=" * 120)
print("ТРАНСПОНИРОВАННАЯ ТАБЛИЦА (статистики в строках)")
print("=" * 120)
print(df_stats_transposed.round(4).to_string())


df_stats.to_csv('descriptive_statistics.csv', index=False, encoding='utf-8-sig')
df_stats.to_excel('descriptive_statistics.xlsx', index=False)

df_stats_transposed.to_csv('descriptive_statistics_transposed.csv', encoding='utf-8-sig')
df_stats_transposed.to_excel('descriptive_statistics_transposed.xlsx')

