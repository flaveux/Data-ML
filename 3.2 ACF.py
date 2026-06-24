import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

file_path = 'log_returns_dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Log_Returns')

print("=" * 80)
print("ЗАГРУЖЕННЫЕ ДАННЫЕ")
print("=" * 80)
print(f"Форма данных: {df.shape}")
print(f"Колонки: {list(df.columns)}")

fig_acf, axes_acf = plt.subplots(7, 2, figsize=(14, 20))
axes_acf = axes_acf.flatten()

for idx, col in enumerate(df.columns):
    plot_acf(df[col].dropna(), lags=20, ax=axes_acf[idx], alpha=0.05)
    axes_acf[idx].set_title(f'{col}', fontsize=10)
    axes_acf[idx].set_xlabel('Лаг')
    axes_acf[idx].set_ylabel('Автокорреляция')
    axes_acf[idx].grid(True, alpha=0.3)

for idx in range(len(df.columns), len(axes_acf)):
    axes_acf[idx].set_visible(False)


plt.tight_layout()
plt.savefig('acf_plots.png', dpi=300, bbox_inches='tight')
plt.show()

fig_pacf, axes_pacf = plt.subplots(7, 2, figsize=(14, 20))
axes_pacf = axes_pacf.flatten()

for idx, col in enumerate(df.columns):
    plot_pacf(df[col].dropna(), lags=20, ax=axes_pacf[idx], alpha=0.05, method='ywm')
    axes_pacf[idx].set_title(f'{col}', fontsize=10)
    axes_pacf[idx].set_xlabel('Лаг')
    axes_pacf[idx].set_ylabel('Частная автокорреляция')
    axes_pacf[idx].grid(True, alpha=0.3)

for idx in range(len(df.columns), len(axes_pacf)):
    axes_pacf[idx].set_visible(False)


plt.tight_layout()
plt.savefig('pacf_plots.png', dpi=300, bbox_inches='tight')
plt.show()


# ВЫВОД ЗНАЧИМЫХ ЛАГОВ

from statsmodels.tsa.stattools import acf, pacf

print("\n" + "=" * 80)
print("СТАТИСТИЧЕСКИ ЗНАЧИМЫЕ КОЭФФИЦИЕНТЫ АВТОКОРРЕЛЯЦИИ")
print("=" * 80)

for col in df.columns:
    series = df[col].dropna()
    n = len(series)
    threshold = 1.96 / np.sqrt(n)

    acf_values = acf(series, nlags=10)
    acf_sig = [i for i, val in enumerate(acf_values) if abs(val) > threshold and i > 0]

    pacf_values = pacf(series, nlags=10, method='ywm')
    pacf_sig = [i for i, val in enumerate(pacf_values) if abs(val) > threshold and i > 0]

    print(f"\n--- {col} ---")
    print(f"  Значимые лаги ACF:  {acf_sig if acf_sig else 'нет'}")
    print(f"  Значимые лаги PACF: {pacf_sig if pacf_sig else 'нет'}")