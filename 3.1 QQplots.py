import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


file_path = 'DATASET.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')


columns_check = ['IMOEX', 'S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000',
                 'Bovespa', 'Euro Stoxx 50', 'DAX', 'CAC 40', 'FTSE 100',
                 'AEX', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225']


existing_columns = [col for col in columns_check if col in df.columns]
df_clean = df[existing_columns].dropna()

# ПОСТРОЕНИЕ QQ-PLOT: 2 столбца × 7 строк

fig, axes = plt.subplots(7, 2, figsize=(14, 20))  # 7 строк, 2 столбца
axes = axes.flatten()  # упрощаем индексацию

for idx, col in enumerate(existing_columns):
    ax = axes[idx]
    data = df_clean[col].dropna()

    # QQ-plot (сравнение с нормальным распределением)
    stats.probplot(data, dist="norm", plot=ax)


    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel(' ', fontsize=9)
    ax.set_ylabel(' ', fontsize=9)

    #
    ax.get_lines()[0].set_color('steelblue')  # точки
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('red')  # линия
    ax.get_lines()[1].set_linewidth(2)

plt.tight_layout()

plt.savefig('qqplot_2x7_all_indices.png', dpi=300, bbox_inches='tight')
plt.show()