import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'DATASET.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

columns_check = ['IMOEX', 'S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000',
                 'Bovespa', 'Euro Stoxx 50', 'DAX', 'CAC 40', 'FTSE 100',
                 'AEX', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225']

existing_columns = [col for col in columns_check if col in df.columns]
df_clean = df[existing_columns].dropna()

n_indices = len(existing_columns)  # 14 индексов

#2 столбца × 7 строк

fig, axes = plt.subplots(7, 2, figsize=(14, 20))
fig.subplots_adjust(top=0.93)  # ← добавить (увеличивает верхнее поле)
axes = axes.flatten()  # упрощаем индексацию

colors = plt.cm.Set3(np.linspace(0, 1, n_indices))

for idx, col in enumerate(existing_columns):
    ax = axes[idx]
    data = df_clean[col].dropna()

    bp = ax.boxplot(data, vert=False, patch_artist=True, whis=1.5)

    for box in bp['boxes']:
        box.set_facecolor('lightsteelblue')
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        box.set_linewidth(1.2)

    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2.5)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)

    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markersize(3)
        flier.set_markeredgecolor('gray')
        flier.set_markerfacecolor('lightgray')
        flier.set_alpha(0.5)

    ax.set_title(col, fontsize=10, fontweight='bold', loc='left')

    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_yticks([])

    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 0))

for idx in range(len(existing_columns), len(axes)):
    axes[idx].set_visible(False)



plt.tight_layout()

plt.savefig('boxplot_2x7_all_indices.png', dpi=300, bbox_inches='tight')
plt.show()