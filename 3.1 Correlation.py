import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


file_path = 'log_returns_dataset.xlsx'
df = pd.read_excel(file_path)

columns_check = ['IMOEX', 'S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000',
                 'Bovespa', 'Euro Stoxx 50', 'DAX', 'CAC 40', 'FTSE 100',
                 'AEX', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225']

existing_columns = [col for col in columns_check if col in df.columns]
df_clean = df[existing_columns].dropna()

corr_spearman = df_clean.corr(method='spearman')

print("\n" + "=" * 100)
print("КОРРЕЛЯЦИОННАЯ МАТРИЦА СПИРМЕНА (по исходным ценам)")
print("=" * 100)
print(corr_spearman.round(4).to_string())

fig, ax = plt.subplots(figsize=(14, 12))

# Маска для верхнего треугольника
mask = np.triu(np.ones_like(corr_spearman, dtype=bool))

sns.heatmap(corr_spearman,
            annot=True,  # показывать значения
            fmt='.2f',  # 2 знака после запятой
            cmap='RdBu_r',  # красно-синяя цветовая схема
            center=0,  # центр палитры (0)
            vmin=-1, vmax=1,  # границы корреляции
            square=True,  # квадратные ячейки
            linewidths=0.5,  # линии между ячейками
            ax=ax)



plt.tight_layout()
plt.savefig('correlation_spearman_prices.png', dpi=300, bbox_inches='tight')
plt.show()


def spearman_significance(df, alpha=0.05):
    """
    Рассчитывает корреляцию Спирмена и проверяет её значимость.
    Возвращает матрицы: коэффициент корреляции, p-value, значимость.
    """
    n = df.shape[1]
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1
                p_matrix[i, j] = 0
            else:
                corr, p_value = spearmanr(df.iloc[:, i], df.iloc[:, j])
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_value

    # Создаём DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)
    p_df = pd.DataFrame(p_matrix, index=df.columns, columns=df.columns)

    # Матрица значимости (True = значимо на уровне alpha)
    sig_df = p_df < alpha

    return corr_df, p_df, sig_df


# Расчёт
corr_spearman_full, p_values_spearman, significance_spearman = spearman_significance(df_clean, alpha=0.05)

print("\n" + "=" * 100)
print("ПРОВЕРКА ЗНАЧИМОСТИ КОРРЕЛЯЦИИ СПИРМЕНА (p-value, α=0.05)")
print("=" * 100)
print("\nМатрица p-value (Спирмен):")
print(p_values_spearman.round(6).to_string())

print("\n" + "=" * 100)
print("МАТРИЦА ЗНАЧИМОСТИ (True = корреляция статистически значима)")
print("=" * 100)
print(significance_spearman.to_string())

print("\n" + "=" * 100)
print("КОРРЕЛЯЦИЯ СПИРМЕНА С IMOEX И СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ")
print("=" * 100)

if 'IMOEX' in corr_spearman_full.columns:
    imoex_corr_spearman = pd.DataFrame({
        'Индекс': corr_spearman_full.index,
        'Корреляция Спирмена с IMOEX': corr_spearman_full['IMOEX'].values,
        'p-value': p_values_spearman['IMOEX'].values,
        'Значимо при α=0.05': significance_spearman['IMOEX'].values
    }).sort_values('Корреляция Спирмена с IMOEX', ascending=False)

    print(imoex_corr_spearman.round(6).to_string(index=False))

# корреляция Пирсона для сравнения
corr_pearson = df_clean.corr(method='pearson')

print("\n" + "=" * 100)
print("СРАВНЕНИЕ КОРРЕЛЯЦИИ ПИРСОНА И СПИРМЕНА (по IMOEX)")
print("=" * 100)

if 'IMOEX' in corr_pearson.columns:
    comparison = pd.DataFrame({
        'Индекс': corr_pearson.index,
        'Пирсон (линейная)': corr_pearson['IMOEX'].values,
        'Спирмен (ранговая)': corr_spearman_full['IMOEX'].values,
        'Разница': np.abs(corr_pearson['IMOEX'].values - corr_spearman_full['IMOEX'].values)
    }).sort_values('Спирмен (ранговая)', ascending=False)

    print(comparison.round(4).to_string(index=False))

print("\n" + "=" * 100)
print("ВСЕ СТАТИСТИЧЕСКИ ЗНАЧИМЫЕ КОРРЕЛЯЦИИ СПИРМЕНА (p-value < 0.05)")
print("=" * 100)

significant_pairs_spearman = []
for i in range(len(df_clean.columns)):
    for j in range(i + 1, len(df_clean.columns)):
        if significance_spearman.iloc[i, j]:
            significant_pairs_spearman.append({
                'Индекс 1': df_clean.columns[i],
                'Индекс 2': df_clean.columns[j],
                'Корреляция Спирмена': corr_spearman_full.iloc[i, j],
                'p-value': p_values_spearman.iloc[i, j]
            })

sig_pairs_spearman_df = pd.DataFrame(significant_pairs_spearman)
sig_pairs_spearman_df = sig_pairs_spearman_df.sort_values('Корреляция Спирмена', ascending=False)

print(
    f"\nВсего значимых пар: {len(sig_pairs_spearman_df)} из {len(df_clean.columns) * (len(df_clean.columns) - 1) // 2}")
print("\nТоп-20 самых сильных значимых корреляций Спирмена:")
print(sig_pairs_spearman_df.head(20).round(6).to_string(index=False))


print("\n" + "=" * 100)
print("КРАТКОЕ РЕЗЮМЕ ПО РЕЗУЛЬТАТАМ КОРРЕЛЯЦИОННОГО АНАЛИЗА (СПИРМЕН)")
print("=" * 100)

n_pairs = len(df_clean.columns) * (len(df_clean.columns) - 1) // 2
n_sig = len(sig_pairs_spearman_df)
n_not_sig = n_pairs - n_sig

print(f"Всего пар индексов: {n_pairs}")
print(f"Статистически значимых корреляций Спирмена (p < 0.05): {n_sig} ({n_sig / n_pairs * 100:.1f}%)")
print(f"Незначимых корреляций: {n_not_sig} ({n_not_sig / n_pairs * 100:.1f}%)")

# Максимальные корреляции
max_corr = sig_pairs_spearman_df.iloc[0]
print(
    f"\nМаксимальная положительная корреляция Спирмена: {max_corr['Индекс 1']} — {max_corr['Индекс 2']} = {max_corr['Корреляция Спирмена']:.4f}")

min_corr = sig_pairs_spearman_df.iloc[-1]
print(
    f"Минимальная (отрицательная) корреляция Спирмена: {min_corr['Индекс 1']} — {min_corr['Индекс 2']} = {min_corr['Корреляция Спирмена']:.4f}")