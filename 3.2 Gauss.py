import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

df = pd.read_excel('log_returns_dataset.xlsx', sheet_name='Log_Returns')

print("=" * 90)
print("ПРОВЕРКА УСЛОВИЙ ГАУССА-МАРКОВА")

y = df['IMOEX']                          # целевая переменная
X = df.drop(columns=['IMOEX'])           # регрессоры
X_const = sm.add_constant(X)             # добавляем константу для регрессии

print(f"\nРазмер выборки: {len(y)} наблюдений")
print(f"Количество регрессоров: {X.shape[1]}")

model = sm.OLS(y, X_const).fit()
residuals = model.resid

print("\n" + "=" * 90)
print("РЕЗУЛЬТАТЫ РЕГРЕССИИ (IMOEX ~ все индексы)")
print("=" * 90)
print(model.summary())

# УСЛОВИЕ 1: M(ε_t) = 0

print("\n" + "=" * 90)
print("УСЛОВИЕ 1: Математическое ожидание ошибок равно 0")
print("=" * 90)
mean_residual = np.mean(residuals)
print(f"Среднее остатков: {mean_residual:.15f}")

if abs(mean_residual) < 1e-10:
    print("УСЛОВИЕ ВЫПОЛНЕНО: M(ε) = 0")
else:
    print("УСЛОВИЕ НЕ ВЫПОЛНЕНО")

# УСЛОВИЕ 2: Экзогенность (M(ε|X) = 0)

print("\n" + "=" * 90)
print("УСЛОВИЕ 2: Экзогенность")
print("=" * 90)

correlations = []
for col in X.columns:
    corr = np.corrcoef(X[col], residuals)[0, 1]
    correlations.append((col, corr))

corr_df = pd.DataFrame(correlations, columns=['Регрессор', 'Корреляция с остатками'])
print(corr_df.to_string(index=False))

max_corr = abs(corr_df['Корреляция с остатками']).max()
n = len(residuals)
threshold = 1.96 / np.sqrt(n)
print(f"\nПорог значимости (95%): ±{threshold:.4f}")

if max_corr < threshold:
    print("УСЛОВИЕ ВЫПОЛНЕНО")
else:
    print("УСЛОВИЕ НАРУШЕНО")

# УСЛОВИЕ 3: Гомоскедастичность (тест Уайта)

print("\n" + "=" * 90)
print("УСЛОВИЕ 3: Гомоскедастичность (тест Уайта)")
print("=" * 90)

white_test = het_white(residuals, X_const)
print(f"LM-статистика: {white_test[0]:.4f}")
print(f"p-value:       {white_test[1]:.6f}")
print(f"F-статистика:  {white_test[2]:.4f}")
print(f"p-value (F):   {white_test[3]:.6f}")

if white_test[1] < 0.05:
    print("ГЕТЕРОСКЕДАСТИЧНОСТЬ")
else:
    print("гомоскедастичность")

# УСЛОВИЕ 4: Отсутствие автокорреляции (тест Бреуша-Годфри)

print("\n" + "=" * 90)
print("УСЛОВИЕ 4: Отсутствие автокорреляции (тест Бреуша-Годфри)")
print("=" * 90)

bg_test_1 = acorr_breusch_godfrey(model, nlags=1)
print("--- Тест с 1 лагом ---")
print(f"LM-статистика: {bg_test_1[0]:.4f}")
print(f"p-value:       {bg_test_1[1]:.6f}")

bg_test_3 = acorr_breusch_godfrey(model, nlags=3)
print("\n--- Тест с 3 лагами ---")
print(f"LM-статистика: {bg_test_3[0]:.4f}")
print(f"p-value:       {bg_test_3[1]:.6f}")

if bg_test_1[1] < 0.05 or bg_test_3[1] < 0.05:
    print("\nприсутствует автокорреляция")
else:
    print("\nавтокорреляция отсутствует")


# УСЛОВИЕ 5: Отсутствие мультиколлинеарности

print("\n" + "=" * 90)
print("УСЛОВИЕ 5: Отсутствие мультиколлинеарности")
print("=" * 90)

# Расчёт VIF (только на X, без const)
vif_data = []
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    vif_data.append((X.columns[i], vif))

vif_df = pd.DataFrame(vif_data, columns=['Регрессор', 'VIF'])
vif_df = vif_df.sort_values('VIF', ascending=False)
print("\nVIF (Variance Inflation Factor):")
print(vif_df.to_string(index=False))

# Определитель корреляционной матрицы
corr_matrix = X.corr()
det_value = np.linalg.det(corr_matrix.values)
print(f"\nОпределитель корреляционной матрицы: {det_value:.10e}")

# Оценка
max_vif = vif_df['VIF'].max()
if max_vif > 10:
    print(f"\nсильная мультиколлинеарность (max VIF = {max_vif:.2f} > 10)")

elif max_vif > 5:
    print(f"\nумеренная мультиколлинеарность (max VIF = {max_vif:.2f} > 5)")
else:
    print(f"\nмультиколлинеарности нет (max VIF = {max_vif:.2f} < 5)")


# ИТОГОВАЯ ТАБЛИЦА

print("\n" + "=" * 90)
print("ИТОГОВАЯ ТАБЛИЦА ПРОВЕРКИ УСЛОВИЙ ГАУССА-МАРКОВА")
print("=" * 90)

# Определение статуса для условия 5
if max_vif < 5:
    status_5 = "Выполнено"
elif max_vif < 10:
    status_5 = "Умеренная"
else:
    status_5 = "Нарушено"

results_summary = pd.DataFrame({
    'Условие': [
        '1. M(ε) = 0',
        '2. Экзогенность (M(ε|X)=0)',
        '3. Гомоскедастичность (σ²)',
        '4. Нет автокорреляции (Cov=0)',
        '5. Нет мультиколлинеарности'
    ],
    'Результат': [
        'Выполнено' if abs(mean_residual) < 1e-10 else '-',
        'Выполнено' if max_corr < threshold else '-',
        'Гетероскедастичность' if white_test[1] < 0.05 else '-',
        'Есть автокорреляция' if bg_test_1[1] < 0.05 else '-',
        status_5
    ],
    'Значение / p-value': [
        f"{mean_residual:.2e}",
        f"max|corr|={max_corr:.4f}",
        f"p = {white_test[1]:.6f}",
        f"p = {bg_test_1[1]:.6f}",
        f"max VIF = {max_vif:.2f}"
    ]
})

print(results_summary.to_string(index=False))
results_summary.to_excel('gauss_markov_check.xlsx', index=False)
vif_df.to_excel('vif_results.xlsx', index=False)
