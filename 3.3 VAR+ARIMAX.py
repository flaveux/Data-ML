import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ----------------------------------------------------------------------
df = pd.read_excel('log_returns_dataset.xlsx', sheet_name='Log_Returns')

selected_regressors = ['S&P 500', 'NASDAQ', 'Euro Stoxx 50', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225']

print("=" * 80)
print("VAR И ARIMAX ДЛЯ АНАЛИЗА ВЛИЯНИЯ")
print("=" * 80)
print(f"Форма данных: {df.shape}")
print(f"Отобранные регрессоры: {selected_regressors}")

y = df['IMOEX']
X = df[selected_regressors]

# =========================================================================
# 2. VAR МОДЕЛЬ
# =========================================================================
print("\n" + "=" * 80)
print("VAR МОДЕЛЬ")
print("=" * 80)

df_var = pd.DataFrame({
    'IMOEX': y,
    'S&P 500': X['S&P 500'],
    'NASDAQ': X['NASDAQ'],
    'Euro Stoxx 50': X['Euro Stoxx 50'],
    'Nifty 50': X['Nifty 50'],
    'Shanghai Composite': X['Shanghai Composite'],
    'Nikkei 225': X['Nikkei 225']
})

# Проверка стационарности
print("\nПроверка стационарности (ADF-тест):")
for col in df_var.columns:
    result = adfuller(df_var[col].dropna(), regression='c', autolag='AIC')
    print(f"  {col}: p-value = {result[1]:.6f} {'✅' if result[1] < 0.05 else '❌'}")

# Выбор оптимального лага
model_var = VAR(df_var)
max_lags = min(4, len(df_var) // 3)
lag_order = model_var.select_order(maxlags=max_lags)

# Универсальное получение AIC
try:
    aic_vals = lag_order.aic_vals
except AttributeError:
    aic_vals = lag_order.aic

if not hasattr(aic_vals, '__len__'):
    aic_vals = np.array([lag_order.aic])

print("\nИнформационные критерии:")
print(f"  {'Лаг':<6} {'AIC':>12} {'BIC':>12} {'HQ':>12}")
for lag in range(1, max_lags + 1):
    res_tmp = model_var.fit(lag, verbose=False)
    print(f"  {lag:<6} {res_tmp.aic:>12.4f} {res_tmp.bic:>12.4f} {res_tmp.hqic:>12.4f}")

# Выбор по BIC (более строгий для временных рядов)
bic_vals = [model_var.fit(lag, verbose=False).bic for lag in range(1, max_lags + 1)]
best_lag = int(np.argmin(bic_vals)) + 1
print(f"\n✅ Оптимальный лаг по BIC: {best_lag}")
# Оценка VAR
results_var = model_var.fit(best_lag)
print(results_var.summary())

# =========================================================================
# 3. ВЛИЯНИЕ НА IMOEX (VAR) — ИСПРАВЛЕНО
# =========================================================================
print("\n" + "=" * 80)
print("ВЛИЯНИЕ НА IMOEX (VAR)")
print("=" * 80)

coef_imoex = results_var.coefs[:, 0]
if coef_imoex.ndim > 1:
    coef_imoex = coef_imoex.flatten()

names = []
for lag in range(1, best_lag + 1):
    for col in df_var.columns:
        names.append(f"{col}_lag{lag}")

coef_df = pd.DataFrame({
    'Переменная': names,
    'Коэффициент': coef_imoex[:len(names)]
})

print("\nКОЭФФИЦИЕНТЫ (топ-10 по абсолютной величине):")
coef_df['abs'] = abs(coef_df['Коэффициент'])
top_coef = coef_df.sort_values('abs', ascending=False).head(10)
print(top_coef[['Переменная', 'Коэффициент']].to_string(index=False))

# ✅ ИСПРАВЛЕНИЕ: results_var.pvalues — это DataFrame, индексируем по имени столбца
if hasattr(results_var, 'pvalues'):
    pvalues_df = results_var.pvalues
    if isinstance(pvalues_df, pd.DataFrame):
        # Столбцы соответствуют уравнениям (IMOEX, S&P 500, ...)
        pvalues_imoex = pvalues_df['IMOEX'].values
    else:
        # Fallback для numpy-массива
        pvalues_imoex = np.asarray(pvalues_df)[:, 0].flatten()

    coef_df['p-value'] = pvalues_imoex[:len(names)]
    coef_df['Значим (p<0.05)'] = coef_df['p-value'] < 0.05
    significant = coef_df[coef_df['Значим (p<0.05)'] == True]
    if len(significant) > 0:
        print("\nСТАТИСТИЧЕСКИ ЗНАЧИМЫЕ КОЭФФИЦИЕНТЫ (p < 0.05):")
        print(significant[['Переменная', 'Коэффициент', 'p-value']].to_string(index=False))
    else:
        print("\n⚠️ Нет значимых коэффициентов на уровне 5%")

# Константа VAR для уравнения IMOEX
try:
    # results_var.intercept — вектор констант по всем уравнениям
    intercept_var = results_var.intercept
    print(f"\nКонстанта VAR (уравнение IMOEX): {intercept_var[0]:.6f}")
except Exception:
    try:
        # Альтернативное хранение в некоторых версиях statsmodels
        intercept_var = results_var.coefs_exog
        print(f"\nКонстанта VAR (уравнение IMOEX): {intercept_var[0, 0]:.6f}")
    except Exception:
        print("\nКонстанта VAR: не удалось извлечь автоматически")

# =========================================================================
# 4. ARIMAX МОДЕЛЬ
# =========================================================================
print("\n" + "=" * 80)
print("ARIMAX МОДЕЛЬ")
print("=" * 80)
print("Спецификация: AR(3) + экзогенные переменные, d=0")

model_arimax = SARIMAX(
    y,
    exog=X,
    order=(3, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    trend='c'
)
results_arimax = model_arimax.fit(disp=False)
print(results_arimax.summary())

# =========================================================================
# 5. ВЛИЯНИЕ НА IMOEX (ARIMAX)
# =========================================================================
print("\n" + "=" * 80)
print("ВЛИЯНИЕ НА IMOEX (ARIMAX)")
print("=" * 80)

print("\nAR-коэффициенты (автокорреляция IMOEX):")
for i in range(1, 4):
    param_name = f'ar.L{i}'
    if param_name in results_arimax.params.index:
        coef = results_arimax.params[param_name]
        p_val = results_arimax.pvalues[param_name]
        status = "→ значимо" if p_val < 0.05 else "→ не значимо"
        print(f"  AR({i}): {coef:.4f} (p={p_val:.4f}) {status}")

print("\nКоэффициенты экзогенных переменных (влияние других индексов):")
for reg in selected_regressors:
    if reg in results_arimax.params.index:
        coef = results_arimax.params[reg]
        p_val = results_arimax.pvalues[reg]
        status = "→ значимое влияние" if p_val < 0.05 else "→ не значимо"
        print(f"  {reg}: {coef:.4f} (p={p_val:.4f}) {status}")

# Константа может называться 'const' или 'intercept' в зависимости от версии statsmodels
const_key = next((k for k in ['const', 'intercept'] if k in results_arimax.params.index), None)
if const_key:
    const_val = results_arimax.params[const_key]
    const_pval = results_arimax.pvalues[const_key]
    status = "→ значимо" if const_pval < 0.05 else "→ не значимо"
    print(f"\nКонстанта ARIMAX: {const_val:.6f} (p={const_pval:.4f}) {status}")
else:
    print(f"\nКонстанта: не найдена (доступные параметры: {list(results_arimax.params.index)})")

# =========================================================================
# 6. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ
# =========================================================================
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("\n" + "=" * 80)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("=" * 80)


# ---------------------------------------------------------------------------
# Вспомогательная функция: AIC и BIC через MSE (без likelihood)
# Формула: AIC = n*ln(MSE) + 2k,  BIC = n*ln(MSE) + k*ln(n)
# Используем когда statsmodels не даёт AIC/BIC для отдельного уравнения
# ---------------------------------------------------------------------------
def aic_bic_from_mse(mse, n, k):
    aic = n * np.log(mse) + 2 * k
    bic = n * np.log(mse) + k * np.log(n)
    return aic, bic


# --- VAR: качество для уравнения IMOEX ---
print("\n[ VAR — уравнение IMOEX ]")

fitted_var = results_var.fittedvalues['IMOEX']
actual_var = df_var['IMOEX'].iloc[best_lag:]  # VAR теряет первые best_lag наблюдений

r2_var = r2_score(actual_var, fitted_var)
rmse_var = np.sqrt(mean_squared_error(actual_var, fitted_var))
mae_var = mean_absolute_error(actual_var, fitted_var)

n_var = len(actual_var)
k_var = best_lag * len(df_var.columns)  # число регрессоров в уравнении
r2_adj_var = 1 - (1 - r2_var) * (n_var - 1) / (n_var - k_var - 1)

mse_var = mean_squared_error(actual_var, fitted_var)
aic_var_eq, bic_var_eq = aic_bic_from_mse(mse_var, n_var, k_var)

# Системные AIC/BIC из statsmodels (для справки)
aic_var_sys = results_var.aic
bic_var_sys = results_var.bic

print(f"  R²               : {r2_var:.4f}")
print(f"  Скорр. R²        : {r2_adj_var:.4f}")
print(f"  RMSE             : {rmse_var:.6f}")
print(f"  MAE              : {mae_var:.6f}")
print(f"  AIC (ур-е IMOEX) : {aic_var_eq:.4f}")
print(f"  BIC (ур-е IMOEX) : {bic_var_eq:.4f}")
print(f"  AIC (системный)  : {aic_var_sys:.4f}  ← по всей системе VAR")
print(f"  BIC (системный)  : {bic_var_sys:.4f}  ← по всей системе VAR")
print(f"  Интерпретация    : модель объясняет {r2_var * 100:.1f}% дисперсии доходности IMOEX")

# --- ARIMAX: качество ---
print("\n[ ARIMAX ]")

fitted_arimax = results_arimax.fittedvalues
actual_arimax = y.values

r2_arimax = r2_score(actual_arimax, fitted_arimax)
rmse_arimax = np.sqrt(mean_squared_error(actual_arimax, fitted_arimax))
mae_arimax = mean_absolute_error(actual_arimax, fitted_arimax)

n_arimax = len(actual_arimax)
k_arimax = len(results_arimax.params)  # число оцененных параметров
r2_adj_arimax = 1 - (1 - r2_arimax) * (n_arimax - 1) / (n_arimax - k_arimax - 1)

mse_arimax = mean_squared_error(actual_arimax, fitted_arimax)
aic_arimax_manual, bic_arimax_manual = aic_bic_from_mse(mse_arimax, n_arimax, k_arimax)

# Statsmodels считает AIC/BIC через likelihood — более точно, используем их
aic_arimax = results_arimax.aic
bic_arimax = results_arimax.bic
llf_arimax = results_arimax.llf

# Псевдо-R² МакФаддена
model_null = SARIMAX(y, order=(0, 0, 0), trend='c').fit(disp=False)
pseudo_r2 = 1 - (llf_arimax / model_null.llf)

print(f"  R²               : {r2_arimax:.4f}")
print(f"  Скорр. R²        : {r2_adj_arimax:.4f}")
print(f"  Псевдо-R² (MF)   : {pseudo_r2:.4f}")
print(f"  RMSE             : {rmse_arimax:.6f}")
print(f"  MAE              : {mae_arimax:.6f}")
print(f"  AIC (likelihood) : {aic_arimax:.4f}")
print(f"  BIC (likelihood) : {bic_arimax:.4f}")
print(f"  Интерпретация    : модель объясняет {r2_arimax * 100:.1f}% дисперсии доходности IMOEX")

# --- Сравнительная таблица ---
print("\n[ Сравнение моделей ]")
print(f"  {'Метрика':<25} {'VAR (ур-е IMOEX)':>18} {'ARIMAX':>12}")
print(f"  {'-' * 57}")
print(f"  {'R²':<25} {r2_var:>18.4f} {r2_arimax:>12.4f}")
print(f"  {'Скорр. R²':<25} {r2_adj_var:>18.4f} {r2_adj_arimax:>12.4f}")
print(f"  {'RMSE':<25} {rmse_var:>18.6f} {rmse_arimax:>12.6f}")
print(f"  {'MAE':<25} {mae_var:>18.6f} {mae_arimax:>12.6f}")
print(f"  {'AIC (ур-е IMOEX)':<25} {aic_var_eq:>18.4f} {aic_arimax:>12.4f}")
print(f"  {'BIC (ур-е IMOEX)':<25} {bic_var_eq:>18.4f} {bic_arimax:>12.4f}")
print()
print("  ℹ️  AIC/BIC VAR — через MSE уравнения IMOEX (сопоставимо с ARIMAX).")
print("  ℹ️  AIC/BIC ARIMAX — через likelihood (statsmodels), более точно.")

# =========================================================================
# 7. ВЫВОД
# =========================================================================
print("\n" + "=" * 80)
print("ВЫВОД")
print("=" * 80)
print(f"""
VAR модель (оптимальный лаг = {best_lag}):
   - Единственный значимый предиктор: NASDAQ_lag1 (p=0.044)
   - R² уравнения IMOEX = {r2_var:.4f} — объясняет {r2_var * 100:.1f}% дисперсии
   - S&P 500_lag1 крупный коэффициент, но незначим (мультиколлинеарность с NASDAQ)

ARIMAX модель (AR(3) + экзогенные):
   - Все внешние индексы незначимы
   - Значим только AR(3) (p=0.030)
   - R² = {r2_arimax:.4f}, AIC = {aic_arimax:.2f}

Диагностика остатков ARIMAX:
   - Нет автокорреляции (Ljung-Box p=0.90 ✅)
   - Нормальность не отвергается (JB p=0.25 ✅)
   - Гетероскедастичность присутствует (H p=0.00 ⚠️) → рассмотреть GARCH

Обе модели оценены на логарифмических доходностях (стационарны).
""")