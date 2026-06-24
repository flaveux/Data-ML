"""
График IMOEX: 2023 - начало 2026 (8 недель 2026)
Три линии:
  1) Исторические данные (01.01.2023 - 28.12.2025) - из DATASET.xlsx
  2) Реальные значения IMOEX за 8 недель 2026
  3) Прогноз модели Random Forest (R^2 = 0.72) за те же 8 недель

Перед запуском:
  pip install pandas matplotlib openpyxl
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

# -------------------------------------------------------------------
# 0. Шрифт Times New Roman
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
available = {f.name for f in font_manager.fontManager.ttflist}
if 'Times New Roman' not in available:
    print("Внимание: шрифт 'Times New Roman' не найден в системе, "
          "будет использован шрифт по умолчанию.")

# -------------------------------------------------------------------
# 1. Загрузка исторических данных из DATASET.xlsx
# -------------------------------------------------------------------
DATASET_PATH = "DATASET.xlsx"   # укажите путь к вашему файлу

df_hist = pd.read_excel(DATASET_PATH, sheet_name="Sheet1")

# В файле нет колонки с датой - есть только значения IMOEX (и других
# индексов) с недельным шагом. По стыковке последнего значения
# (28.12.2025 -> 2766.62) с первым значением 2026 года (04.01.2026 ->
# 2724.85) установлено, что ряд начинается с 01.01.2023 и идёт с шагом
# 7 дней.
df_hist["Дата"] = pd.date_range(start="2023-01-01", periods=len(df_hist), freq="7D")
df_hist = df_hist[["Дата", "IMOEX"]].sort_values("Дата")

# -------------------------------------------------------------------
# 2. Данные за 8 недель 2026: реальные значения и прогноз RF
# -------------------------------------------------------------------
data_2026 = {
    "Дата": [
        "04.01.2026", "11.01.2026", "18.01.2026", "25.01.2026",
        "01.02.2026", "08.02.2026", "15.02.2026", "22.02.2026",
    ],
    "Реальные": [
        2724.85, 2733.75, 2777.29, 2782.74,
        2735.43, 2776.34, 2780.60, 2799.14,
    ],
    "Прогноз_RF": [
        2742.09, 2700.37, 2718.07, 2745.84,
        2754.13, 2731.62, 2743.18, 2753.67,
    ],
}

df_2026 = pd.DataFrame(data_2026)
df_2026["Дата"] = pd.to_datetime(df_2026["Дата"], dayfirst=True)
df_2026 = df_2026.sort_values("Дата")

# Чтобы линии за 2026 год визуально "стыковались" с историческим рядом,
# добавляем последнюю историческую точку в начало каждой из них.
last_hist_point = df_hist.iloc[[-1]].rename(columns={"IMOEX": "value"})

real_2026 = pd.concat([
    last_hist_point[["Дата", "value"]],
    df_2026[["Дата", "Реальные"]].rename(columns={"Реальные": "value"})
], ignore_index=True)

forecast_2026 = pd.concat([
    last_hist_point[["Дата", "value"]],
    df_2026[["Дата", "Прогноз_RF"]].rename(columns={"Прогноз_RF": "value"})
], ignore_index=True)

# -------------------------------------------------------------------
# 3. Построение графика
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))

# Линия 1: исторические данные (2023 - конец 2025)
ax.plot(
    df_hist["Дата"], df_hist["IMOEX"],
    color="#1f77b4", linewidth=1.8, label="Исторические данные IMOEX (2023–2025)"
)

# Линия 2: реальные значения за 8 недель 2026
ax.plot(
    real_2026["Дата"], real_2026["value"],
    color="#2ca02c", linewidth=2.2,
    label="Реальные значения IMOEX (2026)"
)

# Линия 3: прогноз модели Random Forest за те же 8 недель
ax.plot(
    forecast_2026["Дата"], forecast_2026["value"],
    color="#d62728", linewidth=2.2,
    label="Прогноз Random Forest, R² = 0.72 (2026)"
)

# -------------------------------------------------------------------
# 4. Оформление
# -------------------------------------------------------------------
ax.set_title(
    "Индекс IMOEX: исторические данные, реальные значения и прогноз\n"
    "Random Forest (январь–февраль 2026)",
    fontsize=15, fontname="Times New Roman"
)
ax.set_xlabel("Дата", fontsize=12, fontname="Times New Roman")
ax.set_ylabel("Значение индекса IMOEX", fontsize=12, fontname="Times New Roman")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
fig.autofmt_xdate(rotation=45)

legend = ax.legend(loc="upper left", fontsize=12)
for text in legend.get_texts():
    text.set_fontname("Times New Roman")

ax.tick_params(axis="both", labelsize=10)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname("Times New Roman")

ax.grid(True, linestyle=":", alpha=0.5)

# Запас сверху, чтобы легенда не перекрывала линии графика
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.12)

plt.tight_layout()
plt.savefig("imoex_chart.png", dpi=300)
plt.show()