import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

# =========================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =========================================================================
df_raw  = pd.read_excel('DATASET.xlsx')
df_pred = pd.read_excel('PREDICT.xlsx')

TARGET = 'IMOEX'
FEATURE_COLS = [
    'S&P 500', 'NASDAQ', 'Dow Jones', 'Russell 2000',
    'Bovespa', 'Euro Stoxx 50', 'DAX', 'CAC 40',
    'FTSE 100', 'AEX', 'Nifty 50', 'Shanghai Composite', 'Nikkei 225'
]
DATES = [
    '04.01.2026', '11.01.2026', '18.01.2026', '25.01.2026',
    '01.02.2026', '08.02.2026', '15.02.2026', '22.02.2026'
]

ALL_COLS    = FEATURE_COLS + [TARGET]   # 14 переменных
y_real      = df_pred[TARGET].values
last_price  = df_raw[TARGET].iloc[-1]

print("=" * 80)
print("LSTM — ПРОГНОЗ IMOEX НА 8 НЕДЕЛЬ")
print(f"Признаков : {len(ALL_COLS)} (13 регрессоров + IMOEX)")
print(f"Выборка   : {len(df_raw)} наблюдений")
print("=" * 80)

# =========================================================================
# 2. МАСШТАБИРОВАНИЕ
# =========================================================================
data = df_raw[ALL_COLS].values   # (157, 14)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Индекс целевой переменной (IMOEX — последняя колонка)
target_idx = ALL_COLS.index(TARGET)

# =========================================================================
# 3. ФОРМИРОВАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ (окно = LOOKBACK недель)
# =========================================================================
LOOKBACK = 4   # модель смотрит на 4 предыдущие недели

def make_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])        # все 14 переменных за lookback шагов
        y.append(data[i, target_idx])             # IMOEX на шаге i
    return np.array(X), np.array(y)

X_seq, y_seq = make_sequences(data_scaled, LOOKBACK)

print(f"\nФорма X_seq : {X_seq.shape}  (наблюдения, lookback, признаки)")
print(f"Форма y_seq : {y_seq.shape}")

# Разбивка 80/20 (временная)
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"\nРазбивка train/test (80/20):")
print(f"  Train : {len(X_train)} последовательностей")
print(f"  Test  : {len(X_test)}  последовательностей")

# =========================================================================
# 4. АРХИТЕКТУРА LSTM
# =========================================================================
print("\n" + "=" * 80)
print("АРХИТЕКТУРА LSTM")
print("=" * 80)

model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(LOOKBACK, len(ALL_COLS))),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# =========================================================================
# 5. ОБУЧЕНИЕ
# =========================================================================
print("\n" + "=" * 80)
print("ОБУЧЕНИЕ")
print("=" * 80)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nОбучение завершено на эпохе: {len(history.history['loss'])}")

# =========================================================================
# 6. МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ
# =========================================================================
print("\n" + "=" * 80)
print("МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ (20%)")
print("=" * 80)

# Предсказания в масштабированных единицах
y_pred_scaled = model.predict(X_test).flatten()

# Восстановление в исходные уровни
def inverse_target(scaled_vals, scaler, target_idx, n_features):
    """Обратное масштабирование только целевой переменной"""
    dummy = np.zeros((len(scaled_vals), n_features))
    dummy[:, target_idx] = scaled_vals
    return scaler.inverse_transform(dummy)[:, target_idx]

y_pred_levels = inverse_target(y_pred_scaled, scaler, target_idx, len(ALL_COLS))
y_test_levels = inverse_target(y_test,        scaler, target_idx, len(ALL_COLS))

r2   = r2_score(y_test_levels, y_pred_levels)
rmse = np.sqrt(mean_squared_error(y_test_levels, y_pred_levels))
mae  = mean_absolute_error(y_test_levels, y_pred_levels)

print(f"\n  R²   : {r2:.4f}  → объясняет {r2*100:.1f}% дисперсии уровня IMOEX")
print(f"  RMSE : {rmse:.2f} пунктов")
print(f"  MAE  : {mae:.2f} пунктов")

# =========================================================================
# 7. ПЕРЕОБУЧЕНИЕ НА ВСЕЙ ВЫБОРКЕ ДЛЯ ПРОГНОЗА
# =========================================================================
print("\n" + "=" * 80)
print("ПЕРЕОБУЧЕНИЕ НА ПОЛНОЙ ВЫБОРКЕ ДЛЯ ПРОГНОЗА")
print("=" * 80)

model.fit(
    X_seq, y_seq,
    epochs=200,
    batch_size=16,
    callbacks=[EarlyStopping(monitor='loss', patience=20,
                             restore_best_weights=True, verbose=0)],
    verbose=0
)
print("  Готово.")

# =========================================================================
# 8. БЛОК А: РЕКУРСИВНЫЙ ПРОГНОЗ НА 8 НЕДЕЛЬ
# Каждый следующий шаг: предсказанный IMOEX → в окно следующего шага
# Регрессоры фиксированы на последних известных значениях
# =========================================================================
print("\n" + "=" * 80)
print("БЛОК А: РЕКУРСИВНЫЙ ПРОГНОЗ (без PREDICT)")
print("Регрессоры зафиксированы на последних значениях DATASET")
print("Предсказанный IMOEX → в окно следующего шага")
print("=" * 80)

# Стартовое окно — последние LOOKBACK строк из обучающей выборки
window = data_scaled[-LOOKBACK:].copy()   # (LOOKBACK, 14)

rec_forecasts = []
for step in range(8):
    X_input = window[np.newaxis, :, :]         # (1, LOOKBACK, 14)
    pred_s  = model.predict(X_input, verbose=0)[0, 0]

    # Восстанавливаем уровень
    pred_level = inverse_target([pred_s], scaler, target_idx, len(ALL_COLS))[0]
    rec_forecasts.append(round(pred_level, 2))

    # Сдвигаем окно: новая строка = предыдущая последняя строка,
    # но IMOEX заменяем предсказанным
    new_row = window[-1].copy()
    new_row[target_idx] = pred_s
    window = np.vstack([window[1:], new_row])

print(f"\n  Прогнозы: {rec_forecasts}")

# =========================================================================
# 9. БЛОК Б: EX-POST ПРОГНОЗ НА 8 НЕДЕЛЬ
# Реальные значения регрессоров из PREDICT + предсказанный IMOEX
# =========================================================================
print("\n" + "=" * 80)
print("БЛОК Б: EX-POST ПРОГНОЗ (регрессоры из PREDICT)")
print("Регрессоры = реальные значения из PREDICT")
print("Предсказанный IMOEX → в окно следующего шага")
print("=" * 80)

# Масштабируем PREDICT-данные тем же scaler
pred_data = df_pred[ALL_COLS].values
pred_scaled = scaler.transform(pred_data)

# Стартовое окно — последние LOOKBACK строк из DATASET
window_exp = data_scaled[-LOOKBACK:].copy()

exp_forecasts = []
for step in range(8):
    X_input = window_exp[np.newaxis, :, :]
    pred_s  = model.predict(X_input, verbose=0)[0, 0]

    pred_level = inverse_target([pred_s], scaler, target_idx, len(ALL_COLS))[0]
    exp_forecasts.append(round(pred_level, 2))

    # Новая строка: реальные регрессоры из PREDICT + предсказанный IMOEX
    new_row = pred_scaled[step].copy()
    new_row[target_idx] = pred_s
    window_exp = np.vstack([window_exp[1:], new_row])

print(f"\n  Прогнозы: {exp_forecasts}")

# =========================================================================
# 10. ИТОГОВАЯ ТАБЛИЦА
# =========================================================================
print("\n" + "=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА ПРОГНОЗОВ vs РЕАЛЬНЫЕ ЗНАЧЕНИЯ")
print("=" * 80)

result_df = pd.DataFrame({
    'Дата':                DATES,
    'Реальный IMOEX':      y_real,
    'LSTM (рекурсивный)':  rec_forecasts,
    'LSTM (ex-post)':      exp_forecasts,
})
print(result_df.to_string(index=False))

print("\n" + "=" * 80)
print("ПАРАМЕТРЫ МОДЕЛИ")
print("=" * 80)
print(f"  Архитектура : LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)")
print(f"  Lookback    : {LOOKBACK} недели")
print(f"  Оптимизатор : Adam (lr=0.001)")
print(f"  Loss        : MSE")
print(f"  EarlyStopping: patience=20, monitor=val_loss")
print(f"  Признаков   : {len(ALL_COLS)} (все регрессоры + IMOEX)")