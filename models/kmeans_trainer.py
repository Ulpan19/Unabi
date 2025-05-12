import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Пути
DATA_PATH = os.path.join("data", "accident_data.csv")
MODEL_PATH = os.path.join("models", "kmeans_model.pkl")
SCALER_PATH = os.path.join("models", "kmeans_scaler.pkl")

# Загрузка данных
df = pd.read_csv(DATA_PATH)

# Кодирование категориальных признаков
le_car = LabelEncoder()
le_weather = LabelEncoder()
le_time = LabelEncoder()

df["car_type"] = le_car.fit_transform(df["car_type"])
df["weather"] = le_weather.fit_transform(df["weather"])
df["time_of_day"] = le_time.fit_transform(df["time_of_day"])

# Подготовка данных
X = df[["age", "experience", "car_type", "weather", "time_of_day", "speeding", "seatbelt", "alcohol"]]

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # n_init=10 чтобы было стабильнее
kmeans.fit(X_scaled)

# Сохраняем модель KMeans и scaler отдельно
joblib.dump(kmeans, MODEL_PATH)
print(f"[✔] KMeans модель сохранена в {MODEL_PATH}")

joblib.dump(scaler, SCALER_PATH)
print(f"[✔] Scaler сохранён в {SCALER_PATH}")

print("[✔] Всё готово! Вы можете использовать KMeans в app.py")
