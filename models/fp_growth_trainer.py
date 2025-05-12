import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import fpgrowth
import os

# Пути
DATA_PATH = os.path.join("data", "accident_data.csv")  # Путь к данным
MODEL_PATH = os.path.join("models", "fp_growth_model.csv")  # Путь куда сохранить

# Загрузка и подготовка данных
def load_and_prepare_data():
    # Загружаем данные
    df = pd.read_csv(DATA_PATH)

    # Категориальные признаки кодируем в числа
    le_car = LabelEncoder()
    le_weather = LabelEncoder()
    le_time = LabelEncoder()

    df["car_type"] = le_car.fit_transform(df["car_type"])
    df["weather"] = le_weather.fit_transform(df["weather"])
    df["time_of_day"] = le_time.fit_transform(df["time_of_day"])

    # Бинаризуем данные (если больше 0 → 1, иначе 0)
    df_encoded = df.applymap(lambda x: 1 if x > 0 else 0)

    # Приведем колонки к удобному виду
    df_encoded.columns = [
        f"age_{i}" if col == "age" else 
        f"experience_{i}" if col == "experience" else
        f"{col}" for col, i in zip(df_encoded.columns, range(len(df_encoded.columns)))
    ]

    return df_encoded

# Обучение FP-Growth и сохранение
def run_fp_growth():
    df_encoded = load_and_prepare_data()

    # Поиск частых наборов (min_support можно менять)
    frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

    # Преобразуем itemsets в строку, чтобы сохранить в csv
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(lambda x: ", ".join(str(item) for item in x))

    # Сохраняем модель в CSV
    os.makedirs("models", exist_ok=True)
    frequent_itemsets.to_csv(MODEL_PATH, index=False)

    print("[✔] FP-Growth модель сохранена!")
    print(frequent_itemsets)

if __name__ == "__main__":
    run_fp_growth()
