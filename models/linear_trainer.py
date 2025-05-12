import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Пути
DATA_PATH = os.path.join("data", "accident_data.csv")
MODEL_PATH = os.path.join("models", "linear_model.pkl")

# Подготовка данных
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    le_car = LabelEncoder()
    le_weather = LabelEncoder()
    le_time = LabelEncoder()

    df["car_type"] = le_car.fit_transform(df["car_type"])
    df["weather"] = le_weather.fit_transform(df["weather"])
    df["time_of_day"] = le_time.fit_transform(df["time_of_day"])

    X = df.drop("accident", axis=1)
    y = df["accident"]

    return X, y

# Обучение модели
def train_linear_model():
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Оценка качества модели
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    # Сохранение модели
    joblib.dump(model, MODEL_PATH)
    print(f"[✔] Модель сохранена в {MODEL_PATH}")

if __name__ == "__main__":
    train_linear_model()
