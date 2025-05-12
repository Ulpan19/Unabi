import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Пути
DATA_PATH = os.path.join("data", "accident_data.csv")  # Путь к вашим данным
MODEL_PATH = os.path.join("models", "lda_model.pkl")  # Путь для сохранения модели

# Подготовка данных
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Инициализация LabelEncoder для категориальных признаков
    le_car = LabelEncoder()
    le_weather = LabelEncoder()
    le_time = LabelEncoder()

    # Преобразуем категориальные признаки в числовые
    df["car_type"] = le_car.fit_transform(df["car_type"])
    df["weather"] = le_weather.fit_transform(df["weather"])
    df["time_of_day"] = le_time.fit_transform(df["time_of_day"])

    X = df.drop("accident", axis=1)
    y = df["accident"]

    return X, y

# Обучение модели LDA
def train_lda_model():
    X, y = load_and_prepare_data()

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание модели LDA
    lda = LinearDiscriminantAnalysis()

    # Обучение модели
    lda.fit(X_train, y_train)

    # Прогнозирование
    y_pred = lda.predict(X_test)

    # Оценка модели
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Сохранение модели
    joblib.dump(lda, MODEL_PATH)
    print(f"[✔] Модель LDA сохранена в {MODEL_PATH}")

if __name__ == "__main__":
    train_lda_model()
