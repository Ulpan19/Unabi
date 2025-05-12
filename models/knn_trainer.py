import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "accident_data.csv")
MODEL_PATH = os.path.join("models", "knn_model.pkl")

# Prepare data
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

# Train KNN model
def train_knn_model():
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNeighborsClassifier(n_neighbors=5)  # 5 nearest neighbors by default
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"[✔] Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_knn_model()
