import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "accident_data.csv")
MODEL_PATH = os.path.join("models", "svm_model.pkl")

# Prepare data
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Encode categorical features
    le_car = LabelEncoder()
    le_weather = LabelEncoder()
    le_time = LabelEncoder()

    df["car_type"] = le_car.fit_transform(df["car_type"])
    df["weather"] = le_weather.fit_transform(df["weather"])
    df["time_of_day"] = le_time.fit_transform(df["time_of_day"])

    X = df.drop("accident", axis=1)
    y = df["accident"]

    # Save encoders (optional but recommended)
    encoders = {
        "car_type": le_car,
        "weather": le_weather,
        "time_of_day": le_time
    }
    joblib.dump(encoders, "models/svm_encoders.pkl")
    print("[✔] Encoders saved at models/svm_encoders.pkl")

    return X, y

# Train SVM model
def train_svm_model():
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SVM with probability support
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[✔] Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_svm_model()
