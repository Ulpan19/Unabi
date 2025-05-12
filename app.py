from flask import Flask, render_template, request, session
import joblib
import numpy as np
import os
from sklearn.cluster import KMeans
import cv2
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

app = Flask(__name__) 
app.secret_key = "your_secret_key"

car_map = {"sedan": 0, "suv": 1, "truck": 2, "sports": 3, "minivan": 4}
weather_map = {"sunny": 3, "rainy": 2, "foggy": 0, "snowy": 1}
time_map = {"morning": 2, "afternoon": 0, "evening": 1, "night": 3}

MODEL_PATH = os.path.join("models", "logistic_model.pkl")
logistic_model = joblib.load(MODEL_PATH)
yolo_model = YOLO("yolov8n.pt")

model_files = {
    "Logistic Regression": "models/logistic_model.pkl",
    "Decision Tree": "models/decision_tree_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "Gradient Boosting": "models/gradient_boosting_model.pkl",
    "SVM": "models/svm_model.pkl",
    "Naive Bayes": "models/naive_bayes_model.pkl",
    "Linear": "models/linear_model.pkl",
    "LDA": "models/lda_model.pkl",
    "Kmeans": "models/kmeans_model.pkl"
}

@app.route('/')
def models_form():
    return render_template("models.html", models=list(model_files.keys()))

@app.route('/form', methods=['POST'])
def form_with_models():
    session['selected_models'] = request.form.getlist('models')
    return render_template("form.html")

@app.route('/predict_models', methods=['POST'])
def predict_models():
    try:
        selected_models = session.get('selected_models', [])
        age = int(request.form['age'])
        experience = int(request.form['experience'])
        car_type = car_map[request.form['car_type']]
        weather = weather_map[request.form['weather']]
        time_of_day = time_map[request.form['time_of_day']]
        speeding = int(request.form['speeding'])
        seatbelt = int(request.form['seatbelt'])
        alcohol = int(request.form['alcohol'])

        features = np.array([[age, experience, car_type, weather, time_of_day, speeding, seatbelt, alcohol]])
        results = {}

        if not selected_models:
            selected_models = ["Logistic Regression"]

        color_map = {
            "Logistic Regression": "blue",
            "Decision Tree": "green",
            "Random Forest": "orange",
            "Gradient Boosting": "purple",
            "SVM": "red",
            "Naive Bayes": "cyan",
            "Linear": "gray",
            "LDA": "teal",
            "Kmeans": "black"
        }

        for model_name in selected_models:
            model_path = model_files.get(model_name)

            if model_name == "Kmeans":
                scaler = joblib.load("models/kmeans_scaler.pkl")
                kmeans_model = joblib.load("models/kmeans_model.pkl")
                features_scaled = scaler.transform(features)
                cluster = kmeans_model.predict(features_scaled)[0]
                cluster_names = ['Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
                cluster_colors = ['green', 'orange', 'red', 'purple']
                cluster_name = cluster_names[cluster]
                cluster_color = cluster_colors[cluster]
                results[model_name] = f"Кластер: {cluster_name}"
                plt.figure()
                plt.bar(["Кластер"], [cluster], color=cluster_color)
                plt.title(f"{model_name} Clustering")
                plt.ylim(0, 3)
                filename = f"static/{model_name.replace(' ', '')}{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                plt.savefig(filename)
                plt.close()
                results[f"{model_name}_chart"] = filename
            elif model_name == "Linear" and model_path and os.path.exists(model_path):
                model = joblib.load(model_path)
                prediction = model.predict(features)[0]
                results[model_name] = f"Оценка риска: {round(prediction, 2)}"
                plt.figure()
                plt.bar(["Оценка риска"], [prediction], color='orange')
                plt.title(f"{model_name} Prediction")
                filename = f"static/{model_name.replace(' ', '')}{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                plt.savefig(filename)
                plt.close()
                results[f"{model_name}_chart"] = filename
            elif model_path and os.path.exists(model_path):
                model = joblib.load(model_path)
                if hasattr(model, "predict_proba"):
                    prediction = model.predict_proba(features)[0][1] * 100
                else:
                    prediction = model.predict(features)[0]
                results[model_name] = f"{round(prediction, 2)} %"
                model_color = color_map.get(model_name, "blue")
                plt.figure(figsize=(5, 6))
                plt.bar(["Риск"], [prediction], color=model_color)
                plt.ylim(0, 100)
                plt.title(f"{model_name} Prediction")
                plt.text(0, prediction + 1, f"{round(prediction, 2)}%", ha='center', fontsize=12, color='black')
                filename = f"static/{model_name.replace(' ', '')}{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                plt.savefig(filename)
                plt.close()
                results[f"{model_name}_chart"] = filename
                if model_name in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]:
                    try:
                        if hasattr(model, "coef_"):
                            importances = model.coef_[0]
                        elif hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                        else:
                            importances = None
                        if importances is not None:
                            feature_names = ["age", "experience", "car_type", "weather", "time_of_day", "speeding", "seatbelt", "alcohol"]
                            plt.figure(figsize=(10, 6))
                            plt.barh(feature_names, importances, color=model_color)
                            plt.xlabel("Влияние")
                            plt.title(f"Feature Importance ({model_name})")
                            plt.grid(True)
                            plt.tight_layout()
                            filename_imp = f"static/{model_name.replace(' ', '')}_importance{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                            plt.savefig(filename_imp)
                            plt.close()
                            results[f"{model_name}_importance_chart"] = filename_imp
                    except Exception as e:
                        print(f"[!] Ошибка importance для {model_name}: {e}")
        return render_template("result_models.html", results=results)
    except Exception as e:
        return f"Ошибка: {e}"

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        experience = int(request.form['experience'])
        car_type = car_map[request.form['car_type']]
        weather = weather_map[request.form['weather']]
        time_of_day = time_map[request.form['time_of_day']]
        speeding = int(request.form['speeding'])
        seatbelt = int(request.form['seatbelt'])
        alcohol = int(request.form['alcohol'])
        features = np.array([[age, experience, car_type, weather, time_of_day, speeding, seatbelt, alcohol]])
        scaler = joblib.load("models/kmeans_scaler.pkl")
        kmeans_model = joblib.load("models/kmeans_model.pkl")
        features_scaled = scaler.transform(features)
        cluster = kmeans_model.predict(features_scaled)[0]
        cluster_names = ['Low Risk', 'Moderate Risk', 'High Risk', 'Extreme Risk']
        cluster_name = cluster_names[cluster]
        return render_template("kmeans_result.html", cluster=cluster_name)
    return render_template("kmeans_form.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result_image = False
    if request.method == 'POST':
        file = request.files.get("image")
        if file and file.filename != "":
            os.makedirs("static", exist_ok=True)
            path = "static/uploaded_image.jpg"
            file.save(path)
            results = yolo_model(path, conf=0.4)
            res_plotted = results[0].plot()
            cv2.imwrite("static/detected_objects.jpg", res_plotted)
            result_image = True
    return render_template("upload.html", result_image=result_image)

@app.route('/fp_growth_patterns')
def fp_growth_patterns():
    fp_data = pd.read_csv("models/fp_growth_model.csv")
    patterns = []
    for index, row in fp_data.iterrows():
        patterns.append({
            "items": row['itemsets'],
            "support": round(row['support'], 3)
        })
    return render_template("fp_growth_result.html", patterns=patterns)

if __name__ == '__main__': 
    app.run(debug=True)