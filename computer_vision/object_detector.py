import cv2
from ultralytics import YOLO

# Загрузка модели YOLO
model = YOLO("yolov8n.pt")  # Путь к предобученной модели

def detect_objects(image_path):
    # Чтение изображения
    image = cv2.imread(image_path)
    
    # Детекция объектов
    results = model(image)
    
    # Отображение результатов на изображении
    annotated_image = results[0].plot()  # Выводим изображение с рамками

    # Сохраняем результат
    cv2.imwrite("static/detected_objects.jpg", annotated_image)

    return results
