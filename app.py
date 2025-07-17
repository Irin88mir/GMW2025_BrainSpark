from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models  # Добавлен импорт models
import threading
import time
from queue import Queue

app = Flask(__name__)

# Инициализация каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class PredictionSystem:
    def __init__(self, model_path):
        # Загрузка модели PyTorch
        self.model = self.load_custom_inception(model_path)
        self.model.eval()
        self.camera = None
        self.is_active = False
        self.prediction_queue = Queue()
        self.thread = None
        self.latest_prediction = None
        self.target_fps = 1  # 1 FPS
        
        # Трансформации для изображения
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    
    def load_custom_inception(self, weights_path, num_classes=2):
        # Создаем модель с оригинальной архитектурой
        model = models.inception_v3(pretrained=False, aux_logits=True, init_weights=False)

        # Полностью заменяем структуру головы модели
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # Вспомогательный классификатор
        num_aux_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_aux_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # Загружаем веса
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

        # Настраиваем загрузку для неполного соответствия
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    
    def detect_and_crop_face(self, frame):
        """Обнаружение и вырезание лица из кадра"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            padding = int(w * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            return frame[y:y+h, x:x+w]
        return frame
    
    def preprocess_frame(self, frame):
        """Подготовка кадра для модели PyTorch"""
        face_frame = self.detect_and_crop_face(frame)
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_frame)
        return tensor.unsqueeze(0)
    
    def prediction_loop(self):
        """Основной цикл обработки кадров"""
        self.camera = cv2.VideoCapture(0)
        while self.is_active:
            start_time = time.time()
            
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # Препроцессинг и предсказание
            input_tensor = self.preprocess_frame(frame)
            with torch.no_grad():
                outputs, _ = self.model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = predicted.item()
            
            self.latest_prediction = prediction
            self.prediction_queue.put(prediction)
            
            # Поддержание 1 FPS
            processing_time = time.time() - start_time
            sleep_time = max(1.0/self.target_fps - processing_time, 0)
            time.sleep(sleep_time)
        
        self.camera.release()
    
    def start(self):
        """Запуск системы предсказаний"""
        if not self.is_active:
            self.is_active = True
            self.thread = threading.Thread(target=self.prediction_loop)
            self.thread.start()
            return True
        return False
    
    def stop(self):
        """Остановка системы предсказаний"""
        if self.is_active:
            self.is_active = False
            if self.thread is not None:
                self.thread.join()
            return True
        return False

predictor = PredictionSystem('best_Inception_bi_0_7620503418168787.pth')  # Исправлено имя файла

# API Endpoints
@app.route('/start', methods=['POST'])
def start_prediction():
    if predictor.start():
        return jsonify({"status": "started", "fps": predictor.target_fps})
    return jsonify({"error": "already running"}), 400

@app.route('/stop', methods=['POST'])
def stop_prediction():
    if predictor.stop():
        return jsonify({"status": "stopped"})
    return jsonify({"error": "not running"}), 400

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    if not predictor.is_active:
        return jsonify({"error": "system not active"}), 400
    
    if not predictor.prediction_queue.empty():
        return jsonify({"prediction": predictor.prediction_queue.get()})
    
    if predictor.latest_prediction is not None:
        return jsonify({"prediction": predictor.latest_prediction})
    
    return jsonify({"error": "no predictions yet"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
