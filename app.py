from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import threading
import time
from queue import Queue

app = Flask(__name__)

# Состояние системы
class PredictionSystem:
    def __init__(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))
        self.camera = None
        self.is_active = False
        self.prediction_queue = Queue()
        self.thread = None
        self.latest_prediction = None
        
    def preprocess_frame(self, frame):
        """Подготовка кадра для модели"""
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        return np.expand_dims(frame, axis=0)
    
    def prediction_loop(self):
        """Основной цикл обработки кадров"""
        self.camera = cv2.VideoCapture(0)
        while self.is_active:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = self.preprocess_frame(rgb_frame)
            prediction = self.model.predict(processed)[0]
            
            self.latest_prediction = int(prediction)
            self.prediction_queue.put(self.latest_prediction)
            
            time.sleep(0.05)  # ~20 FPS
        
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

# Инициализация системы (замените на путь к вашей модели)
predictor = PredictionSystem("model.pkl")

# API Endpoints
@app.route('/start', methods=['POST'])
def start_prediction():
    if predictor.start():
        return jsonify({"status": "started"})
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
    
    # Если очередь пуста, вернуть последнее предсказание
    if predictor.latest_prediction is not None:
        return jsonify({"prediction": predictor.latest_prediction})
    
    return jsonify({"error": "no predictions yet"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)