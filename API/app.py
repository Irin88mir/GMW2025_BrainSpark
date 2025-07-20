from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import threading
import time
import uvicorn

'''Для работы необходим файл "best_Inception_bi_0_7620503418168787.pth"'''

app = FastAPI()

# Инициализация каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Глобальные переменные
prediction_lock = threading.Lock()
latest_prediction = None

class PredictionSystem:
    def __init__(self, model_path):
        # Загрузка модели PyTorch
        self.model = self.load_custom_inception(model_path)
        self.model.eval()
        self.camera = None
        self.is_active = False
        self.thread = None
        self.target_fps = 1  # 1 FPS
        self.prediction_interval = 1.0  # Интервал предсказаний в секундах
        
        # Трансформации для изображения
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def load_custom_inception(self, weights_path, num_classes=2):
        """Загрузка кастомной модели InceptionV3"""
        model = models.inception_v3(pretrained=False, aux_logits=True, init_weights=False)

        # Основной классификатор
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

        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        return model
    
    def detect_and_crop_face(self, frame):
        try:
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
        except Exception as e:
            print(f"Ошибка при обнаружении лица: {e}")
            return frame
    
    def preprocess_frame(self, frame):
        """Подготовка кадра для модели"""
        face_frame = self.detect_and_crop_face(frame)
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(face_frame)
        return tensor.unsqueeze(0)
    
    def prediction_loop(self):
        """Основной цикл обработки кадров"""
        global latest_prediction
        self.camera = cv2.VideoCapture(0)
        last_prediction_time = time.time()
        
        while self.is_active:
            start_time = time.time()
            
            # Получаем кадр
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # Делаем предсказание с заданным интервалом
            current_time = time.time()
            if current_time - last_prediction_time >= self.prediction_interval:
                last_prediction_time = current_time
                
                # Препроцессинг и предсказание
                input_tensor = self.preprocess_frame(frame)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    prediction = {
                        'time': time.strftime("%H:%M:%S"),
                        'value': predicted.item(),
                        'confidence': round(confidence.item(), 4),
                        'class': 'focus' if predicted.item() == 1 else 'distracted'
                    }
                    
                    with prediction_lock:
                        latest_prediction = prediction
            
            # Поддержание FPS
            processing_time = time.time() - start_time
            sleep_time = max(0.05 - processing_time, 0)
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

# Инициализация системы
predictor = PredictionSystem('best_Inception_bi_0_7620503418168787.pth')

@app.post("/start")
async def start_prediction():
    """Запуск системы отслеживания внимания"""
    if predictor.start():
        return JSONResponse(content={"status": "started"})
    raise HTTPException(status_code=400, detail="System already running")

@app.post("/stop")
async def stop_prediction():
    """Остановка системы отслеживания внимания"""
    if predictor.stop():
        return JSONResponse(content={"status": "stopped"})
    raise HTTPException(status_code=400, detail="System not running")

@app.get("/get_prediction")
async def get_prediction():
    """Получение последнего предсказания"""
    with prediction_lock:
        if latest_prediction is None:
            raise HTTPException(status_code=404, detail="No predictions available")
        return JSONResponse(content=latest_prediction)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
