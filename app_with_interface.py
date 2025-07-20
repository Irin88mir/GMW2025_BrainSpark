from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import threading
import time
from queue import Queue
import json
import uvicorn

'''API с интерфейсом для отладки'''

app = FastAPI()

# Инициализация каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Глобальные переменные
camera_frame = None
predictions = []
frame_lock = threading.Lock()
prediction_lock = threading.Lock()
streaming_active = False  # Добавлен флаг для управления потоковым видео

class PredictionSystem:
    def __init__(self, model_path):
        # Загрузка модели PyTorch
        self.model = self.load_custom_inception(model_path)
        self.model.eval()
        self.camera = None
        self.is_active = False
        self.thread = None
        self.latest_prediction = None
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
        global camera_frame, predictions, streaming_active
        self.camera = cv2.VideoCapture(0)
        last_prediction_time = time.time()
        
        while self.is_active:
            start_time = time.time()
            
            # Получаем кадр
            ret, frame = self.camera.read()
            if ret:
                with frame_lock:
                    camera_frame = frame.copy()
                
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
                            predictions.append(prediction)
                            # Ограничиваем количество хранимых предсказаний
                            if len(predictions) > 10:
                                predictions.pop(0)
                    
                    self.latest_prediction = prediction
            
            # Поддержание FPS
            processing_time = time.time() - start_time
            sleep_time = max(0.05 - processing_time, 0)
            time.sleep(sleep_time)
        
        self.camera.release()
        streaming_active = False  # Отключаем потоковое видео при остановке
    
    def start(self):
        """Запуск системы предсказаний"""
        global streaming_active
        if not self.is_active:
            self.is_active = True
            streaming_active = True  # Включаем потоковое видео
            self.thread = threading.Thread(target=self.prediction_loop)
            self.thread.start()
            return True
        return False
    
    def stop(self):
        """Остановка системы предсказаний"""
        global streaming_active
        if self.is_active:
            self.is_active = False
            if self.thread is not None:
                self.thread.join()
            streaming_active = False  # Отключаем потоковое видео
            return True
        return False

# Инициализация системы
predictor = PredictionSystem('best_Inception_bi_0_7620503418168787.pth')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Проверка концентрации внимания</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .video-container { flex: 1; }
        .predictions-container { flex: 1; }
        #predictions { 
            height: 480px; 
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .prediction-item {
            margin-bottom: 8px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 4px;
        }
        .focus { background-color: #d4edda !important; }
        .distracted { background-color: #f8d7da !important; }
        button { 
            padding: 8px 16px; 
            margin-right: 10px; 
            font-size: 16px;
            cursor: pointer;
        }
        #cameraFeed {
            border: 2px solid #333;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <h1>Eye Tracking Monitor</h1>
    <div class="container">
        <div class="video-container">
            <h3>Live Camera Feed</h3>
            <img id="cameraFeed" src="/video_feed" width="640" height="480">
            <div>
                <button onclick="startSystem()">Start System</button>
                <button onclick="stopSystem()">Stop System</button>
            </div>
        </div>
        <div class="predictions-container">
            <h3>Real-time Predictions</h3>
            <div id="predictions">
                <p>System not active. Press "Start System" to begin.</p>
            </div>
        </div>
    </div>

    <script>
        // Обновление изображения
        let videoInterval = setInterval(updateVideoFeed, 100);
        
        function updateVideoFeed() {
            if (document.getElementById('cameraFeed').src.includes("/video_feed")) {
                document.getElementById('cameraFeed').src = "/video_feed?" + Date.now();
            }
        }

        // Обновление предсказаний
        function updatePredictions() {
            fetch('/get_predictions')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('predictions');
                    if (data.length === 0) {
                        container.innerHTML = '<p>No predictions yet</p>';
                        return;
                    }
                    
                    container.innerHTML = '';
                    data.reverse().forEach(pred => {
                        const item = document.createElement('div');
                        item.className = `prediction-item ${pred.class}`;
                        item.innerHTML = `
                            <strong>${pred.time}</strong><br>
                            Status: <b>${pred.class.toUpperCase()}</b><br>
                            Confidence: ${(pred.confidence * 100).toFixed(1)}%
                        `;
                        container.appendChild(item);
                    });
                });
        }

        // Автоматическое обновление предсказаний
        let predictionInterval;
        
        function startSystem() {
            fetch('/start', {method: 'POST'})
                .then(() => {
                    document.getElementById('predictions').innerHTML = '<p>Starting...</p>';
                    // Обновляем каждую секунду
                    predictionInterval = setInterval(updatePredictions, 1000);
                    // Включаем обновление видео
                    videoInterval = setInterval(updateVideoFeed, 100);
                });
        }
        
        function stopSystem() {
            fetch('/stop', {method: 'POST'})
                .then(() => {
                    clearInterval(predictionInterval);
                    clearInterval(videoInterval);
                    document.getElementById('predictions').innerHTML = 
                        '<p>System stopped. Press "Start System" to begin.</p>';
                    // Останавливаем видео, устанавливая placeholder
                    document.getElementById('cameraFeed').src = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
                });
        }

        // Начальное обновление
        updatePredictions();
    </script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/video_feed")
async def video_feed():
    global camera_frame, streaming_active
    if not streaming_active:
        return Response(content=b'', media_type='image/jpeg')
    
    with frame_lock:
        if camera_frame is not None:
            ret, buffer = cv2.imencode('.jpg', camera_frame)
            frame = buffer.tobytes()
            return Response(content=frame, media_type='image/jpeg')
    return Response(content=b'', media_type='image/jpeg')

@app.post("/start")
async def start_prediction():
    global predictions
    predictions = []  # Очищаем предыдущие предсказания
    if predictor.start():
        return JSONResponse(content={"status": "started"})
    raise HTTPException(status_code=400, detail="System already running")

@app.post("/stop")
async def stop_prediction():
    if predictor.stop():
        return JSONResponse(content={"status": "stopped"})
    raise HTTPException(status_code=400, detail="System not running")

@app.get("/get_predictions")
async def get_predictions():
    with prediction_lock:
        return JSONResponse(content=predictions)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
