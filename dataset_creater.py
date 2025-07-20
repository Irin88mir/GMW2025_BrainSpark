import requests
import cv2
import os
import time
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List

BASE_URL = "http://127.0.0.1:2336"
TIMEOUT = 3  # сек
TARGET_FPS = 1
DISK = 'D:'
MAIN_FOLDER = os.path.join(DISK, 'data')

class FaceExpressionDetector:
    def __init__(self):
        # Инициализация MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.smile_buffer = []
        self.mouth_open_buffer = []
        self.buffer_size = 3  # Количество кадров для подтверждения

    def detect_expression(self, frame, threshold_smile=0.1, threshold_mouth_open=0.2):
        """Определение выражений лица по кадру"""
        # Предварительная обработка изображения
        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Точки для определения улыбки
            mouth_left = np.array([landmarks[61].x, landmarks[61].y])
            mouth_right = np.array([landmarks[291].x, landmarks[291].y])
            mouth_width = np.linalg.norm(mouth_left - mouth_right)

            # Нормализация относительно расстояния между зрачками
            left_pupil = np.array([landmarks[468].x, landmarks[468].y])
            right_pupil = np.array([landmarks[473].x, landmarks[473].y])
            pupil_distance = np.linalg.norm(left_pupil - right_pupil)

            # Улучшенный расчет улыбки
            smile_ratio = mouth_width / pupil_distance
            is_smiling = smile_ratio > (1.3 + threshold_smile)

            # Точки для определения открытого рта
            mouth_upper = np.mean([landmarks[i].y for i in [13, 312]])
            mouth_lower = np.mean([landmarks[i].y for i in [14, 316]])
            mouth_height = mouth_lower - mouth_upper

            mouth_open_ratio = mouth_height / pupil_distance
            is_mouth_open = mouth_open_ratio > threshold_mouth_open

            # Буферизация состояний для устойчивости
            self.smile_buffer.append(is_smiling)
            self.mouth_open_buffer.append(is_mouth_open)
            if len(self.smile_buffer) > self.buffer_size:
                self.smile_buffer.pop(0)
                self.mouth_open_buffer.pop(0)

            return all(self.smile_buffer) or all(self.mouth_open_buffer)
        return False

def get_signal_quality(device_index: int = 0) -> Optional[List[int]]:
    """Получает качество сигнала для всех каналов устройства"""
    try:
        response = requests.get(
            f"{BASE_URL}/currentDevicesInfo",
            timeout=TIMEOUT
        )
        if response.status_code != 200:
            print(f"Сервер вернул код {response.status_code}")
            return None
        data = response.json()
        if not isinstance(data, dict):
            print("Некорректный формат ответа")
            return None
        devices = data.get("devices", [])
        if not devices:
            print("Нет подключенных устройств")
            return None
        if len(devices) <= device_index:
            print(f"Устройство с индексом {device_index} не найдено")
            return None
        quality = devices[device_index].get("quality")
        if quality is None:
            print("Отсутствует информация о качестве сигнала")
            return None
        return quality
    except requests.exceptions.Timeout:
        print("Таймаут при запросе качества сигнала")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения: {e}")
    except ValueError as e:
        print(f"Ошибка парсинга JSON: {e}")
    return None

def get_concentration_with_quality_check() -> Optional[Dict]:
    """Получает концентрацию при хорошем сигнале"""
    quality = get_signal_quality()
    if quality is None:
        print("Проверьте соединение с нейрогарнитурой")
        return None
    print(f"📶 Качество сигнала: {quality}")
    if not all(q >= 80 for q in quality):
        bad_channels = [i for i, q in enumerate(quality) if q < 80]
        print(f"Плохой сигнал в каналах: {bad_channels}")
        return None
    try:
        response = requests.get(
            f"{BASE_URL}/concentration",
            timeout=TIMEOUT
        )
        data = response.json()
        if data.get("concentration") == -1:
            print("Устройство недоступно")
            return None
        print(f"Концентрация: {data.get('concentration')}%")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

def main():
    print("Начало считывания")

    # Создаем папки для сохранения кадров
    if not os.path.exists(MAIN_FOLDER):
        os.makedirs(MAIN_FOLDER)
    for n in range(2):
        dir_name = f"{n}"
        dir_path = os.path.join(MAIN_FOLDER, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Инициализация детектора выражений лица
    expression_detector = FaceExpressionDetector()

    # Настройка камеры
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video.isOpened():
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Ошибка: не удалось открыть камеру")
            exit()

    # Установка параметров камеры
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 15)

    frame_interval = 1.0 / TARGET_FPS
    last_capture_time = time.time()

    while True:
        success, frame = video.read()
        if not success:
            print("Ошибка чтения кадра")
            break

        current_time = time.time()
        result = get_concentration_with_quality_check()
        expression = expression_detector.detect_expression(frame)
        print(expression)

        if current_time - last_capture_time >= frame_interval and result and result['concentration'] > 0:
            timestamp = int(current_time * 1000)

            print(result['concentration'])
            if (result['concentration'] > 0 and result['concentration'] <= 70) or expression:
                dir_number = 0
            elif result['concentration'] > 70 and not expression:
                dir_number = 1

            # Сохраняем с timestamp
            dir_path = os.path.join(MAIN_FOLDER, str(dir_number))
            cv2.imwrite(os.path.join(dir_path, f'frame_{timestamp}.jpg'), frame)
            last_capture_time = current_time

        cv2.imshow(f'Camera Feed ({TARGET_FPS} FPS limit)', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print('Завершение работы программы')

if __name__ == "__main__":
    main()
