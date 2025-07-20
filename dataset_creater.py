import requests
import cv2
import os
import time
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List

'''Создание датасета с учётом задержки нейрогарнитуры и выражения лица'''

BASE_URL = "http://127.0.0.1:2336"
TIMEOUT = 3  # сек
TARGET_FPS = 1
DELAY = 1.7  # Задержка в секундах для синхронизации с ЭЭГ
DISK = 'D:'
MAIN_FOLDER = os.path.join(DISK, 'data_delay_bi')
if not os.path.exists(MAIN_FOLDER):
    os.makedirs(MAIN_FOLDER)
for n in range(2):
    dir_name = f"{n}"
    dir_path = os.path.join(MAIN_FOLDER, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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

            # Расчет улыбки
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
    """Получает концентрацию только при хорошем сигнале"""
    quality = get_signal_quality()
    if quality is None:
        print("Ошибка 1")
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


video = cv2.VideoCapture(0)

frame_interval = 1.0 / TARGET_FPS
last_capture_time = time.time()
frame_buffer = []  # Буфер для хранения кадров с временными метками
if not os.path.exists('data_delay_bi'):
    os.makedirs('data_delay_bi')
for n in range(2):
    dir_name = f'{n}'
    dir_path = os.path.join('data_delay_bi', dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Инициализация детектора выражений лица
expression_detector = FaceExpressionDetector()

while True:
    success, frame = video.read()
    if not success:
        break

    current_time = time.time()

    # Определяем выражение лица
    expression = expression_detector.detect_expression(frame)
    print(f"Выражение лица: {'Улыбка/открытый рот' if expression else 'Нейтральное'}")

    # Сохраняем кадр в буфер с временной меткой и информацией о выражении лица
    frame_buffer.append((frame, current_time, expression))

    # Удаляем старые кадры из буфера (старше DELAY + frame_interval)
    while frame_buffer and current_time - frame_buffer[0][1] > DELAY + frame_interval:
        frame_buffer.pop(0)

    result = get_concentration_with_quality_check()

    if result and current_time - last_capture_time >= frame_interval and result['concentration'] > 0:
        # Ищем кадр, который был захвачен примерно DELAY секунд назад
        target_time = current_time - DELAY
        closest_frame = None
        closest_expression = False
        min_diff = float('inf')

        timestamp = int(current_time * 1000)

        for f, t, expr in frame_buffer:
            diff = abs(t - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_frame = f
                closest_expression = expr

        if closest_frame is not None:
            # Получаем 13-значную временную метку
            timestamp = int(target_time * 1000)
            last_capture_time = current_time

            # Проверка FPS (для отладки, можно убрать)
            if int(timestamp / 1000) % 1 == 0:  # Проверяем каждую секунду
                elapsed = current_time - (last_capture_time - frame_interval * TARGET_FPS)
                actual_fps = TARGET_FPS / elapsed

            # Определяем директорию для сохранения
            if (result["concentration"] >= 1 and result["concentration"] <= 50) or closest_expression:
                cv2.imwrite(
                    f'data_delay_bi/0/{timestamp}_{result["concentration"]}_{int(closest_expression)}.jpg',
                    closest_frame)
            elif result["concentration"] >= 71 and not closest_expression:
                cv2.imwrite(
                    f'data_delay_bi/1/{timestamp}_{result["concentration"]}_{int(closest_expression)}.jpg',
                    closest_frame)

            try:
                cv2.putText(closest_frame, f'Concentration ({result["concentration"]}%)', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(closest_frame, f'Expression: {"Smile/Open" if closest_expression else "Neutral"}',
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except:
                cv2.putText(closest_frame, 'err', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Concentration', closest_frame)

    # Отображаем текущий кадр (без задержки)
    cv2.imshow('Live Feed', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print('Завершение работы программы')
