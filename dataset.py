import requests
import cv2
import os
import time
from typing import Optional, Dict, List

#TO DO
#Имена файлов

BASE_URL = "http://127.0.0.1:2336"
TIMEOUT = 3  # сек
TARGET_FPS = 3

def get_signal_quality(device_index: int = 0) -> Optional[List[int]]:
    """Получает качество сигнала для всех каналов устройства"""
    try:
        response = requests.get(
            f"{BASE_URL}/currentDevicesInfo",
            timeout=TIMEOUT
        )
        # Проверка HTTP-статуса
        if response.status_code != 200:
            print(f"Сервер вернул код {response.status_code}")
            return None
        data = response.json()
        # Проверка структуры ответа
        if not isinstance(data, dict):
            print("Некорректный формат ответа")
            return None
        # Проверка наличия устройств
        devices = data.get("devices", [])
        if not devices:
            print("Нет подключенных устройств")
            return None
        if len(devices) <= device_index:
            print(f"Устройство с индексом {device_index} не найдено")
            return None
        # Получаем качество сигнала
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
    quality[1] = 100
    quality[4] = 100
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

# Создаем папки для сохранения кадров
disk = 'D:'
main_folder = os.path.join(disk, 'data')
if not os.path.exists(main_folder):
    os.makedirs(main_folder)
for n in range(10):
    dir_name = f"{n+1}0%"
    dir_path = os.path.join(main_folder, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

video = cv2.VideoCapture(0)

frame_interval = 1.0 / TARGET_FPS

frame_count = 0
last_capture_time = time.time()

while True:
    success, frame = video.read()
    if not success:
        break

    current_time = time.time()
    result = get_concentration_with_quality_check()
    if current_time - last_capture_time >= frame_interval and result:
        print(result['concentration'])
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        dir_number = result['concentration'] % 10 + 1
        cv2.imwrite(f'{main_folder}/{dir_number}0%/f"frame_{timestamp}.jpg"', frame)
        frame_count += 1
        last_capture_time = current_time

        if frame_count % TARGET_FPS == 0:
            elapsed = current_time - (last_capture_time - frame_interval * TARGET_FPS)
            actual_fps = TARGET_FPS / elapsed
            print(f"Current FPS: {actual_fps:.2f}")

    cv2.imshow('Camera Feed ({TARGET_FPS} FPS limit)', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print(f'Сохранено {frame_count} кадров (~{TARGET_FPS} FPS)')
