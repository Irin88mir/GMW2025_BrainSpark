import os
import cv2
import sys

'''Обрезка лиц на изображениях для датасета'''

def load_cascade():
    """Загрузка каскада Хаара с проверкой нескольких возможных путей"""
    # Пробуем разные варианты путей
    possible_paths = [
        os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'),
        os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
        'haarcascade_frontalface_default.xml'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
            print(f"Файл найден, но не загружается: {path}")
        else:
            print(f"Файл не найден: {path}")

    raise FileNotFoundError("Не удалось найти рабочий каскадный файл")


def process_images():
    """Основная функция обработки изображений"""
    try:
        # 1. Загрузка каскада
        face_cascade = load_cascade()
        print("Каскад успешно загружен")

        # 2. Настройка путей
        input_root = 'data'
        output_root = 'data_faces'

        # 3. Создание структуры папок
        os.makedirs(output_root, exist_ok=True)

        for percent in range(10, 101, 10):
            folder_name = f"{percent}%"
            input_folder = os.path.join(input_root, folder_name)
            output_folder = os.path.join(output_root, folder_name)

            if not os.path.exists(input_folder):
                print(f"Предупреждение: Папка {input_folder} не найдена, пропускаем")
                continue

            os.makedirs(output_folder, exist_ok=True)

            # 4. Обработка изображений
            for img_name in os.listdir(input_folder):
                img_path = os.path.join(input_folder, img_name)

                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                try:
                    # Чтение изображения
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Не удалось прочитать: {img_path}")
                        continue

                    # Обнаружение лиц
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    if len(faces) == 0:
                        print(f"Лица не найдены: {img_path}")
                        continue

                    # Обрезка первого лица
                    x, y, w, h = faces[0]
                    padding = int(min(w, h) * 0.2)
                    x, y = max(0, x - padding), max(0, y - padding)
                    w = min(img.shape[1] - x, w + 2 * padding)
                    h = min(img.shape[0] - y, h + 2 * padding)

                    face = img[y:y + h, x:x + w]
                    output_path = os.path.join(output_folder, img_name)

                    if not cv2.imwrite(output_path, face):
                        print(f"Не удалось сохранить: {output_path}")
                        continue

                    print(f"Успешно: {img_path} -> {output_path}")

                except Exception as e:
                    print(f"Ошибка обработки {img_path}: {str(e)}")

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Запуск обработки изображений...")
    process_images()
    print("Обработка завершена!")
