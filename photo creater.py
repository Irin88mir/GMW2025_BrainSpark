import cv2
import os
import time

# Создаем папку для сохранения кадров
if not os.path.exists('data'):
    os.makedirs('data')

video = cv2.VideoCapture(0)

TARGET_FPS = 30
frame_interval = 1.0 / TARGET_FPS

frame_count = 0
last_capture_time = time.time()

while True:
    success, frame = video.read()
    if not success:
        break

    current_time = time.time()

    if current_time - last_capture_time >= frame_interval:
        cv2.imwrite(f'data/frame_{frame_count:04d}.jpg', frame)
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