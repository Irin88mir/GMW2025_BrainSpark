import cv2
import mediapipe as mp
import math
import time

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# Насколько меньше среднего должен быть открыт глаз, чтобы зачлось моргание
x = 0.05

# Индексы ключевых точек для глаз (левого и правого)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Левый глаз (от зрителя справа)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Правый глаз (от зрителя слева)

def calculate_ear(landmarks, eye_indices, image_w, image_h):
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        points.append((x, y))
    d1 = math.dist(points[1], points[5])  # |p2-p6|
    d2 = math.dist(points[2], points[4])  # |p3-p5|
    d3 = math.dist(points[0], points[3])  # |p1-p4|
    ear = (d1 + d2) / (2.0 * d3) if d3 != 0 else 0.0
    return ear, points


avg_open = []
eye_open = True
blinks = 0
# Для расчета BPM
blink_times = []  # Список временных меток морганий
bpm = 0  # Частота морганий в минуту
last_time = time.time()  # Время последнего моргания

cap = cv2.VideoCapture(0)
with mp_face.FaceMesh(min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Рисуем общие landmarks лица
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face.FACEMESH_CONTOURS)
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                # Рассчитываем EAR для каждого глаза
                left_ear, left_points = calculate_ear(face_landmarks, LEFT_EYE_INDICES, w, h)
                right_ear, right_points = calculate_ear(face_landmarks, RIGHT_EYE_INDICES, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                # Рисуем точки глаз и линии EAR
                for point in left_points + right_points:
                    cv2.circle(image, point, 2, (255, 0, 0), -1)
                for eye_points in [left_points, right_points]:
                    cv2.line(image, eye_points[1], eye_points[5], (0, 255, 255), 1)
                cv2.line(image, eye_points[2], eye_points[4], (0, 255, 255), 1)
                cv2.line(image, eye_points[0], eye_points[3], (0, 255, 0), 1)

                cv2.putText(image, f"Avg EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Blinks: {blinks:}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"BPM: {bpm:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                avg_open.append(avg_ear)
                if len(avg_open) >= 30:
                    avg_open.pop(0)

                if avg_ear < sum(avg_open) / len(avg_open) - x and eye_open:
                    blinks += 1
                    blink_times.append(current_time)
                current_time = time.time()

                # Удаляем старые моргания (старше 30 секунд)
            while blink_times and current_time - blink_times[0] > 30:
                blink_times.pop(0)
            if len(blink_times) >= 2:
                time_interval = (blink_times[-1] - blink_times[0]) / (len(blink_times) - 1)
                bpm = 60 / time_interval if time_interval != 0 else 0
            else:
                bpm = 0
            eye_open = False
        if avg_ear >= sum(avg_open) / len(avg_open) - x:
            eye_open = True
        cv2.imshow("Face Mesh with EAR", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
