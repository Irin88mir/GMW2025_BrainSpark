import time
import cv2
import mediapipe as mp
import math

BLINK_THRESHOLD = 0.05  # Порог для определения моргания
blink_window = 10  # Окно в секундах для расчета BPM
time_side_gaze = 6 # Время отведения взгляда от объекта внимания 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

GAZE_LEFT_EYE_INDICES = [33, 133, 144, 145, 160, 159]
GAZE_RIGHT_EYE_INDICES = [362, 385, 386, 263, 374, 373]
BLINK_LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
BLINK_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

side_gaze = False
gaze_start_time = 0
gaze_direction = "Center"
color = (84, 182, 134)

avg_open = []
eye_open = True
blinks = 0
blink_times = []
bpm = 0

def get_eye_center(eye_indices, landmarks, frame_shape):
    points = []
    for idx in eye_indices:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * frame_shape[1])
        y = int(landmark.y * frame_shape[0])
        points.append((x, y))

    center_x = sum([p[0] for p in points]) // len(points)
    center_y = sum([p[1] for p in points]) // len(points)
    return center_x, center_y

def get_gaze_direction(eye_center, pupil, threshold=5):
    dx = pupil[0] - eye_center[0]
    if abs(dx) > threshold:
        return "Left" if dx > 0 else "Right"
    return "Center"

def calculate_ear(landmarks, eye_indices, image_w, image_h):
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        points.append((x, y))
    d1 = math.dist(points[1], points[5])
    d2 = math.dist(points[2], points[4])
    d3 = math.dist(points[0], points[3])
    ear = (d1 + d2) / (2.0 * d3) if d3 != 0 else 0.0
    return ear, points

def update_blink_stats():
    global bpm, blink_times
    current_time = time.time()
    # Удаляем моргания старше blink_window секунд
    blink_times = [t for t in blink_times if current_time - t <= blink_window]
    if len(blink_times) >= 2:
        time_interval = blink_window / len(blink_times)
        bpm = 60 * len(blink_times) / blink_window
    else:
        bpm = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Отрисовка landmarks лица
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp.solutions.drawing_utils.DrawingSpec(color=(84, 182, 134), thickness=1),
                mp.solutions.drawing_utils.DrawingSpec(color=(84, 182, 134), thickness=1)
            )

            # 1. Обработка направления взгляда
            left_eye_center = get_eye_center(GAZE_LEFT_EYE_INDICES, face_landmarks, frame.shape)
            right_eye_center = get_eye_center(GAZE_RIGHT_EYE_INDICES, face_landmarks, frame.shape)

            left_pupil = (
                int(face_landmarks.landmark[468].x * frame.shape[1]),
                int(face_landmarks.landmark[468].y * frame.shape[0])
            )
            right_pupil = (
                int(face_landmarks.landmark[473].x * frame.shape[1]),
                int(face_landmarks.landmark[473].y * frame.shape[0])
            )

            left_gaze = get_gaze_direction(left_eye_center, left_pupil)
            right_gaze = get_gaze_direction(right_eye_center, right_pupil)

            # Определяем общее направление взгляда
            if left_gaze == right_gaze:
                current_gaze = left_gaze
            else:
                current_gaze = "Center"

            # Обновляем статус side_gaze только если взгляд направлен в сторону
            if current_gaze != "Center":
                if not side_gaze:
                    side_gaze = True
                    gaze_start_time = time.time()
                gaze_direction = current_gaze
            else:
                if side_gaze and (time.time() - gaze_start_time > time_side_gaze):
                    side_gaze = False
                    gaze_direction = "Center"

            # 2. Обработка морганий
            left_ear, left_points = calculate_ear(face_landmarks, BLINK_LEFT_EYE_INDICES, frame.shape[1], frame.shape[0])
            right_ear, right_points = calculate_ear(face_landmarks, BLINK_RIGHT_EYE_INDICES, frame.shape[1], frame.shape[0])
            avg_ear = (left_ear + right_ear) / 2.0

            avg_open.append(avg_ear)
            if len(avg_open) > 30:
                avg_open.pop(0)

            current_time = time.time()
            if len(avg_open) > 0 and avg_ear < (sum(avg_open) / len(avg_open)) - BLINK_THRESHOLD and eye_open:
                blinks += 1
                blink_times.append(current_time)
                update_blink_stats()
                eye_open = False
            update_blink_stats()
            if len(avg_open) > 0 and avg_ear >= (sum(avg_open) / len(avg_open)) - BLINK_THRESHOLD:
                eye_open = True

            # Отрисовка информации
            cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Blinks: {blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"BPM: {bpm:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            focus_status = "Focused" if bpm <= 18 and gaze_direction == "Center" else "Unfocused"
            color = (84, 182, 134) if bpm <= 18 and gaze_direction == "Center" else (0, 0, 255)
            cv2.putText(frame, focus_status, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
