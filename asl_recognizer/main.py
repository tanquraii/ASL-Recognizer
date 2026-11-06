import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

from utils import get_stable_prediction

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

knn = joblib.load("../models/knn.pkl")
scaler = joblib.load("../models/scaler.pkl")

connections = [
    (c.start, c.end)
    for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
]


def model_predict(landmarks):
    landmarks_scaled = scaler.transform(landmarks)
    prediction = knn.predict(landmarks_scaled)
    return prediction[0]


def extract_features(hand):
    coords = np.array(
        [(landmark.x, landmark.y, landmark.z) for landmark in hand],
        dtype=np.float64
    )

    wrist = coords[0]
    coords -= wrist

    max_distance = np.max(
        np.linalg.norm(coords[:, :2], axis=1)
    )
    max_distance = max(1e-6, max_distance)

    normalized = (coords / max_distance).reshape(1, -1)

    assert normalized.shape == (1, 63)
    return normalized


def draw_hand_skeleton(frame, hand):
    height, width = frame.shape[:2]

    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand
    ]

    for start, end in connections:
        cv2.line(frame, points[start], points[end], (0, 0, 255), 1)

    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), 2)

    return frame


cap = cv2.VideoCapture(0)

# Request a smaller camera frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Reduce buffered frames where supported
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

predictions = []
start_time = time.perf_counter()
last_timestamp = -1

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )
            timestamp_ms = max(timestamp_ms, last_timestamp + 1)
            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if not result.hand_landmarks:
                predictions.clear()

            for hand in result.hand_landmarks:
                draw_hand_skeleton(frame, hand)

                landmarks = extract_features(hand)
                prediction = model_predict(landmarks)

                predictions.append(prediction)

                if len(predictions) > 10:
                    predictions.pop(0)

                stable_letter = get_stable_prediction(
                    predictions, 0.8
                )

                if stable_letter is not None:
                    cv2.putText(
                        frame,
                        stable_letter,
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )

            cv2.imshow("ASL Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
