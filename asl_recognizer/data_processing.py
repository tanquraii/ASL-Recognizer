from pathlib import Path
import mediapipe as mp
import cv2
import pandas as pd

folder_path = Path("dataset/train_data")
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

def extract_features(hand):
    normalized = []
    wrist = hand[0]
    max_distance = 0
    for landmark in hand:
        distance = ((landmark.x - wrist.x)**2 + (landmark.y - wrist.y)**2)**0.5
        if distance > max_distance:
            max_distance = distance
    max_distance = max(1e-6, max_distance) #this avoids division by zero
    for landmark in hand:
        normalized.extend([(landmark.x - wrist.x)/max_distance, (landmark.y - wrist.y)/max_distance, (landmark.z - wrist.z)/max_distance])
    assert len(normalized) == 63
    return normalized

columns = []
for i in range(21):
    for coord in ["x", "y", "z"]:
        columns.append(f"{coord}{i}")
columns.append("label")

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        dataset = []
        for item in folder_path.iterdir():
            for image in list(item.iterdir())[:150]:
                instance = cv2.imread(str(image))
                if instance is None:
                    continue

                rgb_frame = cv2.cvtColor(instance, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                result = landmarker.detect(mp_image)

                for hand in result.hand_landmarks:
                    landmarks = extract_features(hand)
                    row = landmarks + [item.name]
                    dataset.append(row)
    df = pd.DataFrame(data = dataset, columns = columns)
    df.to_csv("data/dataset.csv", index = False)


except KeyboardInterrupt:
    print("Keyboard Interrupt")