# ASL Recognition Using Hand Landmarks and Machine Learning

A real-time American Sign Language (ASL) hand gesture recognition project built using **Python, MediaPipe, OpenCV, and Scikit-learn**.

The system detects hand landmarks from webcam frames, extracts normalized spatial features, and uses a trained K-Nearest Neighbors (KNN) classifier to recognize static hand gestures.

## Overview

The project demonstrates a classical machine learning approach to hand gesture recognition without training a deep neural network.

Instead of using raw images as input to the classifier, the system extracts **21 hand landmarks** using MediaPipe. Each landmark contains three coordinates (x, y, z), resulting in a 63-dimensional feature vector.

These features are normalized and passed to a machine learning model for classification.

### Technologies Used

- **Python** — Core programming language
- **MediaPipe** — Hand detection and landmark extraction
- **OpenCV** — Image processing and real-time webcam capture
- **NumPy** — Numerical operations and feature normalization
- **Pandas** — Dataset creation and manipulation
- **Scikit-learn** — Feature scaling, model training and evaluation
- **Matplotlib** — Confusion matrix visualization
- **Joblib** — Model serialization

## How It Works

The recognition pipeline consists of five stages:

**1. Hand Landmark Detection**

MediaPipe Hand Landmarker detects 21 landmarks representing the hand's joints and key points.

**2. Feature Extraction**

Each landmark provides three spatial coordinates:

- x — Horizontal coordinate
- y — Vertical coordinate
- z — Relative depth coordinate

This produces a feature vector containing 63 values.

**3. Feature Normalization**

To reduce sensitivity to hand position and scale, the wrist landmark is used as the reference point.

For each landmark, its coordinates are translated relative to the wrist and divided by the maximum 2D distance between the wrist and any detected landmark.

**4. Model Training**

The normalized features are used to train a K-Nearest Neighbors classifier.

The dataset is divided into training and testing subsets using an 80/20 stratified split.

A StandardScaler is fitted on the training data and applied to both training and testing features.

**5. Real-Time Prediction**

During inference:

- OpenCV captures webcam frames.
- MediaPipe detects hand landmarks.
- Landmarks are normalized and scaled.
- The trained KNN classifier predicts the gesture.
- Predictions are smoothed using a majority-voting mechanism.
- The recognized letter is displayed on the webcam feed.

## Project Structure

```text
ASL-Recognition/
├── asl_recognizer/
│   ├── data_processing.py
│   ├── train_model.py
│   ├── main.py
│   └── utils.py
├── data/
│   └── dataset.csv
├── dataset/
│   ├── train_data/
│   └── test_data/
├── models/
│   ├── knn.pkl
│   └── scaler.pkl
├── hand_landmarker.task
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `data_processing.py` | Extracts and normalizes hand landmarks from images and creates a CSV dataset |
| `train_model.py` | Trains the KNN classifier, evaluates performance and saves the trained model |
| `main.py` | Performs real-time hand gesture recognition using a webcam |
| `utils.py` | Implements prediction smoothing using majority voting |

## Model Evaluation

The model is evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified samples.
- **Macro F1-score:** Average F1-score across all gesture classes, giving equal weight to each class.
- **Confusion Matrix:** Visualization of classification performance and misclassified gestures.

The KNN classifier currently uses `n_neighbors=3`.

## Running the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ASL-Recognition
```

### 2. Install Dependencies

```bash
pip install mediapipe opencv-python numpy pandas scikit-learn matplotlib joblib
```

### 3. Run Real-Time Recognition

From the project root:

```bash
cd asl_recognizer
python main.py
```

The application opens a webcam window and displays the detected hand landmarks and predicted gesture.

Press **Q** to exit.

### 4. Train the Model (Optional)

To retrain the classifier using the prepared dataset:

```bash
python train_model.py
```

The trained classifier and scaler are saved in the `models/` directory.

## Prediction Stabilization

Real-time predictions can fluctuate between consecutive frames due to changes in hand position, detection noise, or ambiguous gestures.

To reduce this fluctuation, the project maintains a history of the 10 most recent predictions.

The most frequent predicted letter is displayed only when its frequency reaches a predefined threshold of 80%.

This provides a simple majority-voting mechanism without requiring an additional temporal model.

## Limitations

- The current implementation recognizes static hand gestures rather than continuous sign language.
- Dynamic gestures and full sentence translation are not supported.
- Recognition accuracy depends on lighting, hand orientation, and landmark detection quality.
- The model may perform differently on webcam images compared with the training dataset.
- Only one hand is processed at a time.

## Future Improvements

- Evaluate additional classifiers such as Support Vector Machines and Random Forest.
- Improve generalization using a more diverse dataset.
- Introduce confidence-based filtering.
- Implement text accumulation for recognized letters.
- Optimize real-time inference performance.
- Explore gesture-based human–robot interaction.

## Author

Developed as a personal machine learning and computer vision project focused on hand landmark extraction, feature engineering, classical machine learning, and real-time inference.# ASL Recognition Using Hand Landmarks and Machine Learning

A real-time American Sign Language (ASL) hand gesture recognition project built using **Python, MediaPipe, OpenCV, and Scikit-learn**.

The system detects hand landmarks from webcam frames, extracts normalized spatial features, and uses a trained K-Nearest Neighbors (KNN) classifier to recognize static hand gestures.

## Overview

The project demonstrates a classical machine learning approach to hand gesture recognition without training a deep neural network.

Instead of using raw images as input to the classifier, the system extracts **21 hand landmarks** using MediaPipe. Each landmark contains three coordinates (x, y, z), resulting in a 63-dimensional feature vector.

These features are normalized and passed to a machine learning model for classification.

### Technologies Used

- **Python** — Core programming language
- **MediaPipe** — Hand detection and landmark extraction
- **OpenCV** — Image processing and real-time webcam capture
- **NumPy** — Numerical operations and feature normalization
- **Pandas** — Dataset creation and manipulation
- **Scikit-learn** — Feature scaling, model training and evaluation
- **Matplotlib** — Confusion matrix visualization
- **Joblib** — Model serialization

## How It Works

The recognition pipeline consists of five stages:

**1. Hand Landmark Detection**

MediaPipe Hand Landmarker detects 21 landmarks representing the hand's joints and key points.

**2. Feature Extraction**

Each landmark provides three spatial coordinates:

- x — Horizontal coordinate
- y — Vertical coordinate
- z — Relative depth coordinate

This produces a feature vector containing 63 values.

**3. Feature Normalization**

To reduce sensitivity to hand position and scale, the wrist landmark is used as the reference point.

For each landmark, its coordinates are translated relative to the wrist and divided by the maximum 2D distance between the wrist and any detected landmark.

**4. Model Training**

The normalized features are used to train a K-Nearest Neighbors classifier.

The dataset is divided into training and testing subsets using an 80/20 stratified split.

A StandardScaler is fitted on the training data and applied to both training and testing features.

**5. Real-Time Prediction**

During inference:

- OpenCV captures webcam frames.
- MediaPipe detects hand landmarks.
- Landmarks are normalized and scaled.
- The trained KNN classifier predicts the gesture.
- Predictions are smoothed using a majority-voting mechanism.
- The recognized letter is displayed on the webcam feed.

## Project Structure

```text
ASL-Recognition/
├── asl_recognizer/
│   ├── data_processing.py
│   ├── train_model.py
│   ├── main.py
│   └── utils.py
├── data/
│   └── dataset.csv
├── dataset/
│   ├── train_data/
│   └── test_data/
├── models/
│   ├── knn.pkl
│   └── scaler.pkl
├── hand_landmarker.task
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `data_processing.py` | Extracts and normalizes hand landmarks from images and creates a CSV dataset |
| `train_model.py` | Trains the KNN classifier, evaluates performance and saves the trained model |
| `main.py` | Performs real-time hand gesture recognition using a webcam |
| `utils.py` | Implements prediction smoothing using majority voting |

## Model Evaluation

The model is evaluated using the following metrics:

- **Accuracy:** Overall proportion of correctly classified samples.
- **Macro F1-score:** Average F1-score across all gesture classes, giving equal weight to each class.
- **Confusion Matrix:** Visualization of classification performance and misclassified gestures.

The KNN classifier currently uses `n_neighbors=3`.

## Running the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ASL-Recognition
```

### 2. Install Dependencies

```bash
pip install mediapipe opencv-python numpy pandas scikit-learn matplotlib joblib
```

### 3. Run Real-Time Recognition

From the project root:

```bash
cd asl_recognizer
python main.py
```

The application opens a webcam window and displays the detected hand landmarks and predicted gesture.

Press **Q** to exit.

### 4. Train the Model (Optional)

To retrain the classifier using the prepared dataset:

```bash
python train_model.py
```

The trained classifier and scaler are saved in the `models/` directory.

## Prediction Stabilization

Real-time predictions can fluctuate between consecutive frames due to changes in hand position, detection noise, or ambiguous gestures.

To reduce this fluctuation, the project maintains a history of the 10 most recent predictions.

The most frequent predicted letter is displayed only when its frequency reaches a predefined threshold of 80%.

This provides a simple majority-voting mechanism without requiring an additional temporal model.

## Limitations

- The current implementation recognizes static hand gestures rather than continuous sign language.
- Dynamic gestures and full sentence translation are not supported.
- Recognition accuracy depends on lighting, hand orientation, and landmark detection quality.
- The model may perform differently on webcam images compared with the training dataset.
- Only one hand is processed at a time.

## Future Improvements

- Evaluate additional classifiers such as Support Vector Machines and Random Forest.
- Improve generalization using a more diverse dataset.
- Introduce confidence-based filtering.
- Implement text accumulation for recognized letters.
- Optimize real-time inference performance.
- Explore gesture-based human–robot interaction.

## Author

Developed as a personal machine learning and computer vision project focused on hand landmark extraction, feature engineering, classical machine learning, and real-time inference.