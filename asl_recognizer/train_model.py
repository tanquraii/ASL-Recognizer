import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("../data/dataset.csv")

X = data.drop(columns = ["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

neigh.fit(X_train_scaled, y_train)

predictions_knn = neigh.predict(X_test_scaled)

print(f"Accuracy score for KNN: {(accuracy_score(y_test, predictions_knn)*100):.2f}%")
print(f"Macro F1 Score for KNN: {(f1_score(y_test, predictions_knn, average='macro')*100):.2f}%")

cm = confusion_matrix(y_test, predictions_knn)
display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = neigh.classes_)
display.plot()
plt.show()

joblib.dump(neigh, "../models/knn.pkl")
joblib.dump(scaler, "../models/scaler.pkl")