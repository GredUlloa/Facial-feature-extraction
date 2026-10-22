import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Definicion de rutas
WORK_DIR = Path(__file__).resolve().parent

DATA_PATH = os.path.join(WORK_DIR, "Datasets")

RESULTS_DIR = WORK_DIR / "Resultados" # variable tipo 'path' para mkdir
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, "svm_insightface.pkl")
ENCODER_PATH = os.path.join(RESULTS_DIR, "label_encoder.pkl")
 

## Implementando Buffalo_1 con aceleracion por CPU
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


## ______________ EXTRACCION DE EMBEDDINGS ______________
# Listas para características y etiquetas
X, y = [], [] 

for person in os.listdir(DATA_PATH):
    person_path = os.path.join(DATA_PATH, person) # Ruta para cada dataset individual

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name) # Ruta para cada imagen
        img = cv2.imread(img_path)
        faces = app.get(img)

        if len(faces) > 0:
            embedding = faces[0].embedding # Características del primer rostro detectado
            X.append(embedding) # Añadiendo embedding
            y.append(person)    # Añadiendo etiqueta correspondiente

X = np.array(X)
y = np.array(y)

# Codificacion de etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

## ______________ ENTRENAMIENTO _______________ 
clf = SVC(kernel='linear', probability=True) # Clasificador SVM

clf.fit(X, y_encoded)

# Guardando los pesos y etiquedas del modelo entrenado
joblib.dump(clf, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)


## ______________ COMPARACION 1:N ________________
def predict_image(img_path):
    img = cv2.imread(img_path)
    faces = app.get(img)

    if len(faces) == 0:
        return "No se detectó rostro", 0.0
    
    embedding = faces[0].embedding.reshape(1, -1)
    pred = clf.predict(embedding)[0] # Predicción del modelo entrenado
    prob = clf.predict_proba(embedding).max()   # Probabilidad mediante validación cruzada

    return le.inverse_transform([pred])[0], prob


## Test de predicción
img_test = os.path.join(WORK_DIR, "Datasets/AndresC.jpg")
nombre, confianza = predict_image(img_test)
print(" ________________ RESULTADOS DEL TEST: ________________")
print(f"Predicción: {nombre} || confianza: {confianza:.2f}")