##%
# En este script, utilizamos la librería DeepFace para detectar, extraer y verificar rostros en dos imágenes.
# Importamos las imágenes desde la carpeta 'dataset', extraemos los rostros y luego verificamos si ambos rostros pertenecen a la misma persona utilizando el modelo ArcFace.
# Finalmente, mostramos los resultados de la verificación.
##%

import os

import cv2

# import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace

# Importando pares de fotos desde sus respectivas rutas en 'dataset'
img1 = plt.imread('FotoTest1.jpg')
img2 = plt.imread('foto1.jpeg')

# Redimensionando a escala 224x224 (opcional, aunque afecta el rendimiento del modelo)
#img1_resized = cv2.resize(img1, (224, 224))
#img2_resized = cv2.resize(img2, (224, 224))

# Mostrando las fotos originales
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.xlabel('Foto 1 (Original)')
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.xlabel('Foto 2 (Original)')
plt.show()

## DETECANDO Y EXTRAYENDO ROSTROS

Datos_extraidos1 = DeepFace.extract_faces(img1, align = False) # align = False desactiva la alineacion de rostros
Datos_extraidos2 = DeepFace.extract_faces(img2, align = False)

# Comprobando si se extrajeron rostros
plt.figure(figsize=(10, 5))
if Datos_extraidos1:
    EF1 = Datos_extraidos1[0]["face"] # Almacena la primera cara detectada
    plt.subplot(1, 2, 1)
    plt.imshow(EF1)
    plt.xlabel('Foto 1 (Cara extraida)')

    # La primera cara detectada se almacena en un archivo temporal
    face1_path = "face1.jpg"
    face1_data = (EF1 * 255).astype(np.uint8)
    cv2.imwrite(face1_path, cv2.cvtColor(face1_data, cv2.COLOR_BGR2RGB))

else:
    plt.subplot(1, 2, 1)
    plt.text(0.5, 0.5, "No face detected", horizontalalignment='center', verticalalignment='center')
    face1_path = None # No almacena si no detecta rostros


if Datos_extraidos2:
    EF2 = Datos_extraidos2[0]["face"]
    plt.subplot(1, 2, 2)
    plt.imshow(EF2)
    plt.xlabel('Foto 2 (Cara extraida)')

    face2_path = "face2.jpg"
    face2_data = (EF2 * 255).astype(np.uint8)
    cv2.imwrite(face2_path, cv2.cvtColor(face2_data, cv2.COLOR_RGB2BGR))

else:
    plt.subplot(1, 2, 2)
    plt.text(0.5, 10, "No face detected", horizontalalignment='center', verticalalignment='center')
    face2_path = None

plt.show()

## VERIFICANDO SI AMBOS ROSTROS FUERON EXTRAIDOS

if face1_path and face2_path:
    try:
        result = DeepFace.verify(face1_path, face2_path,model_name="ArcFace")
        print(result)
    except ValueError as e:
        print(f"Verificacion de rostros fallida: {e}")
else:
    print("1 o mas rostros no fueron detectados.")

## Borrando los archivos temporales creados
if face1_path and os.path.exists(face1_path):
    os.remove(face1_path)
if face2_path and os.path.exists(face2_path):
    os.remove(face2_path)