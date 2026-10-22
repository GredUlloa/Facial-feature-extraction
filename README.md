# Extracción de Características con un Modelo ArcFace y Clasificación SVM para Verificación Facial.

**Extracción de Características** consiste en transformar datos brutos en representaciones más compactas y relevantes que pueden ser utilizadas por algoritmos de inteligencia artificial para aprender, clasificar o reconocer patrones. 

En este proyecto se utiliza la función de pérdida **ArcFace** mediante el modelo preeentrenado *Buffalo_1* para obtener embeddings de un dataset de rostros faciales personalizado, y usando estos embeddings se entrena un modelo clasificador **SVM** que realiza **verificaciones 1:N** respecto al número de personas en el dataset.

Tecnologías usadas:

* **Insightface**: Es un proyecto de código abierto que representa el estado del arte para el análisis facial 2D y 3D, implementa la función de pérdida *ArcFace*, y permite extraer embeddings altamente discriminantes en cada imagen.
* **onnxruntime**: Motor de inferencia rápido y eficiente, usado por Insightface para ejecutar modelos de deep learning usando únicamente la CPU. La aceleración por GPU necesita la instalación adicional del paquete `onnxruntime-gpu`.
* **Opencv**: Usado para la inicialización y preprocesamiento de las imágenes del dataset.
* **Numpy**: Permite realizar operaciones matemáticas sobre arreglos vectoriales o matriciales, en este proyecto se aplica con las matrices de embeddings y etiquetados(*labels*) para realizar el entrenamiento del modelo SVM. 
* **Scikit-learn**: Proporciona el modelo de machine-learning clásico **Suport Vector Machine** (*SVM*) para la clasificación mediante embeddings y codificación de etiquetas.


## Requerimientos

Se trabaja con Python 3.12+ y las librerias del bloque "dependencies" en el archivo `pyproject.toml`:
```
...
dependencies = [
    "numpy>=2,<2.3.0",
    "opencv-python==4.12.0.88",
    "insightface>=2.0,<3",
    "onnxruntime>=1.19",
    "scikit-learn>=1.4",
    "scikit-image>=0.22",
    "joblib",
]
...
```

## Configuración

El proyecto utiliza UV como gestor de paquetes; una vez clonado el proyecto localmente (paso 1), UV permite crear entornos virtuales e instalar dependencias de forma rápida y unificada mediante un único comando (paso 2).

1. Descarga o clona el proyecto en una terminal:

```bash
git clone https://github.com/GredUlloa/Facial-feature-extraction.git
cd # copia la ruta del proyecto
``` 

2. Crea un entorno virtual `.venv` e instala las dependencias con UV, definidos en el archivo `pyproject.toml`:

```bash
uv sync

```

3. Activa el entorno creado:

```bash
# En Linux/Mac
source venv/bin/activate
# En windows
# .\venv\Scripts\activate
```

## Funcionamiento

### 1. Creacion de los datasets

Considerando `src/facerec/` la ruta relativa del proyecto, dentro de `./Datasets` se encuentra por defecto los siguientes archivos que corresponden a imágenes de una misma persona:

* `/AndresC` (7 fotos)
* `AndresC.jpg` (respectiva foto de prueba)

Para que el modelo SVM realice una verificación es necesario disponer de al menos 2 clases (personas) y además una foto nueva que sirva como test de predicción para cada clase:

* `/persona2` (Dataset personalizado)
* `FT2.jpg` (respectiva foto de prueba)

Esto permite al modelo identificar **1:N** entre las diferentes personas registradas en `./Datasets`.

### 2. Ejecutar comparaciones 1:N

Dentro del archivo `verif.py` se define la ruta de la imagen de prueba en ```img_path``` (por defecto `AndresC.jpg`); y se procede a ejecutar el archivo con UV:

```bash
uv run .\src\facerec\verif.py
```

Finalmente, se devuelven y en el directorio del proyecto se guardan los siguientes archivos:

* `svm_insightface.pkl`: Pesos del modelo SVM una vez entrenado
* `label_encoder.pkl`: Etiquetas para el rostro de cada persona en los datasets