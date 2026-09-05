# Extracción de Características con un Modelo ArcFace y Clasificación SVM para Verificación Facial.

**Extracción de Características** consiste en transformar datos brutos en representaciones más compactas y relevantes que pueden ser utilizadas por algoritmos de inteligencia artificial para aprender, clasificar o reconocer patrones. 

En este proyecto se utiliza la función de pérdida **ArcFace** mediante el modelo preeentrenado *LResNet100E-IR* para obtener embeddings de un dataset de rostros faciales personalizado, y usando estos embeddings se entrena un modelo clasificador se realizan **verificaciones 1:1** con nuevas imágenes.

Tecnologías usadas:

* **Insightface**: Es un proyecto de código abierto que representa el estado del arte para el análisis facial 2D y 3D, implementa la función de pérdida *ArcFace*, y permite extraer embeddings altamente discriminantes en cada imagen.
* **onnxruntime**: Motor de inferencia rápido y eficiente, usado por Insightface para ejecutar modelos de deep learning usando únicamente la CPU. La aceleración por GPU necesita la instalación adicional del paquete `onnxruntime-gpu`.
* **Opencv**: Usado para la inicialización y preprocesamiento de las imágenes del dataset.
* **Numpy**: Permite realizar operaciones matemáticas sobre arreglos vectoriales o matriciales, en este proyecto se aplica con las matrices de embeddings y etiquetados(*labels*) para realizar el entrenamiento del modelo SVM. 
* **Scikit-learn**: Proporciona el modelo de machine-learning clásico **Suport Vector Machine** (*SVM*) y codificación de etiquetas.


## Instalación

Requiere Python 3.8+ y las siguientes bibliotecas indicadas en el archivo `requirementes.txt`:
```
numpy
opencv-python
scikit-learn
insightface
onnxruntime
```

## Configuración

1. Descarga o clona el proyecto en una terminal:

```sh
git clone https://github.com/GredUlloa/FineTuning-ArcFace-con-SVM
cd # copia la ruta del proyecto
``` 

2. Crea un entorno con la version 3.8 de python:

```bash
py -3.8 -m venv venv
```

3. Activa el entorno creado:

```bash
# En Linux/Mac
source venv/bin/activate
# En windows
# .\venv\Scripts\activate
```

4. Instala las dependencias del proyecto:

```sh
pip install -r requirements.txt
```

## Modo de uso

### 1. Creacion de los datasets

En la ruta `\FaceRec\dataset` se encuentra por defecto los siguientes archivos que corresponden a fotos de una misma persona:

* `\Andres_C` (carpeta con 7 fotos)
* `Fototest2.jpg` (foto de prueba)

Para que el modelo SVM pueda realizar una predicción se necesitan al menos 2 datasets diferentes. Se debe crear otro dataset con el nombre de la otra persona y además una foto nueva que sirva como test de predicción:

* `\persona2` (carpeta con 7 o más fotos)
* `FotoTest1.jpg` (foto de prueba)

### 2. Ejecutar

En la terminal se ejecuta el código `Finetuning.py`:

```sh
python Finetuning.py
```

Finalmente en el entorno de trabajo se guardaran los siguientes archivos:

* `svm_insightface.pkl`: El modelo SVM entrenado con el dataset
* `label_encoder.pkl`: Las etiquetas codificadas para la predicción