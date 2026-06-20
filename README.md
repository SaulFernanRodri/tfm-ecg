# ECG Multimodal Classifier — TFM

> **Trabajo de Fin de Máster** · Detección de Infarto de Miocardio mediante Deep Learning Multimodal sobre ECG de 12 derivaciones

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-green)](https://physionet.org/content/ptb-xl/1.0.3/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://tfm-ecg-wt5ccz6byxv3uticy4gq66.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/SaulFernanRodri/ecg-diagnosis-ai)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🚀 Demostración en Vivo (Web App)

El modelo entrenado ha sido desplegado en forma de aplicaciones web interactivas para facilitar su evaluación por médicos o investigadores. En estas aplicaciones puedes probar la herramienta subiendo tus propios registros de ECG o utilizando los casos de ejemplo proporcionados:

- **Streamlit App**: [https://tfm-ecg-wt5ccz6byxv3uticy4gq66.streamlit.app/](https://tfm-ecg-wt5ccz6byxv3uticy4gq66.streamlit.app/)
- **Hugging Face Spaces**: [https://huggingface.co/spaces/SaulFernanRodri/ecg-diagnosis-ai](https://huggingface.co/spaces/SaulFernanRodri/ecg-diagnosis-ai)

*(La aplicación cuenta con una interfaz que no solo clasifica el electrocardiograma, sino que además incorpora módulos de Interpretabilidad o XAI para explicar las predicciones del modelo).*

---

## Tabla de Contenidos

- [Demostración en Vivo](#-demostración-en-vivo-web-app)
- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Dataset](#dataset)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Entrenamiento](#entrenamiento)
  - [Evaluación](#evaluación)
  - [Inferencia](#inferencia)
- [Resultados](#resultados)
- [Licencia](#licencia)

---

## Descripción

Este proyecto presenta una solución avanzada e innovadora para el análisis cardiológico, implementando un sistema de apoyo al diagnóstico clínico basado en **Deep Learning multimodal**. A diferencia de los enfoques convencionales que analizan el electrocardiograma de forma aislada, nuestro modelo imita el razonamiento clínico combinando de manera sinérgica dos fuentes de información fundamentales:

- **Señales de ECG de 12 derivaciones**: Capturando la dinámica espacial y temporal de la actividad eléctrica del corazón (1.000 muestras por cada uno de los 12 canales a 100 Hz).
- **Metadatos clínicos del paciente**: Integrando el contexto fisiológico del individuo mediante variables clave (edad, sexo, altura, peso y frecuencia cardíaca).

### Objetivos y Enfoque

El modelo está diseñado para clasificar registros cardiológicos en **5 superclases diagnósticas del estándar internacional SCP-ECG**. Esta categorización permite detectar y diferenciar patologías complejas como:
- Diferentes tipologías y localizaciones de infartos de miocardio (IAM anterior, inferior, lateral...).
- Diversas arritmias.
- Bloqueos de rama y alteraciones de la conducción eléctrica.
- Patrones de ritmo sinusal normal.

Dada la criticidad de un diagnóstico omitido en cardiología, la arquitectura prioriza fundamentalmente la seguridad del paciente. Por ello, la métrica rectora del proyecto es **maximizar la sensibilidad (recall) macro** (con un objetivo de ≥ 0.90), asegurando que ningún caso positivo de infarto pase inadvertido, mientras se mantiene un AUC-ROC macro competitivo frente al actual estado del arte en IA médica.

---

## Arquitectura

El modelo implementa una **fusión multimodal tardía** (*late fusion*):

```text
ECG (1000×12)                  Metadatos (4,)
      │                              │
      ▼                              ▼
 ResNet1D                          MLP
 5 bloques residuales              Dense(32) → Dense(64)
 Conv1D(64/128/256/384/512)        BatchNorm + Dropout(0.3)
 GlobalAvgPool → (512,)            → (64,)
      │                              │
      └──────────────┬───────────────┘
                     ▼
              Concat → (576,)
              Dense(256) → Dense(128) + BN + ReLU + Dropout(0.4)
                     ▼
              Dense(5, sigmoid)
              5 superclases SCP-ECG
```

### Componentes principales

| Módulo | Descripción |
|--------|-------------|
| `model/resnet1d.py` | Bloques residuales 1D para señales ECG |
| `model/mlp.py` | Rama tabular para metadatos clínicos |
| `model/fusion.py` | Fusión de ramas y clasificador final |
| `model/losses.py` | Pérdidas personalizadas (Focal Loss, etc.) |
| `model/calibration.py` | Calibración de probabilidades de salida |
| `data/preprocessor.py` | Normalización de señal y variables clínicas |
| `data/augmentation.py` | Aumentación de datos en ECG |
| `data/loader.py` | Carga del dataset PTB-XL (WFDB) |
| `data/pipeline.py` | Pipeline `tf.data` con sample weights |
| `training/train.py` | Bucle de entrenamiento con MLflow |
| `evaluation/evaluate.py` | Métricas, umbrales óptimos y visualizaciones |
| `utils/metrics.py` | AUC-ROC, F1, sensibilidad y especificidad |
| `utils/mlflow_logger.py` | Logging de experimentos con MLflow |

---

## Dataset

Se utiliza el **PTB-XL** (*Physikalisch-Technische Bundesanstalt Extended*), el mayor dataset público de ECG anotado disponible.

| Característica | Valor |
|----------------|-------|
| Registros | 21.799 |
| Pacientes únicos | 18.869 |
| Derivaciones | 12 (estándar clínico) |
| Frecuencia de muestreo | 100 Hz / 500 Hz |
| Duración por registro | 10 segundos |
| Etiquetas | SCP-ECG (multilabel) |
| Licencia | Creative Commons BY 4.0 |

> El dataset **no está incluido** en este repositorio. Descárgalo desde [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) y colócalo en `physionet.org/files/ptb-xl/1.0.3/`.

---

## Estructura del Proyecto

```text
tfm-ecg/
├── main.py                   # Punto de entrada principal (entrenamiento y evaluación)
├── requirements.txt          # Dependencias Python
├── data/
│   ├── loader.py             # Carga de registros WFDB
│   ├── preprocessor.py       # Normalización y preprocesamiento
│   ├── augmentation.py       # Aumentación de señal ECG
│   └── pipeline.py           # Pipeline tf.data
├── model/
│   ├── resnet1d.py           # Arquitectura ResNet 1D (5 bloques)
│   ├── mlp.py                # Red tabular MLP
│   ├── fusion.py             # Modelo multimodal completo
│   ├── losses.py             # Funciones de pérdida (ASL, ASL per-class)
│   └── calibration.py        # Calibración de salidas
├── training/
│   └── train.py              # Lógica de entrenamiento
├── evaluation/
│   └── evaluate.py           # Evaluación y métricas
├── utils/
│   ├── metrics.py            # Funciones de métricas
│   ├── mlflow_logger.py      # Logger MLflow
│   └── seed.py               # Semilla de reproducibilidad
├── results/                  # Métricas e historial de entrenamiento
└── saved_model/              # Modelos guardados (no en git)
```

---

## Instalación

### Requisitos

- Python 3.10+
- CUDA 11.8+ (opcional, para entrenamiento con GPU)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/SaulFernanRodri/tfm-ecg.git
cd tfm-ecg

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset PTB-XL
#    Coloca los archivos en: physionet.org/files/ptb-xl/1.0.3/
```

---

## Uso

### Entrenamiento y Evaluación

El script principal orquesta el pipeline completo de extremo a extremo, desde la carga de datos hasta la evaluación del modelo en el conjunto de test. No es necesario indicar un modo de ejecución, ya que realiza el entrenamiento y la evaluación de forma secuencial:

```bash
python main.py
```

El script registrará automáticamente métricas, hiperparámetros y artefactos en **MLflow**, y guardará los resultados de la evaluación en `results/v6.2/`. Para visualizar los experimentos:

```bash
mlflow ui
# Abre http://localhost:5000
```

Genera en `results/v6.2/`:
- `metrics.json` — AUC-ROC, F1, sensibilidad y especificidad por clase
- `metrics_baseline.json` — métricas del clasificador base
- `training_history.json` — historial de pérdida y métricas por época
- `plots/` — curvas ROC, matrices de confusión y curvas de aprendizaje

### Inferencia

```python
import tensorflow as tf
import joblib
import numpy as np

# Cargar modelo y artefactos
model  = tf.keras.models.load_model("saved_model/v6.2/best_model.keras",
                                    custom_objects={"AsymmetricLossPerClass": AsymmetricLossPerClass})
scaler = joblib.load("saved_model/scaler.joblib")

# ecg_signal: (1, 1000, 12)  — señal ECG normalizada (z-score global)
# metadata:   (1, 4)          — [age, sex, height, weight] escalados
predictions = model.predict({"ecg_input": ecg_signal, "clinical_input": metadata})
# predictions: (1, 5) — probabilidad por superclase [CD, HYP, MI, NORM, STTC]
```

---

## Resultados

Los resultados completos de cada versión se encuentran en `results/`. A continuación se muestra un resumen de las métricas clave sobre el conjunto de test:

| Versión | AUC-ROC macro | F1 macro | Sensibilidad macro |
|---------|---------------|----------|--------------------|
| v6.2    | 0.9255        | 0.6865   | 0.9029             |

--

<p align="center">
  Trabajo de Fin de Máster · 2026
</p>
