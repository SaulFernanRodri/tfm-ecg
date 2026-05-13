# ECG Multimodal Classifier — TFM

> **Trabajo de Fin de Máster** · Detección de Infarto de Miocardio mediante Deep Learning Multimodal sobre ECG de 12 derivaciones

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-green)](https://physionet.org/content/ptb-xl/1.0.3/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Tabla de Contenidos

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

Este proyecto implementa un sistema de apoyo al diagnóstico cardíaco basado en **Deep Learning multimodal**. El modelo combina:

- **Señales ECG** de 12 derivaciones (1.000 muestras × 12 canales a 100 Hz)
- **Metadatos clínicos** del paciente (edad, sexo, altura, peso, frecuencia cardíaca)

El objetivo es clasificar registros en **23 subclases diagnósticas del estándar SCP-ECG**, incluyendo diferentes tipos de infarto de miocardio (IAM anterior, inferior, lateral…), arritmias, bloqueos de rama y ritmo sinusal normal.

La prioridad clínica del sistema es maximizar la **sensibilidad** (≥ 0.90 macro) para no perder ningún infarto, con el AUC-ROC macro como métrica principal de comparación con el estado del arte.

---

## Arquitectura

El modelo implementa una **fusión multimodal tardía** (*late fusion*):

```
ECG (1000×12)           Metadatos (5,)
      │                       │
      ▼                       ▼
 ResNet1D                   MLP
 3 bloques residuales       Dense(32) → Dense(64)
 Conv1D(64/128/256)         BatchNorm + Dropout(0.3)
 GlobalAvgPool → (256,)     → (64,)
      │                       │
      └───────────┬───────────┘
                  ▼
           Concat → (320,)
           Dense(128) + BN + ReLU + Dropout(0.4)
                  ▼
           Dense(23, sigmoid)
           23 subclases SCP-ECG
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

```
tfm-ecg/
├── main.py                   # Punto de entrada principal
├── requirements.txt          # Dependencias Python
├── data/
│   ├── loader.py             # Carga de registros WFDB
│   ├── preprocessor.py       # Normalización y preprocesamiento
│   ├── augmentation.py       # Aumentación de señal ECG
│   └── pipeline.py           # Pipeline tf.data
├── model/
│   ├── resnet1d.py           # Arquitectura ResNet 1D
│   ├── mlp.py                # Red tabular MLP
│   ├── fusion.py             # Modelo multimodal completo
│   ├── losses.py             # Funciones de pérdida
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
git clone https://github.com/<tu-usuario>/tfm-ecg.git
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

### Entrenamiento

```bash
python main.py --mode train
```

El entrenamiento registra automáticamente métricas, hiperparámetros y artefactos en **MLflow**. Para visualizar los experimentos:

```bash
mlflow ui
# Abre http://localhost:5000
```

### Evaluación

```bash
python main.py --mode evaluate
```

Genera en `results/`:
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
model  = tf.keras.models.load_model("saved_model/v5/best_model.keras",
                                    custom_objects={"AsymmetricLoss": AsymmetricLoss})
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
| v4      | —             | —        | —                  |
| v5      | —             | —        | —                  |
| latest  | —             | —        | —                  |

> Rellena esta tabla con los valores de `results/metrics.json` tras el entrenamiento.

---

## Licencia

Este proyecto está publicado bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

El dataset PTB-XL está licenciado bajo [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) por Wagner et al. (2020). Cita requerida:

> Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45

---

<p align="center">
  Trabajo de Fin de Máster · 2026
</p>
