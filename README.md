# ECG Multimodal Classifier — TFM

> **Trabajo de Fin de Máster** · Detección asistida de Infarto de Miocardio y otras patologías cardíacas mediante Deep Learning multimodal sobre ECG de 12 derivaciones

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-green)](https://physionet.org/content/ptb-xl/1.0.3/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://tfm-ecg-wt5ccz6byxv3uticy4gq66.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/SaulFernanRodri/ecg-diagnosis-ai)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📑 Tabla de contenidos

- [Demo en vivo](#-demo-en-vivo)
- [Resumen del proyecto](#-resumen-del-proyecto)
- [Motivación clínica](#-motivación-clínica)
- [Arquitectura del modelo](#-arquitectura-del-modelo)
- [Dataset](#-dataset)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Instalación](#-instalación)
- [Uso](#-uso)
  - [Entrenamiento y evaluación](#entrenamiento-y-evaluación)
  - [Explicabilidad (XAI)](#explicabilidad-xai)
  - [Inferencia](#inferencia)
  - [Seguimiento de experimentos con MLflow](#seguimiento-de-experimentos-con-mlflow)
- [Resultados](#-resultados)
- [Aplicación desplegada](#-aplicación-desplegada)
- [Solución de problemas](#-solución-de-problemas)
- [Roadmap](#-roadmap)
- [Licencia](#-licencia)
- [Cita](#-cita)
- [Autor](#-autor)

---

## 🚀 Demo en vivo

El modelo entrenado está desplegado como aplicación web interactiva, pensada para que médicos o investigadores puedan probarlo con sus propios registros o con casos de ejemplo:

| Plataforma | Enlace |
|---|---|
| Streamlit | [tfm-ecg-wt5ccz6byxv3uticy4gq66.streamlit.app](https://tfm-ecg-wt5ccz6byxv3uticy4gq66.streamlit.app/) |
| Hugging Face Spaces | [huggingface.co/spaces/SaulFernanRodri/ecg-diagnosis-ai](https://huggingface.co/spaces/SaulFernanRodri/ecg-diagnosis-ai) |

La aplicación no solo clasifica el electrocardiograma, sino que incorpora módulos de **interpretabilidad (XAI)** para explicar visualmente las predicciones del modelo.

---

## 📋 Resumen del proyecto

Este proyecto implementa un sistema de apoyo al diagnóstico clínico basado en **Deep Learning multimodal**. A diferencia de los enfoques que analizan el ECG de forma aislada, el modelo combina dos fuentes de información:

- **Señal de ECG de 12 derivaciones** — dinámica espacio-temporal de la actividad eléctrica cardíaca (1.000 muestras × 12 canales a 100 Hz, 10 segundos de registro).
- **Metadatos clínicos del paciente** — edad, sexo, altura y peso, aportando contexto fisiológico.

El modelo clasifica cada registro en las **5 superclases diagnósticas del estándar SCP-ECG** (clasificación multilabel):

| Superclase | Descripción |
|---|---|
| `MI` | Infarto de miocardio (distintas localizaciones) |
| `CD` | Trastornos de conducción / bloqueos de rama |
| `HYP` | Hipertrofia |
| `STTC` | Alteraciones del segmento ST / onda T |
| `NORM` | Ritmo sinusal normal |

## 🩺 Motivación clínica

Dada la criticidad de un diagnóstico de infarto omitido, la arquitectura y la estrategia de entrenamiento priorizan la **seguridad del paciente** por encima de la precisión bruta. La métrica rectora del proyecto es la **sensibilidad (recall) macro**, con un objetivo de **≥ 0.90**, para minimizar los falsos negativos, manteniendo al mismo tiempo un AUC-ROC competitivo frente al estado del arte en IA médica aplicada a PTB-XL.

---

## 🧠 Arquitectura del modelo

Fusión multimodal **tardía (late fusion)** entre una rama convolucional para la señal y una rama tabular para los metadatos:

```text
ECG (1000×12)                    Metadatos (4,)
      │                                │
      ▼                                ▼
  ResNet1D                            MLP
  5 bloques residuales          Dense(32) → Dense(64)
  Conv1D 64/128/256/384/512     BatchNorm + Dropout(0.3)
  + SE-Attention
  GlobalAvgPool → (512,)              → (64,)
      │                                │
      └────────────────┬───────────────┘
                        ▼
                 Concat → (576,)
        Dense(256) → Dense(128) + BN + ReLU + Dropout(0.4)
                        ▼
                Dense(5, sigmoid)
           [CD, HYP, MI, NORM, STTC]
```

**Pérdida:** `AsymmetricLossPerClass` — variante de Asymmetric Loss con `gamma_neg` configurable por clase, para ajustar de forma independiente el balance precisión/sensibilidad de cada patología (p. ej. `MI` prioriza recall, `HYP` prioriza precisión).

**Umbral de decisión:** optimizado por clase mediante F0.5-score con restricción mínima de sensibilidad, en lugar de usar un umbral fijo de 0.5.

### Componentes principales

| Módulo | Descripción |
|---|---|
| `model/resnet1d.py` | Bloques residuales 1D + atención SE para la señal ECG |
| `model/mlp.py` | Rama tabular para metadatos clínicos |
| `model/fusion.py` | Fusión de ramas y clasificador final |
| `model/losses.py` | Pérdidas personalizadas (Asymmetric Loss, por clase) |
| `model/calibration.py` | Calibración de probabilidades de salida |
| `data/loader.py` | Carga y etiquetado del dataset PTB-XL (WFDB) |
| `data/preprocessor.py` | Normalización de señal y variables clínicas |
| `data/augmentation.py` | Aumentación de señal ECG (ruido, escalado, desplazamiento, lead masking) |
| `data/pipeline.py` | Pipeline `tf.data` con pesos de muestra |
| `training/train.py` | Bucle de entrenamiento con *early stopping* y logging a MLflow |
| `evaluation/evaluate.py` | Métricas, optimización de umbrales y visualizaciones |
| `xai/` | Grad-CAM 1D, importancia por derivación y SHAP clínico |
| `utils/metrics.py` | AUC-ROC, F1, sensibilidad y especificidad |
| `utils/mlflow_logger.py` | Logging de experimentos y registro de modelos con MLflow |

---

## 📊 Dataset

Se utiliza **PTB-XL** (*Physikalisch-Technische Bundesanstalt Extended*), el mayor dataset público de ECG anotado disponible.

| Característica | Valor |
|---|---|
| Registros | 21.799 |
| Pacientes únicos | 18.869 |
| Derivaciones | 12 (estándar clínico) |
| Frecuencia de muestreo usada | 100 Hz |
| Duración por registro | 10 s |
| Etiquetas | SCP-ECG (multilabel, 5 superclases) |
| Licencia | Creative Commons BY 4.0 |
| Fuente | [PhysioNet — PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) |

> ⚠️ El dataset **no está incluido** en este repositorio por su tamaño y licencia. Debe descargarse manualmente y colocarse en la raíz del proyecto, respetando la ruta que espera el loader:
> ```text
> tfm-ecg/physionet.org/files/ptb-xl/1.0.3/
> ├── ptbxl_database.csv
> ├── scp_statements.csv
> └── records100/
> ```

El *split* train/val/test sigue la columna oficial `strat_fold` del dataset: folds 1–8 entrenamiento, fold 9 validación, fold 10 test.

---

## 🗂 Estructura del repositorio

```text
tfm-ecg/
├── main.py                    # Punto de entrada: entrenamiento + evaluación end-to-end
├── xai_main.py                # Punto de entrada: pipeline de explicabilidad (XAI)
├── requirements.txt           # Dependencias Python
├── data/
│   ├── loader.py               # Carga y etiquetado de registros WFDB
│   ├── preprocessor.py         # Normalización y preprocesamiento
│   ├── augmentation.py         # Aumentación de señal ECG
│   └── pipeline.py             # Pipeline tf.data
├── model/
│   ├── resnet1d.py              # Arquitectura ResNet 1D (5 bloques + SE-attention)
│   ├── mlp.py                   # Red tabular para metadatos clínicos
│   ├── fusion.py                # Modelo multimodal completo
│   ├── losses.py                # Asymmetric Loss (global y por clase)
│   └── calibration.py           # Calibración de salidas
├── training/
│   └── train.py                  # Lógica de entrenamiento
├── evaluation/
│   └── evaluate.py               # Evaluación, métricas y umbrales óptimos
├── xai/
│   ├── gradcam.py                # Grad-CAM 1D
│   ├── lead_importance.py        # Importancia por derivación
│   ├── shap_clinical.py          # SHAP sobre variables clínicas
│   ├── clinical_ablation.py      # Estudios de ablación clínica
│   └── visualize.py              # Visualizaciones XAI
├── utils/
│   ├── metrics.py                 # Funciones de métricas
│   ├── mlflow_logger.py           # Logger de experimentos MLflow
│   └── seed.py                    # Semilla de reproducibilidad
├── scripts/                    # Scripts auxiliares (plots, matriz de confusión, umbrales)
├── deployment/
│   ├── app/                     # Aplicación Streamlit desplegada
│   └── scripts/                 # Scripts de despliegue (export de datos demo, subida a HF)
├── results/                    # Métricas e historial de entrenamiento (generado)
└── saved_model/                # Modelos y umbrales guardados (generado, no versionado)
```

---

## ⚙️ Instalación

### Requisitos

- Python 3.10+
- CUDA 11.8+ (opcional, recomendado para entrenamiento con GPU)
- ~15 GB libres para el dataset PTB-XL a 100 Hz

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/SaulFernanRodri/tfm-ecg.git
cd tfm-ecg

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset PTB-XL
#    Colócalo en: physionet.org/files/ptb-xl/1.0.3/ (ver sección Dataset)
```

---

## ▶️ Uso

### Entrenamiento y evaluación

`main.py` orquesta el pipeline completo de extremo a extremo: carga de datos, preprocesamiento, entrenamiento del modelo multimodal y evaluación sobre el conjunto de test. No requiere argumentos ni modo de ejecución:

```bash
python main.py
```

El script:
1. Fija las semillas de reproducibilidad.
2. Carga y preprocesa el dataset PTB-XL.
3. Calcula pesos de muestra para mitigar el desbalanceo de clases.
4. Entrena el modelo con *early stopping* y *reduce-LR-on-plateau*.
5. Evalúa sobre test optimizando el umbral de decisión por clase (F0.5-score).
6. Guarda métricas, gráficas y modelo en `results/` y `saved_model/`.
7. Registra automáticamente parámetros, métricas y artefactos en **MLflow**.

Genera en `results/`:
- `metrics.json` — AUC-ROC, F1, sensibilidad y especificidad por clase (umbral óptimo)
- `metrics_baseline.json` — mismas métricas con umbral fijo 0.5
- `training_history.json` — historial de pérdida y métricas por época
- `plots/` — curvas ROC, matrices de confusión, curvas de aprendizaje

### Explicabilidad (XAI)

Una vez entrenado el modelo, se puede ejecutar el pipeline de interpretabilidad:

```bash
python xai_main.py
```

Genera en `results/xai/`:
- `gradcam/` — mapas de activación temporal por clase (qué tramo del ECG pesa más en la predicción)
- `lead_importance/` — importancia relativa de cada una de las 12 derivaciones
- `shap/` — contribución de las variables clínicas (edad, sexo, altura, peso) mediante SHAP

### Inferencia

```python
import tensorflow as tf
import joblib
import numpy as np
from model.losses import AsymmetricLossPerClass

# Cargar modelo y artefactos entrenados
model = tf.keras.models.load_model(
    "saved_model/best_model.keras",
    custom_objects={"AsymmetricLossPerClass": AsymmetricLossPerClass},
)
scaler = joblib.load("saved_model/scaler.joblib")

# ecg_signal: (1, 1000, 12) — señal ECG normalizada (z-score global)
# metadata:   (1, 4)        — [age, sex, height, weight] escalados con `scaler`
predictions = model.predict({
    "ecg_input": ecg_signal,
    "clinical_input": metadata,
})
# predictions: (1, 5) — probabilidad por superclase [CD, HYP, MI, NORM, STTC]
```

### Seguimiento de experimentos con MLflow

```bash
mlflow ui
# Abre http://localhost:5000
```

> **Nota de compatibilidad:** con versiones recientes de MLflow (≥ 3.x) el backend de archivos local (`./mlruns`) requiere habilitación explícita. Si `main.py` lanza `MlflowException: filesystem tracking backend is in maintenance mode`, exporta la variable de entorno antes de ejecutar:
> ```bash
> export MLFLOW_ALLOW_FILE_STORE=true   # Linux/macOS
> set MLFLOW_ALLOW_FILE_STORE=true      # Windows (cmd)
> ```
> Alternativamente, usa un backend de base de datos: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`.

---

## 📈 Resultados

Los resultados completos de cada versión del modelo se encuentran en `results/`. Resumen de las métricas clave sobre el conjunto de test:
AUC-ROC macro | F1 macro | Sensibilidad macro |
---|---|---|
0.9255 | 0.6865 | 0.9029 |

---

## 🌐 Aplicación desplegada

El directorio `deployment/app` contiene la aplicación Streamlit desplegada en producción (Streamlit Cloud y Hugging Face Spaces), con interfaz de carga de ECG, clasificación y visualización de explicabilidad. Ver [`deployment/app/README.md`](deployment/app/README.md) para su documentación específica.

---

## 🔧 Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| `MlflowException: filesystem tracking backend is in maintenance mode` | Versión reciente de MLflow (≥3.x) | `export MLFLOW_ALLOW_FILE_STORE=true` antes de ejecutar |
| `FileNotFoundError` al cargar `ptbxl_database.csv` | Dataset no descargado o en ruta incorrecta | Verificar que existe `physionet.org/files/ptb-xl/1.0.3/` en la raíz del repo |
| `ValueError: Shape inesperado ... en records100/` | Se están usando registros a 500 Hz en vez de 100 Hz | Asegurarse de usar la carpeta `records100/`, no `records500/` |
| Entrenamiento muy lento en CPU | No se detecta GPU / CUDA no instalado | Instalar CUDA 11.8+ y los drivers correspondientes, o reducir `BATCH_SIZE` |

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📚 Cita

Si utilizas este trabajo, por favor cita el dataset original:

> Wagner, P., Strodthoff, N., Bousseljot, RD. et al. *PTB-XL, a large publicly available electrocardiography dataset.* Sci Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6

---

## 👤 Autor

**Saúl Fernández Rodríguez**
Trabajo de Fin de Máster · 2026