---
title: ECG Diagnosis AI
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🫀 ECG Diagnosis AI

**Sistema de Apoyo a la Decisión Clínica (CDSS)** para el diagnóstico electrocardiográfico multimodal mediante Deep Learning, con módulos integrados de explicabilidad (XAI).

Esta aplicación es el componente de **despliegue** del proyecto [tfm-ecg](../../README.md) (Trabajo de Fin de Máster). Aquí se documenta específicamente cómo ejecutar, configurar y usar la demo — para el entrenamiento del modelo, ver el README raíz del repositorio.

🔗 **Demo en vivo:**
[Streamlit Cloud](https://tfm-ecg-wt5ccz6byxv3uticy4gq66.streamlit.app/) · [Hugging Face Spaces](https://huggingface.co/spaces/SaulFernanRodri/ecg-diagnosis-ai)

> ⚠️ **Aviso importante:** este sistema es un prototipo de investigación académica. **No debe usarse para diagnóstico médico real** ni como sustituto del criterio clínico profesional.

---

## 📑 Tabla de contenidos

- [Modelo](#-modelo)
- [Páginas de la aplicación](#-páginas-de-la-aplicación)
- [Explicabilidad (XAI)](#-explicabilidad-xai)
- [Cómo usar la demo](#-cómo-usar-la-demo)
- [Ejecución local](#-ejecución-local)
- [Despliegue con Docker](#-despliegue-con-docker)
- [Despliegue en Hugging Face Spaces](#-despliegue-en-hugging-face-spaces)
- [Estructura del directorio](#-estructura-del-directorio)
- [Formato de datos de entrada](#-formato-de-datos-de-entrada)
- [Aviso legal](#-aviso-legal)
- [Créditos](#-créditos)

---

## 🧠 Modelo

**ResNet1D-v5** entrenado sobre el dataset público [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) (21.837 registros ECG a 100 Hz).

**Arquitectura:** 5 bloques ResNet1D con atención Squeeze-and-Excitation (SE) para la señal, fusionados en la etapa final con un MLP que procesa variables clínicas (edad, sexo, altura, peso). Ver el diagrama completo en el [README raíz](../../README.md#-arquitectura-del-modelo).

| Métrica | Valor |
|---|---|
| AUC-ROC macro | 0.9255 |
| Sensibilidad macro | 0.9463 |
| F1 macro | 0.8025 |

### Clases detectadas

| Código | Descripción |
|---|---|
| `MI` | Infarto de miocardio |
| `CD` | Trastorno de conducción |
| `HYP` | Hipertrofia |
| `STTC` | Cambios ST/T |
| `NORM` | ECG normal |

La salida es **multilabel**: un mismo ECG puede activar varias superclases a la vez (p. ej. `MI` + `CD`).

---

## 🖥 Páginas de la aplicación

La app está estructurada en tres páginas (`st.navigation`), definidas en `app.py`:

| Página | Archivo | Descripción |
|---|---|---|
| **Panel Institucional** | `pages/1_inicio.py` | Ficha técnica del dataset, arquitectura del modelo y contexto clínico del proyecto |
| **Test Simulador (Demo)** | `pages/2_test_simulador.py` | Explora 50 muestras reales pre-exportadas del conjunto de test de PTB-XL, con etiqueta real vs. predicción |
| **Simulador Clínico (CSV)** | `pages/3_simulador_csv.py` | Sube tu propio registro ECG en CSV y ajusta manualmente los datos biométricos del paciente |

---

## 🔍 Explicabilidad (XAI)

Cada predicción viene acompañada de visualizaciones que explican **por qué** el modelo llegó a esa conclusión:

- **Grad-CAM 1D** — resalta qué tramo temporal de la señal ECG activó más la predicción (mapa de calor azul → rojo sobre la traza).
- **Lead Importance** — indica qué derivación (de las 12) contribuyó más al diagnóstico, mediante ablación por canal.

---

## ▶️ Cómo usar la demo

1. Abre la app (local, Docker o el enlace desplegado).
2. Ve a **Test Simulador (Demo)** para explorar casos reales de PTB-XL, o a **Simulador Clínico (CSV)** para subir tu propio ECG.
3. Si subes un CSV propio, asegúrate de que cumple el [formato de entrada](#-formato-de-datos-de-entrada).
4. Ajusta los datos del paciente (edad, sexo, altura, peso) en el panel lateral.
5. Pulsa **▶ Analizar ECG** para obtener el diagnóstico probabilístico por clase junto con las visualizaciones de Grad-CAM y Lead Importance.

---

## 💻 Ejecución local

```bash
# Desde deployment/app/
cd deployment/app

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Instalar dependencias (versión CPU, ligera para despliegue)
pip install -r requirements.txt

# Lanzar la app
streamlit run app.py
```

La app quedará disponible en `http://localhost:8501`.

> Los pesos del modelo y el escalador de variables clínicas deben estar disponibles en las rutas que espera `app.py` (por defecto, dentro del propio directorio de la app o referenciados desde `saved_model/` en la raíz del proyecto). Si no los tienes, entrena primero el modelo siguiendo el [README raíz](../../README.md#-entrenamiento-y-evaluación).

---

## 🐳 Despliegue con Docker

```bash
# Desde deployment/app/
docker build -t ecg-diagnosis-ai .
docker run -p 7860:7860 ecg-diagnosis-ai
```

La app quedará disponible en `http://localhost:7860` (mismo puerto que usa Hugging Face Spaces).

---

## 🤗 Despliegue en Hugging Face Spaces

Este directorio está preparado para desplegarse directamente como un **Space de tipo Docker**:

- El bloque YAML al inicio de este archivo (`title`, `sdk: docker`, `app_port: 7860`, etc.) es la configuración que Hugging Face Spaces lee automáticamente.
- El `Dockerfile` instala las dependencias, copia el código y lanza Streamlit en modo *headless* sobre el puerto `7860`.
- Los scripts de `deployment/scripts/` (`export_demo_data.py`, `upload_to_hf.py`) automatizan la exportación de las muestras demo y la subida del Space.

Para desplegar tu propia copia: crea un Space en Hugging Face con SDK "Docker", sube el contenido de `deployment/app/` (incluido este README con su cabecera YAML) y el build se disparará automáticamente.

---

## 📂 Estructura del directorio

```text
deployment/
├── app/
│   ├── app.py                    # Punto de entrada Streamlit (navegación entre páginas)
│   ├── Dockerfile                # Imagen para Hugging Face Spaces / despliegue local
│   ├── requirements.txt          # Dependencias (TensorFlow CPU, Streamlit, Plotly...)
│   ├── README.md                 # Este archivo
│   ├── demo_data/
│   │   ├── ecg_samples.npy       # 50 señales ECG pre-exportadas del test set
│   │   ├── clin_samples.npy      # Metadatos clínicos correspondientes
│   │   └── true_labels.npy       # Etiquetas reales para comparar contra la predicción
│   └── pages/
│       ├── 1_inicio.py           # Panel institucional
│       ├── 2_test_simulador.py   # Simulador con muestras demo
│       └── 3_simulador_csv.py    # Simulador con CSV subido por el usuario
└── scripts/
    ├── export_demo_data.py       # Genera los .npy de demo_data/ a partir de PTB-XL
    ├── upload_to_hf.py           # Sube el Space a Hugging Face
    ├── fix_tabs.py                # Utilidad de mantenimiento del código de las páginas
    └── refactor_pages.py          # Utilidad de mantenimiento del código de las páginas
```

---

## 📐 Formato de datos de entrada

Para usar el **Simulador Clínico (CSV)**, el archivo subido debe cumplir:

| Requisito | Valor |
|---|---|
| Formato | CSV, **sin cabecera** |
| Forma | `1000 filas × 12 columnas` |
| Filas | Muestras temporales (10 s a 100 Hz) |
| Columnas | Una por derivación, en el orden estándar de 12 derivaciones |
| Unidades | mV (igual que los registros PTB-XL a 100 Hz) |

Si el CSV no tiene exactamente la forma `(1000, 12)`, la aplicación mostrará un error y no permitirá continuar.

Adicionalmente, debes indicar en el panel lateral los datos biométricos del paciente: **edad, sexo, altura y peso**, usados por la rama tabular del modelo.

---

## ⚖️ Aviso legal

> Este sistema es un **prototipo de investigación académica** desarrollado en el marco de un Trabajo de Fin de Máster. Los resultados no han sido validados clínicamente ni aprobados por ninguna autoridad sanitaria. **No debe utilizarse para tomar decisiones de diagnóstico o tratamiento médico real.** Cualquier uso clínico requiere validación, supervisión y aprobación regulatoria adecuadas.

---

## 🙏 Créditos

- **Dataset:** [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) (Wagner et al., 2020), licencia CC BY 4.0.
- **Proyecto:** Trabajo de Fin de Máster — ver [README principal del repositorio](../../README.md) para la documentación completa del entrenamiento, evaluación y explicabilidad.