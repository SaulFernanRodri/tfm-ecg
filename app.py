"""
Aplicación Streamlit — Demo de Diagnóstico ECG con XAI.

Muestra:
- Señal ECG coloreada con mapa de calor Grad-CAM (Plotly interactivo)
- Predicciones por clase con barras de probabilidad
- Lead Importance: qué derivación activa más cada diagnóstico
- SHAP clínico: contribución de variables del paciente

Uso:
    cd /home/saul/IA/TFM
    streamlit run Desarrollo/tfm-ecg/app.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import time

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from plotly.subplots import make_subplots

from model.losses import AsymmetricLoss
from xai.gradcam import compute_gradcam_all_classes
from xai.lead_importance import compute_lead_importance_per_class, LEAD_NAMES

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ECG Diagnosis AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LABEL_NAMES   = ["CD", "HYP", "MI", "NORM", "STTC"]
LABEL_FULL    = {
    "CD":   "Trastorno de Conducción",
    "HYP":  "Hipertrofia",
    "MI":   "Infarto de Miocardio",
    "NORM": "ECG Normal",
    "STTC": "Cambios ST/T",
}
LABEL_COLOR   = {
    "CD":   "#e74c3c",
    "HYP":  "#e67e22",
    "MI":   "#c0392b",
    "NORM": "#27ae60",
    "STTC": "#8e44ad",
}
FS            = 100   # Hz
MODEL_PATH    = Path("Desarrollo/tfm-ecg/saved_model/v5/best_model.keras")
STATS_PATH    = Path("Desarrollo/tfm-ecg/saved_model/ecg_global_stats.joblib")
THRESHOLDS_PATH = Path("Desarrollo/tfm-ecg/saved_model/optimal_thresholds.json")
SCALER_PATH   = Path("Desarrollo/tfm-ecg/saved_model/scaler.joblib")
MEDIANS_PATH  = Path("Desarrollo/tfm-ecg/saved_model/train_medians.joblib")

# ---------------------------------------------------------------------------
# Cache de recursos pesados
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo v5…")
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AsymmetricLoss": AsymmetricLoss},
    )
    return model


@st.cache_resource(show_spinner=False)
def load_artifacts():
    stats     = joblib.load(STATS_PATH)      # {"mean": float, "std": float}
    scaler    = joblib.load(SCALER_PATH)
    medians   = joblib.load(MEDIANS_PATH)
    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)            # {"CD": 0.xx, ...}
    return stats, scaler, medians, thresholds


@st.cache_data(show_spinner="Cargando muestras de test…")
def load_test_samples(n: int = 50):
    """Carga las primeras N muestras del test set para la demo."""
    from data.loader import load_dataset
    from data.preprocessor import preprocess_ecg_splits, preprocess_clinical

    train_data, val_data, test_data, label_names = load_dataset()
    train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
        train_data["ecg"], val_data["ecg"], test_data["ecg"]
    )
    _, _, test_clin, _, _ = preprocess_clinical(
        train_data["clinical"], val_data["clinical"], test_data["clinical"]
    )
    return (
        test_ecg[:n],
        test_clin[:n],
        test_data["labels"][:n],
    )


# ---------------------------------------------------------------------------
# Utilidades de preprocesamiento
# ---------------------------------------------------------------------------

def preprocess_ecg_single(ecg_raw: np.ndarray, stats: dict) -> np.ndarray:
    """Normalización global z-score sobre una sola muestra (T, 12)."""
    return ((ecg_raw.astype(np.float32) - stats["mean"]) / (stats["std"] + 1e-8))


def preprocess_clinical_single(
    age: float, sex: int, height: float, weight: float,
    scaler, medians: np.ndarray,
) -> np.ndarray:
    """Preprocesa un vector clínico individual."""
    raw = np.array([[age, sex, height, weight]], dtype=np.float32)
    # Imputar nulos con medianas de train
    for i, val in enumerate(raw[0]):
        if np.isnan(val):
            raw[0, i] = medians[i]
    # Escalar columnas continuas (age, height, weight)
    scaled = raw.copy()
    scaled[:, [0, 2, 3]] = scaler.transform(raw[:, [0, 2, 3]])
    return scaled[0]


# ---------------------------------------------------------------------------
# Visualizaciones Plotly
# ---------------------------------------------------------------------------

def plot_ecg_gradcam(
    ecg: np.ndarray,
    cam: np.ndarray,
    class_name: str,
) -> go.Figure:
    """
    ECG 12-lead con Grad-CAM como heatmap de fondo + señal encima.
    Usa 24 trazas en total (12 Heatmap + 12 Scatter), frente a las ~2400
    trazas de la implementación anterior por segmentos (~30 s de render).
    """
    t = np.arange(ecg.shape[0]) / FS

    fig = make_subplots(
        rows=6, cols=2, shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.06,
        subplot_titles=LEAD_NAMES,
    )

    for i, lead in enumerate(LEAD_NAMES):
        row, col = (i % 6) + 1, (i // 6) + 1
        signal = ecg[:, i]
        sig_min, sig_max = float(signal.min()), float(signal.max())
        margin = (sig_max - sig_min) * 0.3 or 0.5

        # Heatmap de fondo (1 traza por derivación)
        fig.add_trace(
            go.Heatmap(
                z=[cam],
                x=t,
                y=[0],
                colorscale=[[0, "#1a3a5c"], [0.5, "#2563eb"], [1, "#f97316"]],
                zmin=0, zmax=1,
                showscale=(i == 0),
                colorbar=dict(
                    title="CAM", len=0.3, y=0.85,
                    tickfont=dict(color="#fafafa", size=9),
                ) if i == 0 else None,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )
        # Señal ECG (1 traza por derivación)
        fig.add_trace(
            go.Scatter(
                x=t, y=signal,
                mode="lines",
                line=dict(color="#e2e8f0", width=1.2),
                showlegend=False,
                hovertemplate=f"{lead}  t=%{{x:.2f}}s<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.update_yaxes(
            range=[sig_min - margin, sig_max + margin],
            showticklabels=False,
            row=row, col=col,
        )

    fig.update_layout(
        height=700,
        title=dict(
            text=f"ECG + Grad-CAM — Clase: <b>{class_name}</b> ({LABEL_FULL[class_name]})",
            font=dict(size=15, color="#fafafa"),
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=10),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#888", tickfont=dict(size=8))
    fig.add_annotation(
        text="Azul oscuro = baja activación · Naranja = alta activación",
        xref="paper", yref="paper",
        x=0.5, y=-0.04, showarrow=False,
        font=dict(size=11, color="#aaa"),
    )
    return fig


def plot_predictions(probas: np.ndarray, thresholds: dict) -> go.Figure:
    """Barras horizontales de probabilidad con umbral marcado."""
    colors = []
    for i, name in enumerate(LABEL_NAMES):
        thr = thresholds.get(name, 0.5)
        colors.append(LABEL_COLOR[name] if probas[i] >= thr else "#555")

    fig = go.Figure(go.Bar(
        x=probas,
        y=[f"{n} — {LABEL_FULL[n]}" for n in LABEL_NAMES],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probas],
        textposition="outside",
    ))

    # Línea de umbral por clase
    for i, name in enumerate(LABEL_NAMES):
        thr = thresholds.get(name, 0.5)
        fig.add_shape(
            type="line",
            x0=thr, x1=thr,
            y0=i - 0.4, y1=i + 0.4,
            line=dict(color="white", width=2, dash="dash"),
        )

    fig.update_layout(
        height=280,
        xaxis=dict(range=[0, 1.1], title="Probabilidad", color="#aaa"),
        yaxis=dict(color="#ccc"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=12),
        margin=dict(l=10, r=60, t=20, b=40),
        showlegend=False,
    )
    return fig


def plot_lead_importance_plotly(importances: np.ndarray, class_name: str) -> go.Figure:
    """Barras horizontales de lead importance para una clase."""
    sorted_idx = np.argsort(importances)
    vals   = importances[sorted_idx]
    leads  = [LEAD_NAMES[i] for i in sorted_idx]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=leads, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=350,
        title=f"Lead Importance — {class_name}",
        xaxis=dict(title="Caída de prob. al ablar lead", color="#aaa", zeroline=True,
                   zerolinecolor="#555"),
        yaxis=dict(color="#ccc"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=11),
        margin=dict(l=20, r=80, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align:center; color:#e74c3c;'>🫀 ECG Diagnosis AI</h1>
    <p style='text-align:center; color:#aaa; margin-top:-10px;'>
        Diagnóstico multilabel de ECG con explicabilidad · ResNet1D-v5 · PTB-XL
    </p>
    <hr style='border-color:#333;'>
    """, unsafe_allow_html=True)

    # ── Cargar recursos ───────────────────────────────────────────────────────
    model = load_model()
    stats, scaler, medians, thresholds = load_artifacts()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Fuente de datos")
        data_source = st.radio(
            "Origen del ECG",
            ["Muestra del test set", "Subir CSV"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 👤 Datos del paciente")
        age    = st.slider("Edad", 10, 90, 55)
        sex    = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
        sex_v  = 0 if sex == "Hombre" else 1
        height = st.number_input("Altura (cm)", 140, 210, 170)
        weight = st.number_input("Peso (kg)", 40, 150, 75)

        st.markdown("---")
        st.markdown("### 🔬 Análisis XAI")
        run_gradcam  = st.checkbox("Grad-CAM", value=True)
        run_leads    = st.checkbox("Lead Importance", value=True)

        st.markdown("---")
        analyze_btn = st.button("▶ Analizar ECG", use_container_width=True, type="primary")

    # ── Carga del ECG ─────────────────────────────────────────────────────────
    ecg_raw = None

    if data_source == "Muestra del test set":
        sample_idx = st.number_input(
            "Índice de muestra (0–49)", min_value=0, max_value=49, value=0,
            help="Selecciona una muestra del conjunto de test del PTB-XL"
        )
        if st.button("Cargar muestra", use_container_width=False):
            with st.spinner("Cargando test set (primera vez tarda ~30 s)…"):
                test_ecg, test_clin, test_labels = load_test_samples(50)
            st.session_state["ecg_raw"]  = test_ecg[sample_idx]
            st.session_state["clin_raw"] = test_clin[sample_idx]
            st.session_state["true_labels"] = test_labels[sample_idx]
            st.session_state["ecg_ready"] = True

    else:
        uploaded = st.file_uploader(
            "CSV con forma (1000, 12) — una fila por muestra, 12 columnas (leads)",
            type=["csv"],
        )
        if uploaded is not None:
            import pandas as pd
            df = pd.read_csv(uploaded, header=None)
            if df.shape == (1000, 12):
                st.session_state["ecg_raw"]  = preprocess_ecg_single(df.values, stats)
                st.session_state["clin_raw"] = preprocess_clinical_single(
                    age, sex_v, height, weight, scaler, medians
                )
                st.session_state["true_labels"] = None
                st.session_state["ecg_ready"] = True
            else:
                st.error(f"El CSV debe tener forma (1000, 12). Tiene {df.shape}.")

    # ── Análisis ─────────────────────────────────────────────────────────────
    if analyze_btn and st.session_state.get("ecg_ready"):
        ecg  = st.session_state["ecg_raw"]
        clin = st.session_state["clin_raw"]

        # Actualizar clínica con valores del sidebar
        clin = preprocess_clinical_single(age, sex_v, height, weight, scaler, medians)

        # Predicción
        with st.spinner("Ejecutando modelo…"):
            ecg_t  = tf.convert_to_tensor(ecg[np.newaxis], dtype=tf.float32)
            clin_t = tf.convert_to_tensor(clin[np.newaxis], dtype=tf.float32)
            probas = model([ecg_t, clin_t], training=False).numpy()[0]

        # Etiquetas predichas
        predicted = [
            LABEL_NAMES[i] for i, p in enumerate(probas)
            if p >= thresholds.get(LABEL_NAMES[i], 0.5)
        ]

        # ── Panel de resultados ───────────────────────────────────────────────
        st.markdown("## 📊 Resultados")

        # Métricas rápidas en columnas
        cols = st.columns(len(LABEL_NAMES))
        for i, (col, name) in enumerate(zip(cols, LABEL_NAMES)):
            p    = probas[i]
            thr  = thresholds.get(name, 0.5)
            flag = "🔴" if p >= thr else "⚪"
            col.metric(
                label=f"{flag} {name}",
                value=f"{p:.1%}",
                delta=f"umbral {thr:.2f}",
                delta_color="off",
            )

        if predicted:
            st.error(f"⚠️ **Diagnóstico(s) detectado(s):** {' · '.join(predicted)}")
        else:
            st.success("✅ No se detectan patologías por encima del umbral")

        if st.session_state.get("true_labels") is not None:
            true = st.session_state["true_labels"]
            true_names = [LABEL_NAMES[i] for i, v in enumerate(true) if v == 1]
            st.info(f"🏷️ **Etiqueta real:** {' · '.join(true_names) if true_names else 'NORM'}")

        # Gráfico de probabilidades
        st.plotly_chart(plot_predictions(probas, thresholds), use_container_width=True)

        # ── Tabs de XAI ──────────────────────────────────────────────────────
        tab1, tab2 = st.tabs(["🌡️ Grad-CAM", "📡 Lead Importance"])

        # Clases a analizar: las detectadas (sobre umbral), o la de mayor prob si ninguna
        detected_classes = [
            LABEL_NAMES[i] for i, p in enumerate(probas)
            if p >= thresholds.get(LABEL_NAMES[i], 0.5)
        ]
        analysis_classes = detected_classes if detected_classes else [LABEL_NAMES[int(np.argmax(probas))]]

        with tab1:
            if run_gradcam:
                from xai.gradcam import compute_gradcam
                if not detected_classes:
                    st.info("Ninguna clase supera el umbral — mostrando la clase con mayor probabilidad.")

                for cam_class in analysis_classes:
                    with st.spinner(f"Calculando Grad-CAM para {cam_class}…"):
                        cam = compute_gradcam(model, ecg, clin, LABEL_NAMES.index(cam_class))
                    st.plotly_chart(
                        plot_ecg_gradcam(ecg, cam, cam_class),
                        use_container_width=True,
                    )
                    st.caption(
                        f"Segmentos **rojos**: activaron la predicción de **{cam_class} — {LABEL_FULL[cam_class]}**. "
                        "Segmentos **azules**: baja influencia."
                    )
            else:
                st.info("Activa 'Grad-CAM' en el sidebar para ver el mapa de calor.")

        with tab2:
            if run_leads:
                from xai.lead_importance import compute_lead_importance_single
                if not detected_classes:
                    st.info("Ninguna clase supera el umbral — mostrando la clase con mayor probabilidad.")

                for cam_class in analysis_classes:
                    with st.spinner(f"Calculando lead importance para {cam_class}…"):
                        importances = compute_lead_importance_single(
                            model, ecg, clin, class_idx=LABEL_NAMES.index(cam_class)
                        )
                    st.plotly_chart(
                        plot_lead_importance_plotly(importances, cam_class),
                        use_container_width=True,
                    )
                    st.caption(
                        "Barras **rojas**: ablar ese lead reduce la probabilidad (lead relevante). "
                        "Barras **azules**: poco informativo para este diagnóstico."
                    )
            else:
                st.info("Activa 'Lead Importance' en el sidebar.")

    elif not st.session_state.get("ecg_ready"):
        # Estado inicial: instrucciones
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### Cómo usar la demo

            1. **Selecciona el origen del ECG** en el sidebar:
               - *Test set*: muestras reales del PTB-XL (primera carga ~30 s)
               - *Subir CSV*: tu propio ECG en formato 1000×12

            2. **Ajusta los datos del paciente** (edad, sexo, altura, peso)

            3. **Elige la clase** para analizar con Grad-CAM

            4. Pulsa **▶ Analizar ECG**

            ---
            **Modelo:** ResNet1D-5 bloques + SE Attention + ASL  
            **Dataset:** PTB-XL v1.0.3 · 21,837 ECGs  
            **AUC macro test:** 0.9255 · Sensibilidad: 0.9463
            """)


if __name__ == "__main__":
    # Inicializar session state
    if "ecg_ready" not in st.session_state:
        st.session_state["ecg_ready"] = False
    main()
