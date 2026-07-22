import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if not (_ROOT / "saved_model").exists():
    # Si no existe, estamos en local (deployment/app/pages)
    _ROOT = Path(__file__).resolve().parent.parent.parent.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# ── Descarga de artefactos si no están disponibles (ej: Streamlit Cloud) ──────
def _ensure_artifacts() -> None:
    """Descarga modelos y datos de demo desde HF Space si no existen en local."""
    if not (_ROOT / "saved_model" / "v5" / "best_model.keras").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="SaulFernanRodri/ecg-diagnosis-ai",
            repo_type="space",
            allow_patterns=["saved_model/**", "demo_data/**"],
            local_dir=str(_ROOT),
        )

_ensure_artifacts()

import json
import time
import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from plotly.subplots import make_subplots

from model.losses import AsymmetricLoss
from xai.gradcam import compute_gradcam_all_classes
from xai.lead_importance import compute_lead_importance_per_class

from utils.ui import (
    FS, LABEL_NAMES, LABEL_FULL, LABEL_SEVERITY,
    inject_css, st_blue_alert, render_diagnosis_badges,
    plot_predictions, plot_ecg_gradcam, plot_ecg_gradcam_single,
    plot_clinical_influence, preprocess_ecg_single, preprocess_clinical_single,
)

# ---------------------------------------------------------------------------
# CSS institucional compartido
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# Rutas de artefactos
# ---------------------------------------------------------------------------
MODEL_PATH    = _ROOT / "saved_model/v5/best_model.keras"
STATS_PATH    = _ROOT / "saved_model/ecg_global_stats.joblib"
_V62_THRESHOLDS = _ROOT / "saved_model/v6.2/optimal_thresholds.json"
_V5_THRESHOLDS  = _ROOT / "saved_model/v5/optimal_thresholds.json"
THRESHOLDS_PATH = _V62_THRESHOLDS if _V62_THRESHOLDS.exists() else _V5_THRESHOLDS
SCALER_PATH   = _ROOT / "saved_model/scaler.joblib"
MEDIANS_PATH  = _ROOT / "saved_model/train_medians.joblib"

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
    stats     = joblib.load(STATS_PATH)
    scaler    = joblib.load(SCALER_PATH)
    medians   = joblib.load(MEDIANS_PATH)
    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    return stats, scaler, medians, thresholds

@st.cache_data(show_spinner="Cargando muestras preprocesadas…")
def load_test_samples(n: int = 50):
    demo_dir = _ROOT / "demo_data"
    if not demo_dir.exists():
        demo_dir = _ROOT / "deployment/app/demo_data"
        
    test_ecg = np.load(demo_dir / "ecg_samples.npy")
    test_clin = np.load(demo_dir / "clin_samples.npy")
    test_labels = np.load(demo_dir / "true_labels.npy")
    
    return test_ecg[:n], test_clin[:n], test_labels[:n]

# ---------------------------------------------------------------------------
# Layout principal de Simulador (Modern Layout sin Sidebar)
# ---------------------------------------------------------------------------
st.markdown("""
<h2 style='text-align:center; color:#0f4c81; font-family:sans-serif;'>Sistema de Apoyo a la Decisión Clínica (CDSS)</h2>
<p style='text-align:center; color:#555;'>Inferencia con Datos Externos (CSV)</p>
<hr style='border-color:#ccc;'>
""", unsafe_allow_html=True)

model = load_model()
stats, scaler, medians, thresholds = load_artifacts()

# Variables para guardar estado
if "ecg_ready" not in st.session_state:
    st.session_state["ecg_ready"] = False

ecg_raw = None

# TOP SECTION: Inputs limpios estructurados en el área principal
st.markdown("<h4 style='color:#0f4c81;'>Panel de Datos Clínicos</h4>", unsafe_allow_html=True)

container_inputs = st.container()
with container_inputs:
    col_origen, col_bio1, col_bio2 = st.columns([1.5, 1, 1], gap="medium")
    
    with col_origen:
        st.markdown("**Subir Registro Electrocardiográfico**")
        uploaded = st.file_uploader("Archivo CSV (1000x12)", type=["csv"])
        if uploaded is not None:
            import pandas as pd
            df = pd.read_csv(uploaded, header=None)
            if df.shape == (1000, 12):
                st.session_state["csv_df"] = df.values
            else:
                st_blue_alert(f"Error: El CSV debe tener forma (1000, 12). Se detectó {df.shape}.")

    with col_bio1:
        st.markdown("**Filiación y Biometría**")
        age = st.number_input("Edad (años)", min_value=10, max_value=110, value=55, step=None)
        height = st.number_input("Altura (cm)", min_value=120, max_value=220, value=170, step=None)
        
    with col_bio2:
        st.markdown("<br>", unsafe_allow_html=True)
        sex = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
        sex_v = 0 if sex == "Hombre" else 1
        weight = st.number_input("Peso (kg)", min_value=30, max_value=200, value=75, step=None)

st.markdown("---")

col_run, col_xai1, col_xai3 = st.columns([1.5, 1, 1])
with col_run:
    analyze_btn = st.button("Procesar ECG (Inferencia)", use_container_width=True, type="primary")

with col_xai1:
    run_gradcam = st.checkbox("Módulo Temporal (Grad-CAM)", value=True)
with col_xai3:
    run_clinical = st.checkbox("Módulo Contrafactual (Clínica)", value=True)

st.markdown("---")

# Procesar CSV si fue subido y se le da al botón
if analyze_btn and "csv_df" in st.session_state:
    st.session_state["ecg_raw"]  = preprocess_ecg_single(st.session_state["csv_df"], stats)
    st.session_state["clin_raw"] = preprocess_clinical_single(age, sex_v, height, weight, scaler, medians)
    st.session_state["true_labels"] = None
    st.session_state["ecg_ready"] = True

if analyze_btn and st.session_state.get("ecg_ready"):
    ecg  = st.session_state["ecg_raw"]
    clin = preprocess_clinical_single(age, sex_v, height, weight, scaler, medians)

    with st.spinner("Ejecutando red neuronal ResNet1D-5 y MLP…"):
        ecg_t  = tf.convert_to_tensor(ecg[np.newaxis], dtype=tf.float32)
        clin_t = tf.convert_to_tensor(clin[np.newaxis], dtype=tf.float32)
        probas = model([ecg_t, clin_t], training=False).numpy()[0]

    predicted_raw = [LABEL_NAMES[i] for i, p in enumerate(probas) if p >= thresholds.get(LABEL_NAMES[i], 0.5)]

    PATHO_CLASSES  = [n for n in LABEL_NAMES if n != "NORM"]
    norm_idx_      = LABEL_NAMES.index("NORM")
    patho_above_thr = any(probas[LABEL_NAMES.index(n)] >= thresholds.get(n, 0.5) for n in PATHO_CLASSES)
    norm_is_absolute_top = (int(np.argmax(probas)) == norm_idx_)
    
    if patho_above_thr and "NORM" in predicted_raw and not norm_is_absolute_top:
        predicted = [n for n in predicted_raw if n != "NORM"]
    else:
        predicted = predicted_raw

    st.markdown("<h3 style='color:#0f4c81;'>Resultados de Inferencia Diagnóstica</h3>", unsafe_allow_html=True)
    
    top_idx   = int(np.argmax(probas))
    top_class = LABEL_NAMES[top_idx]
    norm_idx  = LABEL_NAMES.index("NORM")
    norm_proba = float(probas[norm_idx])
    norm_thr   = thresholds.get("NORM", 0.5)
    norm_is_top = (top_class == "NORM")

    st.caption("Nota Metodológica: Las puntuaciones representan la probabilidad predicha por el modelo, integrando ramas temporal y tabular.")

    render_diagnosis_badges(probas, thresholds)

    # Reemplazo de st.success / st.error por st_blue_alert
    if norm_is_top and norm_proba >= norm_thr:
        st_blue_alert(f"Diagnóstico Principal Sugerido: ECG Normal (Nivel de Confianza: {norm_proba:.2f})", is_bold=True)
        other_detected = [n for n in predicted if n != "NORM"]
        if other_detected:
            st_blue_alert(f"Hallazgos secundarios por encima del umbral clínico: {' · '.join(other_detected)}")
    elif predicted:
        pathos = [n for n in predicted if n != "NORM"]
        norm_detected = "NORM" in predicted
        if pathos:
            st_blue_alert(f"Diagnóstico(s) Patológico(s) Sugerido(s): {' · '.join(pathos)}" + ("  |  (Se detectan patrones residuales compatibles con ECG Normal)" if norm_detected else ""), is_bold=True)
        else:
            st_blue_alert(f"Diagnóstico Principal Sugerido: ECG Normal (Nivel de Confianza: {norm_proba:.2f})", is_bold=True)
    else:
        st_blue_alert("No se detectan hallazgos patológicos por encima del umbral de decisión clínico establecido.")

    if st.session_state.get("true_labels") is not None:
        true = st.session_state["true_labels"]
        true_names = [LABEL_NAMES[i] for i, v in enumerate(true) if v == 1]
        st_blue_alert(f"Diagnóstico Confirmado (Etiqueta Real PTB-XL): {' · '.join(true_names) if true_names else 'NORM'}")

    st.plotly_chart(plot_predictions(probas, thresholds), use_container_width=True)

    

    # ── BOTTOM SECTION: Tabs de XAI ──────────────────────────────────────
    st.markdown("<h3 style='color:#0f4c81;'>Módulos de Interpretabilidad Espacio-Temporal (XAI)</h3>", unsafe_allow_html=True)
    tab1, tab3 = st.tabs(["Localización Temporal (Grad-CAM)", "Análisis Contrafactual Clínico"])

    detected_classes = [LABEL_NAMES[i] for i, p in enumerate(probas) if p >= thresholds.get(LABEL_NAMES[i], 0.5)]
    if norm_is_top:
        analysis_classes = ["NORM"] + [c for c in detected_classes if c != "NORM"]
        if not analysis_classes: analysis_classes = ["NORM"]
    else:
        analysis_classes = detected_classes if detected_classes else [LABEL_NAMES[int(np.argmax(probas))]]

    with tab1:
        if run_gradcam:
            from xai.gradcam import compute_gradcam
            if not detected_classes: st_blue_alert("Visualizando la clase con mayor probabilidad.")

            if len(analysis_classes) > 1:
                cam_class = st.radio("Seleccione patología para inspeccionar mapa temporal Grad-CAM:", analysis_classes, format_func=lambda n: f"{n} — {LABEL_FULL[n]}", horizontal=True, key="gradcam_class_selector")
            else:
                cam_class = analysis_classes[0]

            with st.spinner(f"Calculando Grad-CAM para {cam_class}…"):
                cam = compute_gradcam(model, ecg, clin, LABEL_NAMES.index(cam_class))
            st.plotly_chart(plot_ecg_gradcam(ecg, cam, cam_class), use_container_width=True)
            st.caption(f"Segmentos **azul oscuro**: activaron la predicción de **{cam_class}**. Segmentos **blancos/celestes**: baja influencia.")
        else:
            st.caption("Módulo inactivo.")

    with tab3:
        if run_clinical:
            from xai.clinical_ablation import compute_clinical_influence
            if not detected_classes: st_blue_alert("Visualizando la clase con mayor probabilidad.")

            for cls in analysis_classes:
                with st.spinner(f"Calculando influencia clínica para {cls}…"):
                    infl = compute_clinical_influence(model, ecg, clin, class_idx=LABEL_NAMES.index(cls))
                st.plotly_chart(plot_clinical_influence(infl, cls), use_container_width=True)
            st.caption(
                "Muestra el impacto marginal de la variable biométrica sobre la probabilidad diagnóstica. "
                "Barras **azul oscuro**: la variable incrementó el riesgo. Barras **grises**: bajo impacto o reducción de riesgo.  \n"
                "_Nota Metodológica (Análisis Contrafactual): Para la variable Sexo, se computa la inferencia evaluando qué sucedería en las probabilidades si el paciente "
                "perteneciese a la categoría biológica opuesta._"
            )
        else:
            st.caption("Módulo inactivo.")

elif not st.session_state.get("ecg_ready"):
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st_blue_alert("Por favor, introduzca los datos clínicos y pulse en 'Procesar ECG' para inicializar el CDSS.")
