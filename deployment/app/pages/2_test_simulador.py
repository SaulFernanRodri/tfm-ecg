import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if not (_ROOT / "saved_model").exists():
    # Si no existe, estamos en local (deployment/app/pages)
    _ROOT = Path(__file__).resolve().parent.parent.parent.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
from xai.lead_importance import compute_lead_importance_per_class, LEAD_NAMES

# ---------------------------------------------------------------------------
# Inyección de CSS (Impresión y UI Limpia)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Ocultar flechas numéricas (spinners) para inputs más limpios */
input[type=number]::-webkit-inner-spin-button, 
input[type=number]::-webkit-outer-spin-button { 
    -webkit-appearance: none; 
    margin: 0; 
}
input[type=number] {
    -moz-appearance: textfield;
}

/* Ocultar botones +/- propios de Streamlit */
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constantes Estéticas y Técnicas
# ---------------------------------------------------------------------------
LABEL_NAMES   = ["CD", "HYP", "MI", "NORM", "STTC"]
LABEL_FULL    = {
    "CD":   "Trastorno de Conducción",
    "HYP":  "Hipertrofia",
    "MI":   "Infarto de Miocardio",
    "NORM": "ECG Normal",
    "STTC": "Cambios ST/T",
}
# Todo en azul corporativo #0f4c81
LABEL_COLOR   = {
    "CD":   "#0f4c81",
    "HYP":  "#0f4c81",
    "MI":   "#0f4c81",
    "NORM": "#0f4c81",
    "STTC": "#0f4c81",
}
FS            = 100   # Hz
MODEL_PATH    = _ROOT / "saved_model/v5/best_model.keras"
STATS_PATH    = _ROOT / "saved_model/ecg_global_stats.joblib"
_V61_THRESHOLDS = _ROOT / "saved_model/v6.1/optimal_thresholds.json"
_V5_THRESHOLDS  = _ROOT / "saved_model/v5/optimal_thresholds.json"
THRESHOLDS_PATH = _V61_THRESHOLDS if _V61_THRESHOLDS.exists() else _V5_THRESHOLDS
SCALER_PATH   = _ROOT / "saved_model/scaler.joblib"
MEDIANS_PATH  = _ROOT / "saved_model/train_medians.joblib"

# ---------------------------------------------------------------------------
# Funciones UI Institucionales (Blue Theme)
# ---------------------------------------------------------------------------
def st_blue_alert(message: str, is_bold: bool = False):
    fw = "bold" if is_bold else "normal"
    st.markdown(f"""
    <div style="background-color: #f0f4f8; border-left: 4px solid #0f4c81; color: #0f4c81; padding: 12px; border-radius: 4px; margin-bottom: 1rem; font-family: sans-serif; font-weight: {fw};">
        {message}
    </div>
    """, unsafe_allow_html=True)

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
# Utilidades de preprocesamiento
# ---------------------------------------------------------------------------
def preprocess_ecg_single(ecg_raw: np.ndarray, stats: dict) -> np.ndarray:
    return ((ecg_raw.astype(np.float32) - stats["mean"]) / (stats["std"] + 1e-8))

def preprocess_clinical_single(
    age: float, sex: int, height: float, weight: float,
    scaler, medians: np.ndarray,
) -> np.ndarray:
    raw = np.array([[age, sex, height, weight]], dtype=np.float32)
    for i, val in enumerate(raw[0]):
        if np.isnan(val):
            raw[0, i] = medians[i]
    scaled = raw.copy()
    scaled[:, [0, 2, 3]] = scaler.transform(raw[:, [0, 2, 3]])
    return scaled[0]

# ---------------------------------------------------------------------------
# Visualizaciones Plotly (Blue Theme)
# ---------------------------------------------------------------------------
def plot_ecg_gradcam(ecg: np.ndarray, cam: np.ndarray, class_name: str) -> go.Figure:
    t = np.arange(ecg.shape[0]) / FS
    fig = make_subplots(
        rows=6, cols=2, shared_xaxes=True,
        vertical_spacing=0.04, horizontal_spacing=0.06,
        subplot_titles=LEAD_NAMES,
    )
    for i, lead in enumerate(LEAD_NAMES):
        row, col = (i % 6) + 1, (i // 6) + 1
        signal = ecg[:, i]
        sig_min, sig_max = float(signal.min()), float(signal.max())
        margin = (sig_max - sig_min) * 0.3 or 0.5

        fig.add_trace(
            go.Heatmap(
                z=[cam], x=t, y=[0],
                colorscale=[[0, "#ffffff"], [0.5, "#90caf9"], [1, "#0f4c81"]],
                zmin=0, zmax=1,
                showscale=(i == 0),
                colorbar=dict(
                    title="CAM", len=0.3, y=0.85,
                    tickfont=dict(color="#0f4c81", size=9),
                ) if i == 0 else None,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=signal, mode="lines",
                line=dict(color="#1a237e", width=1.2),
                showlegend=False, hovertemplate=f"{lead}  t=%{{x:.2f}}s<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.update_yaxes(range=[sig_min - margin, sig_max + margin], showticklabels=False, row=row, col=col)

    fig.update_layout(
        height=700,
        title=dict(
            text=f"ECG + Grad-CAM — Clase: <b>{class_name}</b> ({LABEL_FULL[class_name]})",
            font=dict(size=15, color="#0f4c81"),
        ),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=10),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#555", tickfont=dict(size=8))
    fig.add_annotation(
        text="Blanco/Celeste = baja activación · Azul Oscuro = alta activación",
        xref="paper", yref="paper", x=0.5, y=-0.04, showarrow=False,
        font=dict(size=11, color="#555"),
    )
    return fig

def plot_ecg_gradcam_single(ecg: np.ndarray, cam: np.ndarray, class_name: str, lead_idx: int) -> go.Figure:
    lead = LEAD_NAMES[lead_idx]
    t = np.arange(ecg.shape[0]) / FS
    signal = ecg[:, lead_idx]
    sig_min, sig_max = float(signal.min()), float(signal.max())
    margin = (sig_max - sig_min) * 0.3 or 0.5

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[cam], x=t, y=[0],
        colorscale=[[0, "#ffffff"], [0.5, "#90caf9"], [1, "#0f4c81"]],
        zmin=0, zmax=1, showscale=True,
        colorbar=dict(title="CAM", len=0.7, thickness=14, tickfont=dict(color="#0f4c81", size=10), titlefont=dict(color="#0f4c81", size=11)),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines", line=dict(color="#0f4c81", width=1.8),
        showlegend=False, hovertemplate=f"t=%{{x:.2f}}s  val=%{{y:.3f}}<extra>{lead}</extra>",
    ))
    fig.update_layout(
        height=280,
        title=dict(
            text=f"ECG + Grad-CAM — <b>{lead}</b> · <b>{class_name}</b>: {LABEL_FULL[class_name]}",
            font=dict(size=14, color="#0f4c81"),
        ),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=11), margin=dict(l=40, r=70, t=55, b=45),
        xaxis=dict(title="Tiempo (s)", showgrid=False, zeroline=False, color="#555"),
        yaxis=dict(range=[sig_min - margin, sig_max + margin], showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig

def plot_predictions(probas: np.ndarray, thresholds: dict) -> go.Figure:
    order = np.argsort(probas)
    sorted_names  = [LABEL_NAMES[i] for i in order]
    sorted_probas = probas[order]
    colors = [LABEL_COLOR[name] if p >= thresholds.get(name, 0.5) else "#999999" for name, p in zip(sorted_names, sorted_probas)]

    fig = go.Figure(go.Bar(
        x=sorted_probas, y=[f"{n} — {LABEL_FULL[n]}" for n in sorted_names],
        orientation="h", marker_color=colors, text=[f"{p:.2f}" for p in sorted_probas], textposition="outside",
    ))
    for yi, name in enumerate(sorted_names):
        thr = thresholds.get(name, 0.5)
        fig.add_shape(type="line", x0=thr, x1=thr, y0=yi - 0.4, y1=yi + 0.4, line=dict(color="#0f4c81", width=2, dash="dash"))
    fig.update_layout(
        height=300, xaxis=dict(range=[0, 1.15], title="Puntuación del modelo (0–1)", color="#555"),
        yaxis=dict(color="#0f4c81"), paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=12), margin=dict(l=10, r=70, t=20, b=40), showlegend=False,
    )
    return fig


def plot_clinical_influence(influences: np.ndarray, class_name: str) -> go.Figure:
    influences_pct = influences * 100
    sorted_idx = np.argsort(influences_pct)
    vals   = influences_pct[sorted_idx]
    CLINICAL_FEATURE_NAMES = ["Edad", "Sexo", "Altura", "Peso"]
    feats  = [CLINICAL_FEATURE_NAMES[i] for i in sorted_idx]
    colors = ["#0f4c81" if v > 0 else "#999999" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h", marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        height=280, title=dict(text=f"Contribución Clínica al Riesgo — {class_name}", font=dict(color="#0f4c81")),
        xaxis=dict(title="Impacto en la probabilidad (%)", color="#555", zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(color="#0f4c81"), paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=12), margin=dict(l=20, r=80, t=40, b=40),
    )
    return fig

# ---------------------------------------------------------------------------
# Layout principal de Simulador (Modern Layout sin Sidebar)
# ---------------------------------------------------------------------------
st.markdown("""
<h2 style='text-align:center; color:#0f4c81; font-family:sans-serif;'>Sistema de Apoyo a la Decisión Clínica (CDSS)</h2>
<p style='text-align:center; color:#555;'>Test de Validación (Muestras PTB-XL)</p>
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
        st.markdown("**Origen del Electrocardiograma**")
        if True:
            sample_idx = st.number_input("ID de paciente demo (0-49)", min_value=0, max_value=49, value=0, step=None)
            if st.button("Cargar Muestra Demo", use_container_width=True):
                with st.spinner("Cargando registro…"):
                    test_ecg, test_clin, test_labels = load_test_samples(50)
                st.session_state["ecg_raw"]  = test_ecg[sample_idx]
                st.session_state["clin_raw"] = test_clin[sample_idx]
                st.session_state["true_labels"] = test_labels[sample_idx]
                st.session_state["ecg_ready"] = True
        

    with col_bio1:
        st.markdown("**Filiación y Biometría**")
        if True:
            if st.session_state.get("ecg_ready") and "clin_raw" in st.session_state:
                c_data = st.session_state["clin_raw"]
                unscaled = scaler.inverse_transform([[c_data[0], c_data[2], c_data[3]]])[0]
                st.info(f"**Edad:** {int(round(unscaled[0]))} años")
                st.info(f"**Altura:** {int(round(unscaled[1]))} cm")
            else:
                st.caption("Cargue una muestra para ver los datos biométricos.")
        
        
    with col_bio2:
        st.markdown("<br>", unsafe_allow_html=True)
        if True:
            if st.session_state.get("ecg_ready") and "clin_raw" in st.session_state:
                c_data = st.session_state["clin_raw"]
                unscaled = scaler.inverse_transform([[c_data[0], c_data[2], c_data[3]]])[0]
                sex_str = "Mujer" if c_data[1] == 1 else "Hombre"
                st.info(f"**Sexo:** {sex_str}")
                st.info(f"**Peso:** {int(round(unscaled[2]))} kg")
        

st.markdown("---")

col_run, col_xai1, col_xai3 = st.columns([1.5, 1, 1])
with col_run:
    analyze_btn = st.button("Procesar ECG (Inferencia)", use_container_width=True, type="primary")

with col_xai1:
    run_gradcam = st.checkbox("Módulo Temporal (Grad-CAM)", value=True)
with col_xai3:
    run_clinical = st.checkbox("Módulo Contrafactual (Clínica)", value=True)

st.markdown("---")

if analyze_btn and st.session_state.get("ecg_ready"):
    ecg  = st.session_state["ecg_raw"]
    clin = st.session_state["clin_raw"]

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

    cols = st.columns(len(LABEL_NAMES))
    for i, (col, name) in enumerate(zip(cols, LABEL_NAMES)):
        p    = probas[i]
        thr  = thresholds.get(name, 0.5)
        is_above = p >= thr
        
        # Paleta Azul corporativa para activos, gris para inactivos
        bg_color, text_color, border_color = ("#e3f2fd", "#0f4c81", "#0f4c81") if is_above else ("#f8f9fa", "#6c757d", "#dee2e6")

        badge_html = f"""
        <div style="background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color}; border-radius: 6px; padding: 12px; text-align: center; font-family: sans-serif; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
            <div style="font-size: 13px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;">{name}</div>
            <div style="font-size: 24px; font-weight: 700; margin-bottom: 4px;">{p:.2f}</div>
            <div style="font-size: 11px; opacity: 0.8;">Umbral clínico: {thr:.2f}</div>
        </div>
        """
        col.markdown(badge_html, unsafe_allow_html=True)

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
