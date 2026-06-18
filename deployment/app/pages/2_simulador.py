import sys
import os
from pathlib import Path

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
# Inyección de CSS para Impresión (Print Media Query)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@media print {
    /* Ocultar elementos de navegación de Streamlit al imprimir */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .print-button-container { display: none !important; }
    /* Ajustes generales de página para el PDF */
    body { background-color: white !important; color: black !important; }
    .stApp { background-color: white !important; }
}
</style>
""", unsafe_allow_html=True)

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
    "CD":   "#b30000",
    "HYP":  "#800000",
    "MI":   "#cc0000",
    "NORM": "#006064",
    "STTC": "#660000",
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

@st.cache_data(show_spinner="Cargando muestras de test…")
def load_test_samples(n: int = 50):
    from data.loader import load_dataset
    from data.preprocessor import preprocess_ecg_splits, preprocess_clinical

    train_data, val_data, test_data, label_names = load_dataset()
    train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
        train_data["ecg"], val_data["ecg"], test_data["ecg"]
    )
    _, _, test_clin, _, _ = preprocess_clinical(
        train_data["clinical"], val_data["clinical"], test_data["clinical"]
    )
    return test_ecg[:n], test_clin[:n], test_data["labels"][:n]

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
# Visualizaciones Plotly
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
                colorscale=[[0, "#ffffff"], [0.5, "#90caf9"], [1, "#b30000"]],
                zmin=0, zmax=1,
                showscale=(i == 0),
                colorbar=dict(
                    title="CAM", len=0.3, y=0.85,
                    tickfont=dict(color="#1a237e", size=9),
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
        font=dict(color="#1a237e", size=10),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#555", tickfont=dict(size=8))
    fig.add_annotation(
        text="Blanco/Celeste = baja activación · Granate = alta activación",
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
        colorscale=[[0, "#ffffff"], [0.5, "#90caf9"], [1, "#b30000"]],
        zmin=0, zmax=1, showscale=True,
        colorbar=dict(title="CAM", len=0.7, thickness=14, tickfont=dict(color="#1a237e", size=10), titlefont=dict(color="#1a237e", size=11)),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines", line=dict(color="#1a237e", width=1.8),
        showlegend=False, hovertemplate=f"t=%{{x:.2f}}s  val=%{{y:.3f}}<extra>{lead}</extra>",
    ))
    fig.update_layout(
        height=280,
        title=dict(
            text=f"ECG + Grad-CAM — <b>{lead}</b> · <b>{class_name}</b>: {LABEL_FULL[class_name]}",
            font=dict(size=14, color="#0f4c81"),
        ),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#1a237e", size=11), margin=dict(l=40, r=70, t=55, b=45),
        xaxis=dict(title="Tiempo (s)", showgrid=False, zeroline=False, color="#555"),
        yaxis=dict(range=[sig_min - margin, sig_max + margin], showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig

def plot_predictions(probas: np.ndarray, thresholds: dict) -> go.Figure:
    order = np.argsort(probas)
    sorted_names  = [LABEL_NAMES[i] for i in order]
    sorted_probas = probas[order]
    colors = [LABEL_COLOR[name] if p >= thresholds.get(name, 0.5) else "#555" for name, p in zip(sorted_names, sorted_probas)]

    fig = go.Figure(go.Bar(
        x=sorted_probas, y=[f"{n} — {LABEL_FULL[n]}" for n in sorted_names],
        orientation="h", marker_color=colors, text=[f"{p:.2f}" for p in sorted_probas], textposition="outside",
    ))
    for yi, name in enumerate(sorted_names):
        thr = thresholds.get(name, 0.5)
        fig.add_shape(type="line", x0=thr, x1=thr, y0=yi - 0.4, y1=yi + 0.4, line=dict(color="#1a237e", width=2, dash="dash"))
    fig.update_layout(
        height=300, xaxis=dict(range=[0, 1.15], title="Puntuación del modelo (0–1)", color="#555"),
        yaxis=dict(color="#1a237e"), paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#1a237e", size=12), margin=dict(l=10, r=70, t=20, b=40), showlegend=False,
    )
    return fig

def plot_lead_importance_plotly(importances: np.ndarray, class_name: str) -> go.Figure:
    sorted_idx = np.argsort(importances)
    vals   = importances[sorted_idx]
    leads  = [LEAD_NAMES[i] for i in sorted_idx]
    colors = ["#b30000" if v > 0 else "#0f4c81" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=leads, orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        height=350, title=dict(text=f"Importancia Espacial (Derivaciones) — {class_name}", font=dict(color="#0f4c81")),
        xaxis=dict(title="Caída de prob. al ablar derivación", color="#555", zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(color="#1a237e"), paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#1a237e", size=11), margin=dict(l=20, r=80, t=40, b=40),
    )
    return fig

def plot_clinical_influence(influences: np.ndarray, class_name: str) -> go.Figure:
    influences_pct = influences * 100
    sorted_idx = np.argsort(influences_pct)
    vals   = influences_pct[sorted_idx]
    CLINICAL_FEATURE_NAMES = ["Edad", "Sexo", "Altura", "Peso"]
    feats  = [CLINICAL_FEATURE_NAMES[i] for i in sorted_idx]
    colors = ["#b30000" if v > 0 else "#0f4c81" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h", marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        height=280, title=dict(text=f"Contribución Clínica al Riesgo — {class_name}", font=dict(color="#0f4c81")),
        xaxis=dict(title="Impacto en la probabilidad (%)", color="#555", zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(color="#1a237e"), paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#1a237e", size=12), margin=dict(l=20, r=80, t=40, b=40),
    )
    return fig

# ---------------------------------------------------------------------------
# Layout principal de Simulador
# ---------------------------------------------------------------------------

st.markdown("""
<h2 style='text-align:center; color:#0f4c81; font-family:sans-serif;'>Simulador Clínico (CDSS)</h2>
<hr style='border-color:#ccc;'>
""", unsafe_allow_html=True)

model = load_model()
stats, scaler, medians, thresholds = load_artifacts()

with st.sidebar:
    st.markdown("### Fuente de Datos")
    data_source = st.radio("Origen del ECG", ["Muestra del test set", "Subir CSV"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Filiación y Biometría")
    age    = st.slider("Edad", 10, 90, 55)
    sex    = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
    sex_v  = 0 if sex == "Hombre" else 1
    height = st.number_input("Altura (cm)", 140, 210, 170)
    weight = st.number_input("Peso (kg)", 40, 150, 75)
    st.markdown("---")
    st.markdown("### Módulo de Interpretabilidad (XAI)")
    run_gradcam  = st.checkbox("Grad-CAM", value=True)
    run_leads    = st.checkbox("Lead Importance", value=True)
    run_clinical = st.checkbox("Variables clínicas", value=True)
    st.markdown("---")
    analyze_btn = st.button("Ejecutar Inferencia Diagnóstica", use_container_width=True, type="primary")

if "ecg_ready" not in st.session_state:
    st.session_state["ecg_ready"] = False

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
    uploaded = st.file_uploader("CSV con forma (1000, 12) — una fila por muestra, 12 columnas (leads)", type=["csv"])
    if uploaded is not None:
        import pandas as pd
        df = pd.read_csv(uploaded, header=None)
        if df.shape == (1000, 12):
            st.session_state["ecg_raw"]  = preprocess_ecg_single(df.values, stats)
            st.session_state["clin_raw"] = preprocess_clinical_single(age, sex_v, height, weight, scaler, medians)
            st.session_state["true_labels"] = None
            st.session_state["ecg_ready"] = True
        else:
            st.error(f"El CSV debe tener forma (1000, 12). Tiene {df.shape}.")

if analyze_btn and st.session_state.get("ecg_ready"):
    ecg  = st.session_state["ecg_raw"]
    clin = preprocess_clinical_single(age, sex_v, height, weight, scaler, medians)

    with st.spinner("Ejecutando modelo…"):
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

    # ── Panel de resultados ───────────────────────────────────────────────
    st.markdown("## Resultados de Inferencia")
    
    top_idx   = int(np.argmax(probas))
    top_class = LABEL_NAMES[top_idx]
    norm_idx  = LABEL_NAMES.index("NORM")
    norm_proba = float(probas[norm_idx])
    norm_thr   = thresholds.get("NORM", 0.5)
    norm_is_top = (top_class == "NORM")

    st.caption("Nota Metodológica: Las puntuaciones representan valores de confianza del modelo (0–1), no probabilidades clínicas calibradas. Exclusivo para uso en investigación.")

    cols = st.columns(len(LABEL_NAMES))
    for i, (col, name) in enumerate(zip(cols, LABEL_NAMES)):
        p    = probas[i]
        thr  = thresholds.get(name, 0.5)
        is_above = p >= thr
        
        if name == "NORM":
            bg_color, text_color, border_color = ("#e0f2f1", "#006064", "#006064") if is_above else ("#f8f9fa", "#6c757d", "#dee2e6")
        else:
            bg_color, text_color, border_color = ("#ffebee", "#b30000", "#b30000") if is_above else ("#f8f9fa", "#6c757d", "#dee2e6")

        badge_html = f"""
        <div style="background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color}; border-radius: 6px; padding: 12px; text-align: center; font-family: sans-serif; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
            <div style="font-size: 13px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;">{name}</div>
            <div style="font-size: 24px; font-weight: 700; margin-bottom: 4px;">{p:.2f}</div>
            <div style="font-size: 11px; opacity: 0.8;">Umbral de decisión: {thr:.2f}</div>
        </div>
        """
        col.markdown(badge_html, unsafe_allow_html=True)

    if norm_is_top and norm_proba >= norm_thr:
        st.success(f"Diagnóstico Principal: ECG Normal — Nivel de Confianza: {norm_proba:.2f} (umbral {norm_thr:.2f})")
        other_detected = [n for n in predicted if n != "NORM"]
        if other_detected:
            st.warning(f"Hallazgos secundarios por encima del umbral: {' · '.join(other_detected)}")
    elif predicted:
        pathos = [n for n in predicted if n != "NORM"]
        norm_detected = "NORM" in predicted
        if pathos:
            st.error(f"Diagnóstico(s) Patológico(s) Detectado(s): {' · '.join(pathos)}" + ("  |  Se detectan patrones compatibles con ECG Normal" if norm_detected else ""))
        else:
            st.success(f"Diagnóstico Principal: ECG Normal — Nivel de Confianza: {norm_proba:.2f}")
    else:
        st.success("No se detectan hallazgos patológicos por encima del umbral de decisión.")

    if st.session_state.get("true_labels") is not None:
        true = st.session_state["true_labels"]
        true_names = [LABEL_NAMES[i] for i, v in enumerate(true) if v == 1]
        st.info(f"Diagnóstico Confirmado (Ground Truth): {' · '.join(true_names) if true_names else 'NORM'}")

    st.plotly_chart(plot_predictions(probas, thresholds), use_container_width=True)

    # ── BOTÓN IMPRIMIR INFORME CLÍNICO ────────────────────────────────────
    print_btn_html = """
    <div class="print-button-container" style="margin-top: 10px; margin-bottom: 20px;">
        <button onclick="window.print()" style="
            background-color: #0f4c81;
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: sans-serif;
            width: 100%;
            font-weight: 600;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        ">
            Imprimir Informe Clínico
        </button>
    </div>
    """
    components.html(print_btn_html, height=70)

    # ── Tabs de XAI ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Localización Temporal (Grad-CAM)", "Análisis Espacial (Derivaciones)", "Factores de Riesgo (Clínica)"])

    detected_classes = [LABEL_NAMES[i] for i, p in enumerate(probas) if p >= thresholds.get(LABEL_NAMES[i], 0.5)]
    if norm_is_top:
        analysis_classes = ["NORM"] + [c for c in detected_classes if c != "NORM"]
        if not analysis_classes: analysis_classes = ["NORM"]
    else:
        analysis_classes = detected_classes if detected_classes else [LABEL_NAMES[int(np.argmax(probas))]]

    with tab1:
        if run_gradcam:
            from xai.gradcam import compute_gradcam
            if not detected_classes: st.info("Ninguna clase supera el umbral de decisión — mostrando la clase con mayor probabilidad.")

            if len(analysis_classes) > 1:
                cam_class = st.radio("Seleccione la patología para inspeccionar el mapa Grad-CAM:", analysis_classes, format_func=lambda n: f"{n} — {LABEL_FULL[n]}", horizontal=True, key="gradcam_class_selector")
            else:
                cam_class = analysis_classes[0]

            with st.spinner(f"Calculando Grad-CAM para {cam_class}…"):
                cam = compute_gradcam(model, ecg, clin, LABEL_NAMES.index(cam_class))
            st.plotly_chart(plot_ecg_gradcam(ecg, cam, cam_class), use_container_width=True)
            st.caption(f"Segmentos **granates**: activaron la predicción de **{cam_class} — {LABEL_FULL[cam_class]}**. Segmentos **blancos/celestes**: baja influencia.")
        else:
            st.info("Active 'Grad-CAM' en el panel lateral para visualizar el mapa de calor.")

    with tab2:
        if run_leads:
            from xai.lead_importance import compute_lead_importance_single
            if not detected_classes: st.info("Ninguna clase supera el umbral de decisión — mostrando la clase con mayor probabilidad.")

            if len(analysis_classes) > 1:
                leads_class = st.radio("Seleccione la patología para inspeccionar la importancia de derivaciones:", analysis_classes, format_func=lambda n: f"{n} — {LABEL_FULL[n]}", horizontal=True, key="leads_class_selector")
            else:
                leads_class = analysis_classes[0]

            with st.spinner(f"Calculando importancia espacial para {leads_class}…"):
                importances = compute_lead_importance_single(model, ecg, clin, class_idx=LABEL_NAMES.index(leads_class))
            st.plotly_chart(plot_lead_importance_plotly(importances, leads_class), use_container_width=True)
            st.caption("Barras **granates**: suprimir esta derivación reduce la confianza (alta relevancia). Barras **azules**: aporte de información limitado para este diagnóstico.")
        else:
            st.info("Active 'Lead Importance' en el panel lateral.")

    with tab3:
        if run_clinical:
            from xai.clinical_ablation import compute_clinical_influence
            if not detected_classes: st.info("Ninguna clase supera el umbral de decisión — mostrando la clase con mayor probabilidad.")

            for cls in analysis_classes:
                with st.spinner(f"Calculando influencia clínica para {cls}…"):
                    infl = compute_clinical_influence(model, ecg, clin, class_idx=LABEL_NAMES.index(cls))
                st.plotly_chart(plot_clinical_influence(infl, cls), use_container_width=True)
            st.caption(
                "Muestra el impacto marginal de las variables sobre el paciente actual frente a la media. "
                "**Granate**: aumenta la probabilidad. **Azul**: protector o bajo impacto.  \n"
                "_Nota Metodológica: Para la variable categórica (Sexo), se realiza un análisis contrafactual evaluando qué sucedería si el paciente "
                "perteneciese a la otra categoría biológica._"
            )
        else:
            st.info("Active 'Variables clínicas' en el panel lateral para ejecutar este análisis.")

elif not st.session_state.get("ecg_ready"):
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### Guía de Uso del Sistema de Inferencia

        1. **Seleccione el origen del registro ECG** en el panel lateral (Filiación y Biometría):
            - *Muestra del test set*: Casos clínicos extraídos del dataset PTB-XL.
            - *Subir CSV*: Carga de un registro propio estructurado en matriz (1000×12).

        2. **Determine los parámetros biométricos del paciente** (edad, sexo, altura, peso).

        3. **Configure los módulos de interpretabilidad** en el panel lateral.

        4. Ejecute el análisis haciendo clic en **Ejecutar Inferencia Diagnóstica**.
        """)
