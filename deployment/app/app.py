"""
Aplicación Streamlit para HuggingFace Spaces — ECG Diagnosis AI.

Diferencias respecto a la versión local (app.py):
  - Rutas sin prefijo "Desarrollo/tfm-ecg/"
  - load_test_samples() lee demo_data/*.npy (sin dataset completo)
  - Sin dependencia de data/loader.py ni data/preprocessor.py

Estructura esperada en el Space:
    app.py
    requirements.txt
    model/losses.py
    xai/gradcam.py
    xai/lead_importance.py
    saved_model/
        ecg_global_stats.joblib
        scaler.joblib
        train_medians.joblib
        optimal_thresholds.json
        v5/best_model.keras
    demo_data/
        ecg_samples.npy
        clin_samples.npy
        true_labels.npy
"""

import sys
import os
from pathlib import Path

_ROOT = Path(__file__).parent
_REPO_ROOT = _ROOT.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from plotly.subplots import make_subplots

from model.losses import AsymmetricLoss
from xai.lead_importance import LEAD_NAMES

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema de Apoyo al Diagnóstico Electrocardiográfico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LABEL_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
LABEL_FULL  = {
    "CD":   "Trastorno de Conducción",
    "HYP":  "Hipertrofia",
    "MI":   "Infarto de Miocardio",
    "NORM": "ECG Normal",
    "STTC": "Cambios ST/T",
}
LABEL_COLOR = {
    "CD":   "#b30000",
    "HYP":  "#800000",
    "MI":   "#cc0000",
    "NORM": "#006064",
    "STTC": "#660000",
}

# Paleta médica: fondo blanco, azules clínicos
BG      = "#ffffff"
BG_PLOT = "#f5f8ff"
TXT     = "#1a237e"
GRID    = "#dee8ff"
CLINICAL_FEATURE_NAMES = ["Edad", "Sexo", "Altura", "Peso"]
FS = 100

HF_REPO_ID  = "SaulFernanRodri/ecg-diagnosis-ai"
HF_REPO_TYPE = "space"

def _hf_token() -> str | None:
    """Lee HF_TOKEN desde st.secrets o variable de entorno (opcional)."""
    try:
        import streamlit as st
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.environ.get("HF_TOKEN")

def _get_path(relative_path: str) -> str:
    """Intenta cargar localmente; si no existe, lo descarga del Hub."""
    local_1 = _ROOT / relative_path
    local_2 = _REPO_ROOT / relative_path
    if local_1.exists():
        return str(local_1)
    if local_2.exists():
        return str(local_2)
    
    from huggingface_hub import hf_hub_download
    token = _hf_token()
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=relative_path,
        repo_type=HF_REPO_TYPE,
        token=token,
    )

DEMO_DIR = _ROOT / "demo_data"
if not DEMO_DIR.exists():
    DEMO_DIR = _REPO_ROOT / "deployment" / "app" / "demo_data"

# ---------------------------------------------------------------------------
# Cache de recursos
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo…")
def load_model():
    model_path = _get_path("saved_model/v5/best_model.keras")
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"AsymmetricLoss": AsymmetricLoss},
    )


@st.cache_resource(show_spinner=False)
def load_artifacts():
    stats    = joblib.load(_get_path("saved_model/ecg_global_stats.joblib"))
    scaler   = joblib.load(_get_path("saved_model/scaler.joblib"))
    medians  = joblib.load(_get_path("saved_model/train_medians.joblib"))
    
    # v6.1: intentar umbrales F0.5 primero; si no existen, usar los originales
    try:
        thr_path = _get_path("saved_model/v6.1/optimal_thresholds.json")
    except Exception:
        thr_path = _get_path("saved_model/optimal_thresholds.json")
        
    with open(thr_path) as f:
        thresholds = json.load(f)
    return stats, scaler, medians, thresholds


@st.cache_data(show_spinner=False)
def load_demo_samples():
    """Carga las 50 muestras de demo pre-exportadas (sin PTB-XL completo)."""
    ecg    = np.load(DEMO_DIR / "ecg_samples.npy")
    clin   = np.load(DEMO_DIR / "clin_samples.npy")
    labels = np.load(DEMO_DIR / "true_labels.npy")
    return ecg, clin, labels


# ---------------------------------------------------------------------------
# Preprocesamiento
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
# Visualizaciones
# ---------------------------------------------------------------------------

def plot_ecg_gradcam(ecg: np.ndarray, cam: np.ndarray, class_name: str) -> go.Figure:
    """ECG 12-lead con Grad-CAM como heatmap de fondo + señal encima.
    Usa 24 trazas en total (vs ~2400 de la versión por segmentos)."""
    t = np.arange(ecg.shape[0]) / FS

    fig = make_subplots(
        rows=6, cols=2, shared_xaxes=True,
        vertical_spacing=0.05, horizontal_spacing=0.08,
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
                colorscale=[[0, "#dbeafe"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
                zmin=0, zmax=1,
                showscale=(i == 0),
                colorbar=dict(
                    title="CAM", len=0.3, y=0.85,
                    tickfont=dict(color=TXT, size=9),
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
                line=dict(color="#1e3a5f", width=1.2),
                showlegend=False,
                hovertemplate=f"{lead}  t=%{{x:.2f}}s<extra></extra>",
            ),
            row=row, col=col,
        )
        # Ajustar rango Y para que el heatmap quede detrás
        fig.update_yaxes(
            range=[sig_min - margin, sig_max + margin],
            showticklabels=False,
            row=row, col=col,
        )

    fig.update_layout(
        height=680,
        title=dict(
            text=f"ECG + Grad-CAM — <b>{class_name}</b>: {LABEL_FULL[class_name]}",
            font=dict(size=14, color=TXT),
        ),
        paper_bgcolor=BG, plot_bgcolor=BG_PLOT,
        font=dict(color=TXT, size=10),
        margin=dict(l=30, r=20, t=55, b=30),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=TXT, tickfont=dict(size=8))
    fig.add_annotation(
        text="Azul claro = baja activación · Azul oscuro = alta activación",
        xref="paper", yref="paper", x=0.5, y=-0.03,
        showarrow=False, font=dict(size=10, color="#64748b"),
    )
    return fig


def plot_ecg_gradcam_single(
    ecg: np.ndarray,
    cam: np.ndarray,
    class_name: str,
    lead_idx: int,
) -> go.Figure:
    """ECG de una sola derivación con Grad-CAM a pantalla completa."""
    lead = LEAD_NAMES[lead_idx]
    t = np.arange(ecg.shape[0]) / FS
    signal = ecg[:, lead_idx]
    sig_min, sig_max = float(signal.min()), float(signal.max())
    margin = (sig_max - sig_min) * 0.3 or 0.5

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[cam], x=t, y=[0],
        colorscale=[[0, "#dbeafe"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title="CAM", len=0.7, thickness=14,
            tickfont=dict(color=TXT, size=10),
            titlefont=dict(color=TXT, size=11),
        ),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal,
        mode="lines",
        line=dict(color="#1e3a5f", width=1.8),
        showlegend=False,
        hovertemplate=f"t=%{{x:.2f}}s  val=%{{y:.3f}}<extra>{lead}</extra>",
    ))
    fig.update_layout(
        height=280,
        title=dict(
            text=f"ECG + Grad-CAM — <b>{lead}</b> · <b>{class_name}</b>: {LABEL_FULL[class_name]}",
            font=dict(size=14, color=TXT),
        ),
        paper_bgcolor=BG, plot_bgcolor=BG_PLOT,
        font=dict(color=TXT, size=11),
        margin=dict(l=40, r=70, t=55, b=45),
        xaxis=dict(title="Tiempo (s)", showgrid=False, zeroline=False, color=TXT),
        yaxis=dict(
            range=[sig_min - margin, sig_max + margin],
            showticklabels=False, showgrid=False, zeroline=False,
        ),
    )
    return fig


def plot_predictions(probas: np.ndarray, thresholds: dict) -> go.Figure:
    colors = [
        LABEL_COLOR[n] if probas[i] >= thresholds.get(n, 0.5) else "#cbd5e1"
        for i, n in enumerate(LABEL_NAMES)
    ]
    fig = go.Figure(go.Bar(
        x=probas,
        y=[f"{n} — {LABEL_FULL[n]}" for n in LABEL_NAMES],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.2f}" for p in probas],
        textposition="outside",
    ))
    for i, name in enumerate(LABEL_NAMES):
        thr = thresholds.get(name, 0.5)
        fig.add_shape(
            type="line", x0=thr, x1=thr, y0=i - 0.4, y1=i + 0.4,
            line=dict(color="#374151", width=2, dash="dash"),
        )
    fig.update_layout(
        height=280,
        xaxis=dict(range=[0, 1.15], title="Puntuación del modelo (0–1)",
                   color=TXT, gridcolor=GRID),
        yaxis=dict(color=TXT),
        paper_bgcolor=BG, plot_bgcolor=BG_PLOT,
        font=dict(color=TXT, size=12),
        margin=dict(l=10, r=70, t=20, b=40),
        showlegend=False,
    )
    return fig


def plot_lead_importance_plotly(importances: np.ndarray, class_name: str) -> go.Figure:
    sorted_idx = np.argsort(importances)
    vals   = importances[sorted_idx]
    leads  = [LEAD_NAMES[i] for i in sorted_idx]
    colors = ["#1565C0" if v > 0 else "#90a4ae" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=leads, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=350, title=dict(text=f"Importancia de derivaciones — {class_name}",
                               font=dict(color=TXT)),
        xaxis=dict(title="Caída de puntuación al suprimir derivación",
                   color=TXT, zeroline=True, zerolinecolor="#94a3b8",
                   gridcolor=GRID),
        yaxis=dict(color=TXT),
        paper_bgcolor=BG, plot_bgcolor=BG_PLOT,
        font=dict(color=TXT, size=11),
        margin=dict(l=20, r=80, t=45, b=40),
    )
    return fig


def plot_clinical_influence(influences: np.ndarray, class_name: str) -> go.Figure:
    # Multiplicar por 100 para mostrar en porcentaje
    influences_pct = influences * 100
    sorted_idx = np.argsort(influences_pct)
    vals   = influences_pct[sorted_idx]
    feats  = [CLINICAL_FEATURE_NAMES[i] for i in sorted_idx]
    colors = ["#B71C1C" if v > 0 else "#90a4ae" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=240,
        title=dict(text=f"Contribución al riesgo del paciente — {class_name}",
                   font=dict(color=TXT)),
        xaxis=dict(title="Impacto en la probabilidad (%)",
                   color=TXT, zeroline=True, zerolinecolor="#94a3b8",
                   gridcolor=GRID),
        yaxis=dict(color=TXT),
        paper_bgcolor=BG, plot_bgcolor=BG_PLOT,
        font=dict(color=TXT, size=12),
        margin=dict(l=20, r=80, t=45, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main():
    st.markdown("""
    <h1 style='text-align:center; color:#0f4c81; font-family:sans-serif;'>Sistema de Apoyo al Diagnóstico Electrocardiográfico</h1>
    <p style='text-align:center; color:#555; margin-top:-10px; font-family:sans-serif;'>
        Diagnóstico multilabel de ECG con explicabilidad · ResNet1D-v5 · PTB-XL
    </p>
    <hr style='border-color:#ccc;'>
    """, unsafe_allow_html=True)

    model = load_model()
    stats, scaler, medians, thresholds = load_artifacts()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Fuente de Datos")
        data_source = st.radio(
            "Origen del ECG",
            ["Muestra de demo (PTB-XL test)", "Subir CSV"],
            label_visibility="collapsed",
        )
        st.markdown("---")

        # Datos clínicos: solo editables en modo CSV
        if data_source == "Subir CSV":
            st.markdown("### Filiación y Biometría")
            age    = st.slider("Edad", 10, 90, 55)
            sex    = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
            sex_v  = 0 if sex == "Hombre" else 1
            height = st.number_input("Altura (cm)", 140, 210, 170)
            weight = st.number_input("Peso (kg)", 40, 150, 75)
            st.markdown("---")
        else:
            # En modo demo, mostrar los datos del paciente cargado (solo lectura)
            loaded = st.session_state.get("loaded_idx")
            if loaded is not None:
                _, demo_clin_all, _ = load_demo_samples()
                raw_clin = demo_clin_all[loaded].copy()
                raw_clin[[0, 2, 3]] = scaler.inverse_transform(raw_clin[[0, 2, 3]].reshape(1, -1))[0]
                sex_label = "Mujer" if int(round(raw_clin[1])) == 1 else "Hombre"
                st.markdown("### Filiación y Biometría (PTB-XL)")
                st.caption("Datos reales del registro cargado")
                c1, c2 = st.columns(2)
                c1.metric("Edad", f"{raw_clin[0]:.0f} años")
                c2.metric("Sexo", sex_label)
                c1.metric("Altura", f"{raw_clin[2]:.0f} cm")
                c2.metric("Peso", f"{raw_clin[3]:.0f} kg")
                st.markdown("---")

        st.markdown("### Módulo de Interpretabilidad (XAI)")
        run_gradcam  = st.checkbox("Grad-CAM", value=True)
        run_leads    = st.checkbox("Lead Importance", value=True)
        run_clinical = st.checkbox("Variables clínicas", value=True)
        st.markdown("---")
        analyze_btn = st.button("Ejecutar Inferencia Diagnóstica", use_container_width=True, type="primary")

    # ── Carga del ECG ─────────────────────────────────────────────────────────
    if data_source == "Muestra de demo (PTB-XL test)":
        demo_ecg, demo_clin, demo_labels = load_demo_samples()
        sample_idx = st.number_input(
            "Índice de muestra (0–49)", min_value=0, max_value=49, value=0,
        )
        if st.button("Cargar muestra", use_container_width=False):
            st.session_state["ecg_raw"]       = demo_ecg[sample_idx]
            st.session_state["clin_raw"]      = demo_clin[sample_idx]
            st.session_state["true_labels"]   = demo_labels[sample_idx]
            st.session_state["ecg_ready"]     = True
            st.session_state["loaded_idx"]    = int(sample_idx)
            st.session_state["data_source"]   = "demo"
            true = demo_labels[sample_idx]
            diags = [["CD","HYP","MI","NORM","STTC"][j] for j,v in enumerate(true) if v==1]
            st.success(f"Muestra {sample_idx} cargada · Etiqueta real: **{' + '.join(diags) if diags else 'NORM'}** · Ejecute la inferencia", icon="✓")
    else:
        uploaded = st.file_uploader(
            "CSV (1000 filas × 12 columnas, una por derivación)", type=["csv"]
        )
        if uploaded is not None:
            import pandas as pd
            df = pd.read_csv(uploaded, header=None)
            if df.shape == (1000, 12):
                st.session_state["ecg_raw"]     = preprocess_ecg_single(df.values, stats)
                st.session_state["clin_raw"]    = preprocess_clinical_single(
                    age, sex_v, height, weight, scaler, medians
                )
                st.session_state["true_labels"] = None
                st.session_state["ecg_ready"]   = True
                st.session_state["data_source"] = "csv"
            else:
                st.error(f"El CSV debe tener forma (1000, 12). Tiene {df.shape}.")

    # ── Análisis ──────────────────────────────────────────────────────────────
    if analyze_btn and st.session_state.get("ecg_ready"):
        ecg  = st.session_state["ecg_raw"]
        # En modo demo se usan los datos clínicos del paciente real (ya escalados)
        # En modo CSV se usan los valores introducidos en el sidebar
        clin = st.session_state["clin_raw"]

        with st.spinner("Ejecutando modelo…"):
            ecg_t  = tf.convert_to_tensor(ecg[np.newaxis], dtype=tf.float32)
            clin_t = tf.convert_to_tensor(clin[np.newaxis], dtype=tf.float32)
            probas = model([ecg_t, clin_t], training=False).numpy()[0]

        detected_raw = [
            LABEL_NAMES[i] for i, p in enumerate(probas)
            if p >= thresholds.get(LABEL_NAMES[i], 0.5)
        ]

        # ── Regla de exclusión mutua NORM ↔ patología ───────────────────
        # NORM se suprime si alguna patología supera su umbral propio
        # y NORM no es el score más alto absoluto.
        _PATHO  = [n for n in LABEL_NAMES if n != "NORM"]
        _ni     = LABEL_NAMES.index("NORM")
        _patho_over_thr = any(
            probas[LABEL_NAMES.index(n)] >= thresholds.get(n, 0.5)
            for n in _PATHO
        )
        _norm_absolute_top = (int(np.argmax(probas)) == _ni)
        if _patho_over_thr and "NORM" in detected_raw and not _norm_absolute_top:
            detected = [n for n in detected_raw if n != "NORM"]
        else:
            detected = detected_raw

        analysis_classes = detected if detected else [LABEL_NAMES[int(np.argmax(probas))]]

        st.markdown("## Resultados de Inferencia")
        st.caption(
            "Nota Metodológica: Las puntuaciones representan valores de confianza del modelo (0–1), "
            "no probabilidades clínicas calibradas. Exclusivo para uso en investigación."
        )
        cols = st.columns(len(LABEL_NAMES))
        for i, (col, name) in enumerate(zip(cols, LABEL_NAMES)):
            p   = probas[i]
            thr = thresholds.get(name, 0.5)
            is_above = p >= thr
            
            if name == "NORM":
                bg_color = "#e0f2f1" if is_above else "#f8f9fa"
                text_color = "#006064" if is_above else "#6c757d"
                border_color = "#006064" if is_above else "#dee2e6"
            else:
                bg_color = "#ffebee" if is_above else "#f8f9fa"
                text_color = "#b30000" if is_above else "#6c757d"
                border_color = "#b30000" if is_above else "#dee2e6"

            badge_html = f"""
            <div style="
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 12px;
                text-align: center;
                font-family: sans-serif;
                margin-bottom: 16px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            ">
                <div style="font-size: 13px; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;">{name}</div>
                <div style="font-size: 24px; font-weight: 700; margin-bottom: 4px;">{p:.2f}</div>
                <div style="font-size: 11px; opacity: 0.8;">Umbral de decisión: {thr:.2f}</div>
            </div>
            """
            col.markdown(badge_html, unsafe_allow_html=True)

        top_class  = LABEL_NAMES[int(np.argmax(probas))]
        norm_proba = float(probas[LABEL_NAMES.index("NORM")])
        norm_thr   = thresholds.get("NORM", 0.5)
        if top_class == "NORM" and norm_proba >= norm_thr:
            st.success(
                f"Diagnóstico Principal: ECG Normal — Nivel de Confianza: **{norm_proba:.2f}** (umbral {norm_thr:.2f})", icon="✓"
            )
            other_det = [n for n in detected if n != "NORM"]
            if other_det:
                st.warning(f"Hallazgos secundarios por encima del umbral: {' · '.join(other_det)}", icon="!")
        elif detected:
            pathos = [n for n in detected if n != "NORM"]
            if pathos:
                st.error(f"Diagnóstico(s) Patológico(s) Detectado(s): {' · '.join(pathos)}", icon="!")
            else:
                st.success(f"Diagnóstico Principal: ECG Normal — Nivel de Confianza: **{norm_proba:.2f}**", icon="✓")
        else:
            st.success("No se detectan hallazgos patológicos por encima del umbral de decisión.", icon="✓")

        if st.session_state.get("true_labels") is not None:
            true      = st.session_state["true_labels"]
            true_names = [LABEL_NAMES[i] for i, v in enumerate(true) if v == 1]
            st.info(f"Diagnóstico Confirmado (Ground Truth PTB-XL): {' · '.join(true_names) if true_names else 'NORM'}", icon="i")

        st.plotly_chart(plot_predictions(probas, thresholds), use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Localización Temporal (Grad-CAM)", "Análisis Espacial (Derivaciones)", "Factores de Riesgo (Clínica)"])

        with tab1:
            if run_gradcam:
                if not detected:
                    st.info("Ninguna clase supera el umbral de decisión — mostrando la de mayor probabilidad.", icon="i")
                from xai.gradcam import compute_gradcam

                # Selector de derivación
                lead_options = ["Todas (6×2)"] + list(LEAD_NAMES)
                sel_lead = st.radio(
                    "Derivación a visualizar:",
                    lead_options,
                    horizontal=True,
                    key="gradcam_lead_selector",
                )

                # Selector de patología cuando hay más de una
                if len(analysis_classes) > 1:
                    cam_class = st.radio(
                        "Seleccione la patología para inspeccionar el mapa Grad-CAM:",
                        analysis_classes,
                        format_func=lambda n: f"{n} — {LABEL_FULL[n]}",
                        horizontal=True,
                        key="gradcam_class_selector",
                    )
                else:
                    cam_class = analysis_classes[0]

                with st.spinner(f"Calculando Grad-CAM para {cam_class}…"):
                    cam = compute_gradcam(model, ecg, clin, LABEL_NAMES.index(cam_class))
                if sel_lead == "Todas (6×2)":
                    st.plotly_chart(plot_ecg_gradcam(ecg, cam, cam_class), use_container_width=True)
                else:
                    lead_idx = list(LEAD_NAMES).index(sel_lead)
                    st.plotly_chart(plot_ecg_gradcam_single(ecg, cam, cam_class, lead_idx), use_container_width=True)
                st.caption(
                    f"**Azul oscuro**: segmentos que más activaron la predicción de "
                    f"**{cam_class} — {LABEL_FULL[cam_class]}**. **Azul claro/Blanco**: baja influencia."
                )
            else:
                st.info("Active 'Grad-CAM' en el panel lateral.", icon="i")

        with tab2:
            if run_leads:
                if not detected:
                    st.info("Ninguna clase supera el umbral de decisión — mostrando la de mayor probabilidad.", icon="i")
                from xai.lead_importance import compute_lead_importance_single

                # Selector de patología cuando hay más de una
                if len(analysis_classes) > 1:
                    leads_class = st.radio(
                        "Seleccione la patología para inspeccionar la importancia de derivaciones:",
                        analysis_classes,
                        format_func=lambda n: f"{n} — {LABEL_FULL[n]}",
                        horizontal=True,
                        key="leads_class_selector",
                    )
                else:
                    leads_class = analysis_classes[0]

                with st.spinner(f"Calculando importancia de derivaciones para {leads_class}…"):
                    importances = compute_lead_importance_single(
                        model, ecg, clin, class_idx=LABEL_NAMES.index(leads_class)
                    )
                st.plotly_chart(plot_lead_importance_plotly(importances, leads_class), use_container_width=True)
                st.caption("**Azul oscuro/Granate**: derivación altamente relevante para el diagnóstico. **Gris**: aporte de información limitado.")
            else:
                st.info("Active 'Lead Importance' en el panel lateral.", icon="i")

        with tab3:
            if run_clinical:
                if not detected:
                    st.info("Ninguna clase supera el umbral de decisión — mostrando la de mayor probabilidad.", icon="i")
                from xai.clinical_ablation import compute_clinical_influence
                for cls in analysis_classes:
                    with st.spinner(f"Calculando influencia clínica para {cls}…"):
                        infl = compute_clinical_influence(
                            model, ecg, clin, class_idx=LABEL_NAMES.index(cls)
                        )
                    st.plotly_chart(plot_clinical_influence(infl, cls), use_container_width=True)
                st.caption(
                    "Muestra el impacto marginal de las variables sobre el paciente actual frente a la media. "
                    "**Granate**: aumenta la probabilidad. **Gris**: sin impacto significativo o protector.  \n"
                    "_Nota Metodológica: Para la variable categórica (Sexo), se realiza un análisis contrafactual evaluando qué sucedería si el paciente "
                    "perteneciese a la otra categoría biológica._"
                )
            else:
                st.info("Active 'Variables clínicas' en el panel lateral para ejecutar este análisis.", icon="i")

    elif not st.session_state.get("ecg_ready"):
        st.markdown("---")
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.markdown("""
            ### Guía de Uso del Sistema de Inferencia
            1. Seleccione **"Muestra de demo"** en el panel lateral y presione **Cargar muestra**
            2. Alternativamente, suba su propio registro ECG como archivo CSV (matriz 1000×12)
            3. Pulse **Ejecutar Inferencia Diagnóstica**
            ---
            **Arquitectura del Modelo:** ResNet1D-5 + SE Attention + ASL  
            **Cohorte de Entrenamiento:** PTB-XL v1.0.3 · 21.837 registros ECG  
            **Métricas de Desempeño (Test):** AUC ROC Macro: 0.9255 · Sensibilidad Global: 0.9463  

            > Prototipo de investigación (CDSS). Exclusivo para uso investigativo y no apto para diagnóstico clínico directo sin supervisión facultativa.
            """)


if __name__ == "__main__":
    if "ecg_ready" not in st.session_state:
        st.session_state["ecg_ready"] = False
    main()
