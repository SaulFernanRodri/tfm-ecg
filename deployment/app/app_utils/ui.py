"""
app_utils/ui.py
===============
Constantes de UI, colores de severidad clínica y funciones compartidas
entre todas las páginas del CDSS.  Una sola fuente de verdad.

Nota: este paquete se llama 'app_utils' (no 'utils') deliberadamente, para
no colisionar con el paquete 'utils/' de la raíz del proyecto (utils.metrics,
utils.seed, etc.). Un nombre compartido puede provocar que Python cachee en
sys.modules['utils'] el paquete equivocado entre reruns de Streamlit,
provocando 'ModuleNotFoundError: No module named utils.ui' de forma
intermitente e independiente del orden de sys.path.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constantes técnicas
# ---------------------------------------------------------------------------
FS: int = 100  # Hz

LEAD_NAMES: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

LABEL_NAMES: list[str] = ["CD", "HYP", "MI", "NORM", "STTC"]

LABEL_FULL: dict[str, str] = {
    "CD":   "Trastorno de Conducción",
    "HYP":  "Hipertrofia",
    "MI":   "Infarto de Miocardio",
    "NORM": "ECG Normal",
    "STTC": "Cambios ST/T",
}

# ---------------------------------------------------------------------------
# Paleta de severidad clínica
# Cada tupla: (fondo, texto, borde, color_barra)
#   NORM  → verde   (tranquilizador)
#   MI/CD → rojo    (urgente)
#   HYP   → ámbar   (moderado)
#   STTC  → naranja (atención)
# ---------------------------------------------------------------------------
LABEL_SEVERITY: dict[str, tuple[str, str, str, str]] = {
    "CD":   ("#fce4ec", "#880e4f", "#ad1457", "#ad1457"),
    "HYP":  ("#fff8e1", "#e65100", "#f57f17", "#f57f17"),
    "MI":   ("#ffebee", "#b71c1c", "#c62828", "#c62828"),
    "NORM": ("#e8f5e9", "#1b5e20", "#2e7d32", "#2e7d32"),
    "STTC": ("#fff3e0", "#bf360c", "#e64a19", "#e64a19"),
}
_INACTIVE: tuple[str, str, str, str] = ("#f8f9fa", "#6c757d", "#dee2e6", "#999999")

# Alias de compatibilidad (color de barra activa por clase)
LABEL_COLOR: dict[str, str] = {k: v[3] for k, v in LABEL_SEVERITY.items()}


# ---------------------------------------------------------------------------
# CSS institucional
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inyecta el CSS compartido: oculta spinners de inputs numéricos."""
    st.markdown("""
<style>
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
input[type=number] { -moz-appearance: textfield; }
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers de Streamlit
# ---------------------------------------------------------------------------
def st_blue_alert(message: str, is_bold: bool = False) -> None:
    """Bloque de alerta con borde izquierdo azul institucional (#0f4c81)."""
    fw = "bold" if is_bold else "normal"
    st.markdown(
        f'<div style="background-color:#f0f4f8;border-left:4px solid #0f4c81;'
        f'color:#0f4c81;padding:12px;border-radius:4px;margin-bottom:1rem;'
        f'font-family:sans-serif;font-weight:{fw};">{message}</div>',
        unsafe_allow_html=True,
    )


def render_diagnosis_badges(probas: np.ndarray, thresholds: dict) -> None:
    """
    Renderiza las 5 tarjetas de diagnóstico con color según severidad clínica.

    Activo (≥ umbral): color propio de la superclase
        NORM → verde, MI → rojo, CD → vino, HYP → ámbar, STTC → naranja
    Inactivo: gris neutro.
    """
    cols = st.columns(len(LABEL_NAMES))
    for i, (col, name) in enumerate(zip(cols, LABEL_NAMES)):
        p = float(probas[i])
        thr = thresholds.get(name, 0.5)
        bg, text, border, _ = LABEL_SEVERITY[name] if p >= thr else _INACTIVE
        col.markdown(
            f'<div style="background-color:{bg};color:{text};border:1px solid {border};'
            f'border-radius:6px;padding:12px;text-align:center;font-family:sans-serif;'
            f'margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.05);">'
            f'<div style="font-size:13px;font-weight:600;margin-bottom:4px;'
            f'text-transform:uppercase;">{name}</div>'
            f'<div style="font-size:24px;font-weight:700;margin-bottom:4px;">{p:.2f}</div>'
            f'<div style="font-size:11px;opacity:.8;">Umbral clínico: {thr:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def preprocess_ecg_single(ecg_raw: np.ndarray, stats: dict) -> np.ndarray:
    return (ecg_raw.astype(np.float32) - stats["mean"]) / (stats["std"] + 1e-8)


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
def plot_predictions(probas: np.ndarray, thresholds: dict) -> go.Figure:
    """Gráfica de barras horizontal con probabilidades y umbral por clase."""
    order = np.argsort(probas)
    sorted_names  = [LABEL_NAMES[i] for i in order]
    sorted_probas = probas[order]
    colors = [
        LABEL_SEVERITY[name][3] if p >= thresholds.get(name, 0.5) else "#999999"
        for name, p in zip(sorted_names, sorted_probas)
    ]
    fig = go.Figure(go.Bar(
        x=sorted_probas,
        y=[f"{n} — {LABEL_FULL[n]}" for n in sorted_names],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.2f}" for p in sorted_probas],
        textposition="outside",
    ))
    for yi, name in enumerate(sorted_names):
        thr = thresholds.get(name, 0.5)
        fig.add_shape(
            type="line", x0=thr, x1=thr, y0=yi - 0.4, y1=yi + 0.4,
            line=dict(color="#0f4c81", width=2, dash="dash"),
        )
    fig.update_layout(
        height=300,
        xaxis=dict(range=[0, 1.15], title="Puntuación del modelo (0–1)", color="#555"),
        yaxis=dict(color="#0f4c81"),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=12),
        margin=dict(l=10, r=70, t=20, b=40),
        showlegend=False,
    )
    return fig


def plot_ecg_gradcam(ecg: np.ndarray, cam: np.ndarray, class_name: str) -> go.Figure:
    """ECG de 12 derivaciones con superposición de mapa Grad-CAM."""
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
                showlegend=False,
                hovertemplate=f"{lead}  t=%{{x:.2f}}s<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.update_yaxes(
            range=[sig_min - margin, sig_max + margin],
            showticklabels=False, row=row, col=col,
        )
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


def plot_ecg_gradcam_single(
    ecg: np.ndarray, cam: np.ndarray, class_name: str, lead_idx: int
) -> go.Figure:
    """ECG de una sola derivación con mapa Grad-CAM."""
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
        colorbar=dict(
            title="CAM", len=0.7, thickness=14,
            tickfont=dict(color="#0f4c81", size=10),
            titlefont=dict(color="#0f4c81", size=11),
        ),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines",
        line=dict(color="#0f4c81", width=1.8),
        showlegend=False,
        hovertemplate=f"t=%{{x:.2f}}s  val=%{{y:.3f}}<extra>{lead}</extra>",
    ))
    fig.update_layout(
        height=280,
        title=dict(
            text=(
                f"ECG + Grad-CAM — <b>{lead}</b> · "
                f"<b>{class_name}</b>: {LABEL_FULL[class_name]}"
            ),
            font=dict(size=14, color="#0f4c81"),
        ),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=11),
        margin=dict(l=40, r=70, t=55, b=45),
        xaxis=dict(title="Tiempo (s)", showgrid=False, zeroline=False, color="#555"),
        yaxis=dict(
            range=[sig_min - margin, sig_max + margin],
            showticklabels=False, showgrid=False, zeroline=False,
        ),
    )
    return fig


def plot_lead_importance_plotly(importances: np.ndarray, class_name: str) -> go.Figure:
    """
    Barras horizontales de importancia (ablación) de cada derivación del ECG.
    El color activo refleja la severidad de la clase analizada.
    """
    sorted_idx = np.argsort(importances)
    vals  = importances[sorted_idx]
    leads = [LEAD_NAMES[i] for i in sorted_idx]
    active_bar = LABEL_SEVERITY.get(class_name, _INACTIVE)[3]
    colors = [active_bar if v > 0 else "#999999" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=leads, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=350,
        title=dict(
            text=f"Importancia Espacial (Derivaciones) — {class_name}",
            font=dict(color="#0f4c81"),
        ),
        xaxis=dict(
            title="Caída de prob. al ablar derivación", color="#555",
            zeroline=True, zerolinecolor="#ccc",
        ),
        yaxis=dict(color="#0f4c81"),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=11),
        margin=dict(l=20, r=80, t=40, b=40),
    )
    return fig


def plot_clinical_influence(influences: np.ndarray, class_name: str) -> go.Figure:
    """
    Barras de impacto marginal de variables clínicas sobre la probabilidad.
    El color activo refleja la severidad de la clase analizada.
    """
    influences_pct = influences * 100
    sorted_idx = np.argsort(influences_pct)
    vals  = influences_pct[sorted_idx]
    feats = [["Edad", "Sexo", "Altura", "Peso"][i] for i in sorted_idx]
    active_bar = LABEL_SEVERITY.get(class_name, _INACTIVE)[3]
    colors = [active_bar if v > 0 else "#999999" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=280,
        title=dict(
            text=f"Contribución Clínica al Riesgo — {class_name}",
            font=dict(color="#0f4c81"),
        ),
        xaxis=dict(
            title="Impacto en la probabilidad (%)", color="#555",
            zeroline=True, zerolinecolor="#ccc",
        ),
        yaxis=dict(color="#0f4c81"),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#0f4c81", size=12),
        margin=dict(l=20, r=80, t=40, b=40),
    )
    return fig
