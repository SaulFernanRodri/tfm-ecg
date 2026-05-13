"""
Visualizaciones XAI para el modelo ECG multimodal.

Genera plots para:
- Grad-CAM 1D: señal ECG coloreada por el mapa de activación
- Lead Importance: barras horizontales por clase
- SHAP clínico: beeswarm y barras de importancia media
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


LEAD_NAMES: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"]

# Segundos por muestra a 100 Hz
FS: int = 100


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def plot_gradcam(
    ecg_sample:  np.ndarray,
    cam:         np.ndarray,
    class_name:  str,
    lead_names:  List[str] = LEAD_NAMES,
    fs:          int       = FS,
    save_path:   Optional[Path] = None,
) -> plt.Figure:
    """
    Superpone el mapa Grad-CAM sobre las 12 derivaciones del ECG.

    Args:
        ecg_sample: Array (T, 12) — señal ECG normalizada.
        cam:        Array (T,) — mapa de calor Grad-CAM en [0, 1].
        class_name: Nombre de la clase para el título.
        lead_names: Lista de nombres de derivación.
        fs:         Frecuencia de muestreo (Hz).
        save_path:  Ruta donde guardar la figura. Si None, no guarda.

    Returns:
        fig: Figura matplotlib.
    """
    T = ecg_sample.shape[0]
    t = np.arange(T) / fs

    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Grad-CAM — Clase: {class_name}", fontsize=14, fontweight="bold")

    axes_flat = axes.flatten()
    cmap = cm.get_cmap("RdYlGn_r")  # rojo=alta activación, verde=baja

    for i, ax in enumerate(axes_flat):
        signal = ecg_sample[:, i]
        # Colorear la señal segmento a segmento por intensidad del CAM
        for j in range(T - 1):
            color = cmap(cam[j])
            ax.plot(t[j:j+2], signal[j:j+2], color=color, linewidth=0.8)

        ax.set_ylabel(lead_names[i], fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_yticks([])

    axes_flat[-2].set_xlabel("Tiempo (s)", fontsize=9)
    axes_flat[-1].set_xlabel("Tiempo (s)", fontsize=9)

    # Colorbar compartida
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Activación Grad-CAM", fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_gradcam_all_classes(
    ecg_sample:  np.ndarray,
    cams:        Dict[str, np.ndarray],
    output_dir:  Path,
    fs:          int = FS,
) -> None:
    """
    Genera un plot Grad-CAM por cada clase y lo guarda en output_dir.

    Args:
        ecg_sample: Array (T, 12).
        cams:       Diccionario {clase: cam(T,)}.
        output_dir: Directorio de salida.
        fs:         Frecuencia de muestreo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, cam in cams.items():
        path = output_dir / f"gradcam_{class_name}.png"
        fig = plot_gradcam(ecg_sample, cam, class_name, fs=fs, save_path=path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Lead Importance
# ---------------------------------------------------------------------------

def plot_lead_importance(
    importances: np.ndarray,
    class_name:  str,
    lead_names:  List[str] = LEAD_NAMES,
    save_path:   Optional[Path] = None,
) -> plt.Figure:
    """
    Barras horizontales de importancia de cada lead para una clase.

    Args:
        importances: Array (12,) — importancia por lead.
        class_name:  Nombre de la clase.
        lead_names:  Lista de nombres de derivación.
        save_path:   Ruta de guardado.

    Returns:
        fig: Figura matplotlib.
    """
    sorted_idx = np.argsort(importances)
    sorted_imp = importances[sorted_idx]
    sorted_leads = [lead_names[i] for i in sorted_idx]

    colors = ["#d73027" if v > 0 else "#4575b4" for v in sorted_imp]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(sorted_leads, sorted_imp, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Caída de probabilidad al ablar lead", fontsize=10)
    ax.set_title(f"Lead Importance — Clase: {class_name}", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    # Anotación de valores
    for bar, val in zip(bars, sorted_imp):
        x = bar.get_width()
        offset = 0.001 if x >= 0 else -0.001
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left" if x >= 0 else "right", fontsize=7)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_lead_importance_all_classes(
    importances_per_class: Dict[str, np.ndarray],
    output_dir:            Path,
) -> None:
    """
    Genera plots de lead importance para todas las clases.

    Args:
        importances_per_class: Diccionario {clase: array(12,)}.
        output_dir:            Directorio de salida.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, importances in importances_per_class.items():
        path = output_dir / f"lead_importance_{class_name}.png"
        fig = plot_lead_importance(importances, class_name, save_path=path)
        plt.close(fig)


def plot_lead_importance_heatmap(
    importances_per_class: Dict[str, np.ndarray],
    lead_names:            List[str] = LEAD_NAMES,
    save_path:             Optional[Path] = None,
) -> plt.Figure:
    """
    Heatmap de importancia de leads × clases.

    Args:
        importances_per_class: Diccionario {clase: array(12,)}.
        lead_names:            Nombres de derivaciones.
        save_path:             Ruta de guardado.

    Returns:
        fig: Figura matplotlib.
    """
    classes  = list(importances_per_class.keys())
    matrix   = np.stack([importances_per_class[c] for c in classes], axis=0)  # (C, 12)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())
    ax.set_xticks(range(len(lead_names)))
    ax.set_xticklabels(lead_names, fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_title("Lead Importance por clase (ablación)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Caída de probabilidad")

    # Anotaciones de valor
    for i in range(len(classes)):
        for j in range(len(lead_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(matrix[i, j]) > 0.5 * np.abs(matrix).max() else "black")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# SHAP clínico
# ---------------------------------------------------------------------------

def plot_shap_bar(
    mean_abs_shap:   np.ndarray,
    class_name:      str,
    feature_names:   List[str],
    save_path:       Optional[Path] = None,
) -> plt.Figure:
    """
    Barras de importancia SHAP media (|SHAP| promedio) para una clase.

    Args:
        mean_abs_shap: Array (n_features,) — |SHAP| medio por feature.
        class_name:    Nombre de la clase.
        feature_names: Nombres de las variables clínicas.
        save_path:     Ruta de guardado.

    Returns:
        fig: Figura matplotlib.
    """
    sorted_idx    = np.argsort(mean_abs_shap)
    sorted_values = mean_abs_shap[sorted_idx]
    sorted_names  = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(sorted_names, sorted_values, color="#e66101", edgecolor="white")
    ax.set_xlabel("mean(|SHAP value|)", fontsize=10)
    ax.set_title(f"SHAP Feature Importance — Clase: {class_name}",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_beeswarm(
    shap_values:   np.ndarray,
    clinical_data: np.ndarray,
    class_name:    str,
    feature_names: List[str],
    save_path:     Optional[Path] = None,
) -> plt.Figure:
    """
    Beeswarm plot de SHAP para visualizar dirección e impacto.

    Cada punto es una muestra. El color indica el valor de la feature
    (rojo=alto, azul=bajo). El eje X es el SHAP value.

    Args:
        shap_values:   Array (M, n_features).
        clinical_data: Array (M, n_features) — valores reales de las features.
        class_name:    Nombre de la clase.
        feature_names: Nombres de las variables clínicas.
        save_path:     Ruta de guardado.

    Returns:
        fig: Figura matplotlib.
    """
    import shap as shap_lib

    fig, ax = plt.subplots(figsize=(7, 4))
    shap_lib.plots.beeswarm(
        shap_lib.Explanation(
            values=shap_values,
            base_values=np.zeros(len(shap_values)),
            data=clinical_data,
            feature_names=feature_names,
        ),
        show=False,
        ax=ax,
    )
    ax.set_title(f"SHAP Beeswarm — Clase: {class_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_all_classes(
    shap_per_class:  Dict[str, np.ndarray],
    clinical_data:   np.ndarray,
    feature_names:   List[str],
    output_dir:      Path,
) -> None:
    """
    Genera plots SHAP (barras + beeswarm) para todas las clases.

    Args:
        shap_per_class: Diccionario {clase: shap_values(M, n_features)}.
        clinical_data:  Array (M, n_features).
        feature_names:  Nombres de features.
        output_dir:     Directorio de salida.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, shap_values in shap_per_class.items():
        mean_abs = np.mean(np.abs(shap_values), axis=0)

        fig_bar = plot_shap_bar(
            mean_abs, class_name, feature_names,
            save_path=output_dir / f"shap_bar_{class_name}.png",
        )
        plt.close(fig_bar)

        try:
            fig_bee = plot_shap_beeswarm(
                shap_values, clinical_data, class_name, feature_names,
                save_path=output_dir / f"shap_beeswarm_{class_name}.png",
            )
            plt.close(fig_bee)
        except Exception:
            # beeswarm requiere shap >= 0.42; si falla, solo guardamos barras
            pass
