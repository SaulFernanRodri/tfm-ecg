"""
Módulo de evaluación del modelo sobre el conjunto de test.

Responsabilidades:
- Cargar el modelo guardado en formato SavedModel
- Generar predicciones probabilísticas sobre el test set
- Calcular métricas completas (AUC-ROC, F1, Recall, Especificidad)
- Construir la matriz de confusión multilabel
- Guardar todos los resultados en results/metrics.json

El objetivo mínimo de sensibilidad es 0.90 macro, reflejando la
prioridad clínica de no dejar infartos sin detectar
(alta penalización de los falsos negativos).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla para servidores sin display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.metrics import compute_specificity

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
SAVED_MODEL_DIR = Path("Desarrollo/tfm-ecg/saved_model")
RESULTS_DIR     = Path("Desarrollo/tfm-ecg/results")
PLOTS_DIR       = RESULTS_DIR / "plots"

# Umbral de decisión por defecto para binarizar probabilidades
THRESHOLD: float = 0.5

# Objetivo clínico mínimo de sensibilidad macro
MIN_SENSITIVITY: float = 0.90


# ===========================================================================
# OPTIMIZACIÓN DE THRESHOLDS POR CLASE
# ===========================================================================

def find_optimal_thresholds(
    y_true:          np.ndarray,
    y_pred_proba:    np.ndarray,
    label_names:     List[str],
    min_specificity: float = 0.65,
) -> Dict[str, float]:
    """
    Encuentra el umbral óptimo por clase en el conjunto de validación.

    Estrategia: para cada clase, maximizar la sensibilidad (recall)
    sujeto a especificidad ≥ min_specificity. Si ningún umbral satisface
    la restricción, se usa el threshold que maximiza el F1-Score.

    Args:
        y_true:          Array binario (N_val, n_classes) de etiquetas reales.
        y_pred_proba:    Array (N_val, n_classes) de probabilidades.
        label_names:     Lista de nombres de clase.
        min_specificity: Especificidad mínima requerida. Por defecto 0.65.

    Returns:
        Diccionario {nombre_clase: threshold_óptimo}.
    """
    candidate_thrs = np.linspace(0.05, 0.95, 91)
    optimal: Dict[str, float] = {}

    print(f"\n[Threshold] Optimizando umbrales en validación "
          f"(min_spec={min_specificity:.2f})...")

    for i, name in enumerate(label_names):
        col   = y_true[:, i]
        proba = y_pred_proba[:, i]

        best_sens   = -1.0
        best_thr_s  = None   # threshold óptimo por sensibilidad
        best_f1     = -1.0
        best_thr_f1 = 0.5    # fallback F1

        for thr in candidate_thrs:
            y_pred_i = (proba >= thr).astype(int)
            tn = int(np.sum((col == 0) & (y_pred_i == 0)))
            fp = int(np.sum((col == 0) & (y_pred_i == 1)))
            fn = int(np.sum((col == 1) & (y_pred_i == 0)))
            tp = int(np.sum((col == 1) & (y_pred_i == 1)))

            spec = tn / (tn + fp + 1e-9)
            sens = tp / (tp + fn + 1e-9)
            f1   = float(f1_score(col, y_pred_i, zero_division=0))

            # Sensibilidad máxima con restricción de especificidad
            if spec >= min_specificity and sens > best_sens:
                best_sens  = sens
                best_thr_s = float(thr)

            # F1 máximo (fallback)
            if f1 > best_f1:
                best_f1     = f1
                best_thr_f1 = float(thr)

        chosen = best_thr_s if best_thr_s is not None else best_thr_f1
        tag    = "sens-constrained" if best_thr_s is not None else "f1-fallback"
        print(f"  {name:6s}: thr={chosen:.2f}  sens={best_sens:.3f}  [{tag}]")
        optimal[name] = chosen

    return optimal


def find_optimal_thresholds_fbeta(
    y_true:       np.ndarray,
    y_pred_proba: np.ndarray,
    label_names:  List[str],
    beta:         float = 0.5,
    min_recall:   Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Encuentra el umbral óptimo por clase maximizando F-beta score.

    F_beta pondera precisión (beta<1) o recall (beta>1). Con beta=0.5
    la precisión vale el doble que el recall, lo que reduce falsos
    positivos manteniendo una sensibilidad aceptable.

    Se puede imponer un recall mínimo por clase (por defecto 0.85 para
    CD/HYP/NORM/STTC y 0.90 para MI) para respetar el criterio clínico
    de seguridad en el infarto.

    Estrategia por clase:
        1. Candidatos en [0.05, 0.95] con paso 0.01
        2. Filtrar los que cumplen min_recall[clase]
        3. Elegir el que maximiza F_beta entre los filtrados
        4. Si ninguno cumple recall mínimo → maximizar F_beta sin restricción

    Args:
        y_true:       Array binario (N, n_classes).
        y_pred_proba: Array de probabilidades (N, n_classes).
        label_names:  Lista de nombres de clase.
        beta:         Parámetro de F-score. Por defecto 0.5 (precision-focused).
        min_recall:   Dict {clase: recall_mínimo}. Si None usa defaults clínicos.

    Returns:
        Diccionario {nombre_clase: threshold_óptimo}.
    """
    # Defaults clínicos: MI y CD son críticas (0.90), el resto 0.85
    _default_min_recall = {
        "CD":   0.85,
        "HYP":  0.80,
        "MI":   0.90,
        "NORM": 0.85,
        "STTC": 0.85,
    }
    if min_recall is None:
        min_recall = _default_min_recall

    beta_sq = beta ** 2
    candidate_thrs = np.linspace(0.05, 0.95, 181)  # paso 0.005
    optimal: Dict[str, float] = {}

    print(f"\n[Threshold v6.1] Optimizando con F-{beta} score "
          f"(precision-focused, beta={beta})...")

    for i, name in enumerate(label_names):
        col   = y_true[:, i]
        proba = y_pred_proba[:, i]
        req_recall = min_recall.get(name, 0.85)

        best_fbeta_constrained = -1.0
        best_thr_constrained   = None
        best_fbeta_free        = -1.0
        best_thr_free          = 0.5

        for thr in candidate_thrs:
            y_pred_i = (proba >= thr).astype(int)
            tp = int(np.sum((col == 1) & (y_pred_i == 1)))
            fp = int(np.sum((col == 0) & (y_pred_i == 1)))
            fn = int(np.sum((col == 1) & (y_pred_i == 0)))

            prec   = tp / (tp + fp + 1e-9)
            rec    = tp / (tp + fn + 1e-9)
            fbeta  = (1 + beta_sq) * prec * rec / (beta_sq * prec + rec + 1e-9)

            if fbeta > best_fbeta_free:
                best_fbeta_free = fbeta
                best_thr_free   = float(thr)

            if rec >= req_recall and fbeta > best_fbeta_constrained:
                best_fbeta_constrained = fbeta
                best_thr_constrained   = float(thr)

        if best_thr_constrained is not None:
            chosen = best_thr_constrained
            tag    = f"F{beta}-constrained (recall≥{req_recall})"
        else:
            chosen = best_thr_free
            tag    = f"F{beta}-free (recall<{req_recall} inalcanzable)"

        print(f"  {name:6s}: thr={chosen:.3f}  F{beta}={best_fbeta_constrained:.3f}  [{tag}]")
        optimal[name] = chosen

    return optimal


# ===========================================================================
# GRÁFICAS DE EVALUACIÓN
# ===========================================================================

def plot_roc_curves(
    y_true:       np.ndarray,
    y_pred_proba: np.ndarray,
    label_names:  List[str],
    plots_dir:    Path,
) -> None:
    """
    Genera la curva ROC macro y las curvas por clase individual.

    Produce dos ficheros PNG:
    - roc_macro.png:     curva ROC promediada (macro, interpolación lineal).
    - roc_por_clase.png: curvas ROC superpuestas de todas las clases.

    Args:
        y_true:       Array binario (N, 23).
        y_pred_proba: Array de probabilidades (N, 23).
        label_names:  Lista de 23 nombres de clase.
        plots_dir:    Directorio de salida.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(label_names)

    fpr_dict: Dict[str, np.ndarray] = {}
    tpr_dict: Dict[str, np.ndarray] = {}
    auc_dict: Dict[str, float]      = {}

    for i, name in enumerate(label_names):
        if y_true[:, i].sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            fpr_dict[name] = fpr
            tpr_dict[name] = tpr
            auc_dict[name] = float(roc_auc_score(y_true[:, i], y_pred_proba[:, i]))
        except ValueError:
            continue

    if not fpr_dict:
        print("[Plots] WARN: no hay clases con muestras positivas; omitiendo ROC.")
        return

    all_fpr  = np.linspace(0, 1, 300)
    mean_tpr = np.zeros_like(all_fpr)
    for name in fpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[name], tpr_dict[name])
    mean_tpr /= len(fpr_dict)
    auc_macro = float(np.mean(list(auc_dict.values())))

    # ── ROC macro ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(all_fpr, mean_tpr,
            label=f"ROC macro (AUC = {auc_macro:.3f})",
            linewidth=2.5, color="royalblue")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de falsos positivos", fontsize=13)
    ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)", fontsize=13)
    ax.set_title("Curva ROC — Media macro", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "roc_macro.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Curva ROC macro guardada → {out}")

    # ── ROC por clase ────────────────────────────────────────────────────────────
    cmap   = plt.get_cmap("tab20", n_classes)
    fig, ax = plt.subplots(figsize=(11, 8))
    for idx, name in enumerate(fpr_dict):
        ax.plot(fpr_dict[name], tpr_dict[name],
                label=f"{name} (AUC={auc_dict[name]:.2f})",
                linewidth=1.2, color=cmap(idx))
    ax.plot(all_fpr, mean_tpr,
            label=f"Macro (AUC={auc_macro:.3f})",
            linewidth=2.5, color="black", linestyle="--")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Tasa de falsos positivos", fontsize=12)
    ax.set_ylabel("Sensibilidad", fontsize=12)
    ax.set_title("Curvas ROC por subclase diagnóstica", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = plots_dir / "roc_por_clase.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Curvas ROC por clase guardadas → {out}")


def plot_confusion_matrix_multilabel(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    label_names: List[str],
    plots_dir:   Path,
) -> None:
    """
    Visualiza las matrices de confusión 2×2 de las 23 clases en rejilla.

    Produce:
    - confusion_matrix.png: rejilla de 5×5 con matrices 2×2 normalizadas
      por fila (porcentaje sobre el total de instancias reales de cada clase).

    Args:
        y_true:      Array binario (N, 23).
        y_pred:      Array binario (N, 23) con umbral ya aplicado.
        label_names: Lista de 23 nombres de clase.
        plots_dir:   Directorio de salida.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    n_classes     = len(label_names)
    nrows, ncols  = 5, 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 16))
    fig.suptitle(
        "Matrices de confusión por subclase diagnóstica\n(valores normalizados por fila)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n_classes:
            ax.axis("off")
            continue
        cm       = conf_matrices[i]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)
        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0, vmax=1,
            cbar=False,
            xticklabels=["Neg pred.", "Pos pred."],
            yticklabels=["Neg real", "Pos real"],
            annot_kws={"size": 8},
        )
        ax.set_title(label_names[i], fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = plots_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Matrices de confusión guardadas → {out}")


def plot_f1_barplot(
    f1_per_class: Dict[str, float],
    label_names:  List[str],
    plots_dir:    Path,
    suffix:       str = "",
) -> None:
    """
    Genera un gráfico de barras horizontal con el F1-Score por clase.

    Las clases se ordenan de menor a mayor F1 (de arriba a abajo).
    Se traza una línea vertical con el F1 macro de referencia.
    Código de color: verde (F1≥0.7), naranja (0.5≤F1<0.7), rojo (F1<0.5).

    Args:
        f1_per_class: Diccionario {nombre_clase: f1_score}.
        label_names:  Lista ordenada de nombres de clase.
        plots_dir:    Directorio de salida.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    nombres  = [n for n in label_names if n in f1_per_class]
    valores  = [f1_per_class[n] for n in nombres]
    f1_macro = float(np.mean(valores))

    order   = np.argsort(valores)
    nombres = [nombres[i] for i in order]
    valores = [valores[i] for i in order]
    colores = [
        "#5cb85c" if v >= 0.7 else "#f0ad4e" if v >= 0.5 else "#d9534f"
        for v in valores
    ]

    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(nombres, valores, color=colores, edgecolor="white", height=0.7)
    ax.axvline(f1_macro, color="navy", linewidth=1.5, linestyle="--",
               label=f"F1 macro = {f1_macro:.3f}")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("F1-Score", fontsize=13)
    ax.set_title("F1-Score por subclase diagnóstica", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = plots_dir / f"f1_por_clase{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Barplot F1 guardado → {out}")


def plot_sensitivity_barplot(
    sens_per_class: Dict[str, float],
    spec_per_class: Dict[str, float],
    label_names:    List[str],
    plots_dir:      Path,
    suffix:         str = "",
) -> None:
    """
    Gráfico de barras agrupadas con Sensibilidad y Especificidad por clase.

    Args:
        sens_per_class: {nombre_clase: sensibilidad}.
        spec_per_class: {nombre_clase: especificidad}.
        label_names:    Lista de nombres de clase.
        plots_dir:      Directorio de salida.
        suffix:         Sufijo para el nombre del fichero (p.ej. '_v2').
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    nombres = [n for n in label_names if n in sens_per_class]
    sens    = [sens_per_class[n] for n in nombres]
    spec    = [spec_per_class[n] for n in nombres]

    x      = np.arange(len(nombres))
    width  = 0.38
    sens_macro = float(np.mean(sens))
    spec_macro = float(np.mean(spec))

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, sens, width, label="Sensibilidad", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x + width / 2, spec, width, label="Especificidad", color="#4CAF50", alpha=0.85)
    ax.axhline(sens_macro, color="#0D47A1", linewidth=1.5, linestyle="--",
               label=f"Sens. macro = {sens_macro:.3f}")
    ax.axhline(MIN_SENSITIVITY, color="red", linewidth=1.2, linestyle=":",
               label=f"Objetivo ≥ {MIN_SENSITIVITY}")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Sensibilidad y Especificidad por clase", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = plots_dir / f"sensibilidad_especificidad{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Sensibilidad/Especificidad guardado → {out}")


def plot_precision_recall_barplot(
    prec_per_class: Dict[str, float],
    sens_per_class: Dict[str, float],
    label_names:    List[str],
    plots_dir:      Path,
) -> None:
    """
    Gráfico de barras agrupadas Precisión vs Sensibilidad por clase.

    Documenta el trade-off clave en diagnóstico médico:
    alta sensibilidad implica menor precisión al bajar el threshold.

    Args:
        prec_per_class: {nombre_clase: precisión}.
        sens_per_class: {nombre_clase: sensibilidad}.
        label_names:    Lista de nombres de clase.
        plots_dir:      Directorio de salida.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    nombres = [n for n in label_names if n in prec_per_class]
    prec    = [prec_per_class[n] for n in nombres]
    sens    = [sens_per_class[n] for n in nombres]

    x     = np.arange(len(nombres))
    width = 0.38
    prec_macro = float(np.mean(prec))
    sens_macro = float(np.mean(sens))

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, prec, width, label="Precisión",    color="#FF7043", alpha=0.85)
    b2 = ax.bar(x + width / 2, sens, width, label="Sensibilidad", color="#2196F3", alpha=0.85)
    ax.axhline(prec_macro, color="#BF360C", linewidth=1.5, linestyle="--",
               label=f"Prec. macro = {prec_macro:.3f}")
    ax.axhline(sens_macro, color="#0D47A1", linewidth=1.5, linestyle="-.",
               label=f"Sens. macro = {sens_macro:.3f}")
    ax.axhline(MIN_SENSITIVITY, color="green", linewidth=1.2, linestyle=":",
               label=f"Objetivo sens ≥ {MIN_SENSITIVITY}")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylabel("Valor", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "Precisión vs Sensibilidad por clase\n"
        "(trade-off inherente al contexto de cribado médico)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "precision_vs_sensibilidad.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Precisión vs Sensibilidad guardado → {out}")


def plot_class_distribution(
    train_labels: np.ndarray,
    val_labels:   np.ndarray,
    test_labels:  np.ndarray,
    label_names:  List[str],
    plots_dir:    Path,
) -> None:
    """
    Visualiza la proporción de positivos de cada clase en train/val/test.

    Permite detectar visualmente si el split estratificado mantiene
    una distribución de clases homogénea entre los tres subconjuntos.

    Args:
        train_labels: Array binario (N_train, 23).
        val_labels:   Array binario (N_val,   23).
        test_labels:  Array binario (N_test,  23).
        label_names:  Lista de 23 nombres de clase.
        plots_dir:    Directorio de salida.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    splits  = {
        "Entrenamiento": train_labels,
        "Validación":    val_labels,
        "Test":          test_labels,
    }
    width   = 0.27
    x       = np.arange(len(label_names))
    colores = {"Entrenamiento": "#4C72B0", "Validación": "#DD8452", "Test": "#55A868"}

    fig, ax = plt.subplots(figsize=(16, 6))
    for j, (split_name, labels) in enumerate(splits.items()):
        proporciones = labels.mean(axis=0) * 100
        ax.bar(x + (j - 1) * width, proporciones, width,
               label=split_name, color=colores[split_name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Proporción de positivos (%)", fontsize=12)
    ax.set_title(
        "Distribución de clases por split (entrenamiento / validación / test)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "distribucion_clases.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Distribución de clases guardada → {out}")


# ===========================================================================
# CARGA DEL MODELO
# ===========================================================================

def load_saved_model(
    model_path: Optional[Path] = None,
) -> tf.keras.Model:
    """
    Carga el modelo guardado en formato SavedModel.

    Args:
        model_path: Ruta al directorio del modelo.
                    Por defecto SAVED_MODEL_DIR/ecg_model.

    Returns:
        Modelo Keras cargado y listo para predict().

    Raises:
        FileNotFoundError: Si el directorio no existe.
    """
    if model_path is None:
        model_path = SAVED_MODEL_DIR / "ecg_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. "
            "Ejecuta main.py para entrenar primero."
        )
    model = tf.keras.models.load_model(str(model_path))
    print(f"[Evaluate] Modelo cargado desde {model_path}")
    return model


# ===========================================================================
# PREDICCIONES
# ===========================================================================

def get_predictions(
    model:   tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> np.ndarray:
    """
    Genera predicciones probabilísticas sobre el dataset de test.

    Args:
        model:   Modelo Keras cargado.
        test_ds: tf.data.Dataset de test (sin sample_weights).

    Returns:
        Array float32 de shape (N, 23) con probabilidades [0, 1].
    """
    predictions = model.predict(test_ds, verbose=1)
    return predictions.astype(np.float32)


# ===========================================================================
# MÉTRICAS
# ===========================================================================

def compute_all_metrics(
    y_true:       np.ndarray,
    y_pred_proba: np.ndarray,
    label_names:  List[str],
    threshold:    float = THRESHOLD,
) -> Dict:
    """
    Calcula el conjunto completo de métricas de evaluación.

    Métricas incluidas:
    - AUC-ROC macro y por clase (solo para clases con muestras positivas)
    - F1-Score macro y por clase
    - Sensibilidad (Recall) macro y por clase
    - Especificidad macro y por clase
    - Matriz de confusión multilabel (lista 3D serializable)
    - Informe de clasificación completo (classification_report)
    - Flag de si se alcanzó el objetivo mínimo de sensibilidad (≥0.90)

    Args:
        y_true:       Array binario (N, 23) de etiquetas reales.
        y_pred_proba: Array (N, 23) de probabilidades predichas.
        label_names:  Lista de 23 nombres de subclase.
        threshold:    Umbral de binarización. Por defecto 0.5.

    Returns:
        Diccionario anidado con todas las métricas.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # ── AUC-ROC ─────────────────────────────────────────────────────────────
    auc_per_class: Dict[str, Optional[float]] = {}
    for i, name in enumerate(label_names):
        if y_true[:, i].sum() > 0:
            try:
                auc_per_class[name] = float(
                    roc_auc_score(y_true[:, i], y_pred_proba[:, i])
                )
            except ValueError:
                auc_per_class[name] = None
        else:
            auc_per_class[name] = None

    valid_aucs  = [v for v in auc_per_class.values() if v is not None]
    auc_macro   = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # ── F1-Score ─────────────────────────────────────────────────────────────
    f1_arr   = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_per_class = {name: float(f1_arr[i]) for i, name in enumerate(label_names)}

    # ── Precisión ────────────────────────────────────────────────────────────
    prec_arr   = precision_score(y_true, y_pred, average=None, zero_division=0)
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_per_class = {name: float(prec_arr[i]) for i, name in enumerate(label_names)}

    # ── Sensibilidad (Recall) ────────────────────────────────────────────────
    recall_arr   = recall_score(y_true, y_pred, average=None, zero_division=0)
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    recall_per_class = {name: float(recall_arr[i]) for i, name in enumerate(label_names)}

    # ── Especificidad ────────────────────────────────────────────────────────
    spec_per_class, spec_macro = compute_specificity(y_true, y_pred, label_names)

    # ── Matriz de confusión multilabel ───────────────────────────────────────
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred).tolist()

    # ── Informe completo ─────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    # ── Comprobación objetivo clínico ────────────────────────────────────────
    sensitivity_ok = recall_macro >= MIN_SENSITIVITY
    status_msg = "✓ OBJETIVO ALCANZADO" if sensitivity_ok else f"✗ Por debajo del objetivo (≥{MIN_SENSITIVITY})"
    print(f"[Evaluate] Sensibilidad macro: {recall_macro:.4f}  {status_msg}")
    print(f"[Evaluate] Precisión macro:    {prec_macro:.4f}")

    return {
        "threshold": threshold,
        "auc_roc": {
            "macro":     auc_macro,
            "per_class": auc_per_class,
        },
        "f1_score": {
            "macro":     f1_macro,
            "per_class": f1_per_class,
        },
        "precision": {
            "macro":     prec_macro,
            "per_class": prec_per_class,
        },
        "sensitivity_recall": {
            "macro":           recall_macro,
            "per_class":       recall_per_class,
            "target_achieved": sensitivity_ok,
            "target_value":    MIN_SENSITIVITY,
        },
        "specificity": {
            "macro":     spec_macro,
            "per_class": spec_per_class,
        },
        "confusion_matrix_multilabel": conf_matrix,
        "classification_report":       report,
    }


# ===========================================================================
# PIPELINE COMPLETO DE EVALUACIÓN
# ===========================================================================

def evaluate_model(
    test_ds:      tf.data.Dataset,
    y_true:       np.ndarray,
    label_names:  List[str],
    model:        Optional[tf.keras.Model] = None,
    train_labels: Optional[np.ndarray] = None,
    val_labels:   Optional[np.ndarray] = None,
    val_ds:       Optional[tf.data.Dataset] = None,
    val_true:     Optional[np.ndarray] = None,
    results_dir:  Optional[Path] = None,
    saved_model_dir: Optional[Path] = None,
    threshold_beta: Optional[float] = None,
) -> Dict:
    """
    Pipeline completo de evaluación sobre el conjunto de test.

    Pasos:
    1. Cargar modelo si no se pasa como argumento
    2. Generar predicciones probabilísticas (test y, opcionalmente, val)
    3. Calcular métricas con threshold=0.5 (baseline)
    4. Si se pasan val_ds y val_true: optimizar thresholds por clase en val
       y re-evaluar test con thresholds óptimos
    5. Guardar resultados en results/metrics.json (thresholds óptimos)
       y results/metrics_baseline.json (threshold=0.5)
    6. Guardar thresholds óptimos en saved_model/optimal_thresholds.json
    7. Generar gráficas

    Args:
        test_ds:      tf.data.Dataset de test (sin sample_weights).
        y_true:       Array binario (N, 5) de etiquetas reales de test.
        label_names:  Lista de 5 nombres de superclase.
        model:        Modelo cargado (opcional).
        train_labels: Array (N_train, 5) para gráfica de distribución.
        val_labels:   Array (N_val, 5) para gráfica de distribución.
        val_ds:       tf.data.Dataset de validación para optimizar thresholds.
        val_true:     Array (N_val, 5) etiquetas reales de validación.

    Returns:
        Diccionario completo con métricas (con thresholds optimizados si
        se proporciona val_ds, o con threshold=0.5 en caso contrario).
    """
    # 1. Cargar modelo
    if model is None:
        model = load_saved_model()

    # Directorios: usar los pasados por argumento o los de módulo por defecto
    _results_dir     = results_dir     if results_dir     is not None else RESULTS_DIR
    _saved_model_dir = saved_model_dir if saved_model_dir is not None else SAVED_MODEL_DIR
    _plots_dir       = _results_dir / "plots"

    # 2. Predicciones en test
    print("[Evaluate] Generando predicciones sobre test set...")
    y_pred_proba = get_predictions(model, test_ds)

    # 3. Métricas baseline (threshold = 0.5)
    print("[Evaluate] Calculando métricas baseline (thr=0.5)...")
    metrics_baseline = compute_all_metrics(y_true, y_pred_proba, label_names, threshold=0.5)

    # Guardar métricas baseline
    _results_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = _results_dir / "metrics_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(metrics_baseline, f, indent=2, ensure_ascii=False)
    print(f"[Evaluate] Métricas baseline guardadas → {baseline_path}")

    # 4. Optimización de thresholds en validación
    optimal_thresholds: Optional[Dict[str, float]] = None
    if val_ds is not None and val_true is not None:
        print("[Evaluate] Generando predicciones sobre val set...")
        y_val_proba = get_predictions(model, val_ds)
        if threshold_beta is not None:
            print(f"[Evaluate] Usando estrategia F{threshold_beta}-score (v6.x)...")
            optimal_thresholds = find_optimal_thresholds_fbeta(
                val_true, y_val_proba, label_names, beta=threshold_beta
            )
        else:
            optimal_thresholds = find_optimal_thresholds(val_true, y_val_proba, label_names)

        # Guardar thresholds óptimos
        thr_path = _saved_model_dir / "optimal_thresholds.json"
        _saved_model_dir.mkdir(parents=True, exist_ok=True)
        with open(thr_path, "w", encoding="utf-8") as f:
            json.dump(optimal_thresholds, f, indent=2, ensure_ascii=False)
        print(f"[Evaluate] Thresholds óptimos guardados → {thr_path}")

    # 5. Métricas con thresholds óptimos
    if optimal_thresholds is not None:
        print("[Evaluate] Calculando métricas con thresholds óptimos...")
        # Binarizar con threshold óptimo por clase
        thr_arr   = np.array([optimal_thresholds[n] for n in label_names], dtype=np.float32)
        y_pred_opt = (y_pred_proba >= thr_arr).astype(int)

        # compute_all_metrics necesita un scalar, así que calculamos manualmente
        # las métricas usando y_pred ya binarizado
        from sklearn.metrics import (
            roc_auc_score, f1_score as _f1, recall_score as _recall,
            precision_score as _prec,
            multilabel_confusion_matrix, classification_report,
        )

        auc_per_class: Dict[str, Optional[float]] = {}
        for i, name in enumerate(label_names):
            if y_true[:, i].sum() > 0:
                try:
                    auc_per_class[name] = float(roc_auc_score(y_true[:, i], y_pred_proba[:, i]))
                except ValueError:
                    auc_per_class[name] = None
            else:
                auc_per_class[name] = None
        valid_aucs = [v for v in auc_per_class.values() if v is not None]
        auc_macro  = float(np.mean(valid_aucs)) if valid_aucs else 0.0

        f1_arr       = _f1(y_true, y_pred_opt, average=None, zero_division=0)
        f1_macro     = float(_f1(y_true, y_pred_opt, average="macro", zero_division=0))
        f1_per_class = {name: float(f1_arr[i]) for i, name in enumerate(label_names)}

        prec_arr      = _prec(y_true, y_pred_opt, average=None, zero_division=0)
        prec_macro    = float(_prec(y_true, y_pred_opt, average="macro", zero_division=0))
        prec_per_class = {name: float(prec_arr[i]) for i, name in enumerate(label_names)}

        recall_arr   = _recall(y_true, y_pred_opt, average=None, zero_division=0)
        recall_macro = float(_recall(y_true, y_pred_opt, average="macro", zero_division=0))
        recall_per_class = {name: float(recall_arr[i]) for i, name in enumerate(label_names)}

        spec_per_class, spec_macro = compute_specificity(y_true, y_pred_opt, label_names)
        conf_matrix = multilabel_confusion_matrix(y_true, y_pred_opt).tolist()
        report      = classification_report(
            y_true, y_pred_opt, target_names=label_names, zero_division=0, output_dict=True
        )

        sensitivity_ok = recall_macro >= MIN_SENSITIVITY
        status_msg = "✓ OBJETIVO ALCANZADO" if sensitivity_ok else f"✗ Por debajo del objetivo (≥{MIN_SENSITIVITY})"
        print(f"[Evaluate] Sensibilidad macro (opt): {recall_macro:.4f}  {status_msg}")
        print(f"[Evaluate] Precisión macro   (opt): {prec_macro:.4f}")

        metrics = {
            "thresholds": optimal_thresholds,
            "auc_roc": {"macro": auc_macro, "per_class": auc_per_class},
            "f1_score": {"macro": f1_macro, "per_class": f1_per_class},
            "precision": {"macro": prec_macro, "per_class": prec_per_class},
            "sensitivity_recall": {
                "macro":           recall_macro,
                "per_class":       recall_per_class,
                "target_achieved": sensitivity_ok,
                "target_value":    MIN_SENSITIVITY,
            },
            "specificity": {"macro": spec_macro, "per_class": spec_per_class},
            "confusion_matrix_multilabel": conf_matrix,
            "classification_report":       report,
        }
        suffix = "_opt"
    else:
        metrics = metrics_baseline
        suffix  = ""

    # 6. Guardar métricas principales
    metrics_path = _results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[Evaluate] Métricas guardadas → {metrics_path}")

    # 7. Gráficas con thresholds seleccionados
    print("[Evaluate] Generando gráficas de evaluación...")
    y_pred_bin = (
        (y_pred_proba >= np.array([optimal_thresholds[n] for n in label_names], dtype=np.float32)).astype(int)
        if optimal_thresholds is not None
        else (y_pred_proba >= THRESHOLD).astype(int)
    )

    plot_roc_curves(y_true, y_pred_proba, label_names, _plots_dir)
    plot_confusion_matrix_multilabel(y_true, y_pred_bin, label_names, _plots_dir)
    plot_f1_barplot(metrics["f1_score"]["per_class"], label_names, _plots_dir, suffix=suffix)
    plot_sensitivity_barplot(
        metrics["sensitivity_recall"]["per_class"],
        metrics["specificity"]["per_class"],
        label_names, _plots_dir, suffix=suffix,
    )
    if optimal_thresholds is not None and "precision" in metrics:
        plot_precision_recall_barplot(
            metrics["precision"]["per_class"],
            metrics["sensitivity_recall"]["per_class"],
            label_names, _plots_dir,
        )
    if train_labels is not None and val_labels is not None:
        plot_class_distribution(train_labels, val_labels, y_true, label_names, _plots_dir)

    # Si también tenemos baseline, guardar gráficas baseline para comparación
    if optimal_thresholds is not None:
        y_pred_base = (y_pred_proba >= THRESHOLD).astype(int)
        plot_f1_barplot(metrics_baseline["f1_score"]["per_class"], label_names, _plots_dir, suffix="_base")
        plot_sensitivity_barplot(
            metrics_baseline["sensitivity_recall"]["per_class"],
            metrics_baseline["specificity"]["per_class"],
            label_names, _plots_dir, suffix="_base",
        )

    # 8. Resumen en consola
    sep = "=" * 52
    print(f"\n{sep}")
    print("  RESUMEN DE EVALUACIÓN — TEST SET")
    print(sep)
    if optimal_thresholds is not None:
        bsl = metrics_baseline
        print("  [Baseline thr=0.50]")
        print(f"  AUC-ROC macro       : {bsl['auc_roc']['macro']:.4f}")
        print(f"  Precisión macro     : {bsl['precision']['macro']:.4f}")
        print(f"  F1-Score macro      : {bsl['f1_score']['macro']:.4f}")
        print(f"  Sensibilidad macro  : {bsl['sensitivity_recall']['macro']:.4f}")
        print(f"  Especificidad macro : {bsl['specificity']['macro']:.4f}")
        print(f"\n  [Thresholds óptimos por clase]")
    print(f"  AUC-ROC macro       : {metrics['auc_roc']['macro']:.4f}")
    prec_str = f"{metrics['precision']['macro']:.4f}" if 'precision' in metrics else "N/A"
    print(f"  Precisión macro     : {prec_str}")
    print(f"  F1-Score macro      : {metrics['f1_score']['macro']:.4f}")
    print(f"  Sensibilidad macro  : {metrics['sensitivity_recall']['macro']:.4f}")
    status = "✓" if metrics["sensitivity_recall"]["target_achieved"] else "✗"
    print(f"  Especificidad macro : {metrics['specificity']['macro']:.4f}")
    print(f"  Objetivo sens≥{MIN_SENSITIVITY} : {status}")
    print(sep + "\n")

    return metrics
