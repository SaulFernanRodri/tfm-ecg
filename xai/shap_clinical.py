"""
SHAP para variables clínicas del modelo ECG multimodal.

Usa SHAP KernelExplainer sobre la rama clínica del modelo para
calcular la contribución de cada variable (edad, sexo, altura, peso)
a la predicción final.

KernelExplainer es agnóstico al modelo y funciona con cualquier función
de predicción, lo que lo hace adecuado para el modelo multimodal donde
la rama clínica y la ECG están fusionadas.

La estrategia es:
1. Fijar las señales ECG (usar la media del batch de referencia)
2. Perturbar solo las variables clínicas
3. Calcular SHAP values sobre esa función parcial

Referencia:
    Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions.
    NeurIPS. https://arxiv.org/abs/1705.07874
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import shap
import tensorflow as tf


CLINICAL_FEATURE_NAMES: List[str] = ["age", "sex", "height", "weight"]


def _build_predict_fn(
    model:           tf.keras.Model,
    ecg_background:  np.ndarray,
    class_idx:       Optional[int] = None,
):
    """
    Construye una función de predicción que acepta solo variables clínicas,
    usando el ECG de referencia (media del background) como señal fija.

    Args:
        model:          Modelo Keras (dos entradas: ECG + clínica).
        ecg_background: Array (N, T, 12) background ECG. Se usa la media.
        class_idx:      Si se indica, la función devuelve proba de esa clase.
                        Si None, devuelve todas las clases (N, n_classes).

    Returns:
        predict_fn: función callable que acepta (N, n_clinical) y devuelve (N,) o (N, C).
    """
    # ECG fijo: media del background
    ecg_mean = np.mean(ecg_background, axis=0, keepdims=True)  # (1, T, 12)

    def predict_fn(clinical_data: np.ndarray) -> np.ndarray:
        n = len(clinical_data)
        ecg_tiled = np.tile(ecg_mean, (n, 1, 1))  # (N, T, 12)
        ecg_t   = tf.convert_to_tensor(ecg_tiled, dtype=tf.float32)
        clin_t  = tf.convert_to_tensor(clinical_data, dtype=tf.float32)
        probas  = model([ecg_t, clin_t], training=False).numpy()
        if class_idx is not None:
            return probas[:, class_idx]
        return probas

    return predict_fn


def compute_shap_values(
    model:              tf.keras.Model,
    ecg_background:     np.ndarray,
    clinical_background: np.ndarray,
    clinical_samples:   np.ndarray,
    class_idx:          int,
    n_background:       int = 50,
) -> np.ndarray:
    """
    Calcula SHAP values para variables clínicas usando KernelExplainer.

    Args:
        model:               Modelo Keras.
        ecg_background:      Array (N, T, 12) — conjunto de referencia (train).
        clinical_background: Array (N, n_clinical) — variables clínicas de referencia.
        clinical_samples:    Array (M, n_clinical) — muestras a explicar.
        class_idx:           Índice de la clase objetivo.
        n_background:        Número de muestras de background para KernelExplainer.
                             Menos muestras = más rápido pero menos preciso.

    Returns:
        shap_values: Array (M, n_clinical) con SHAP values.
    """
    # Subsamplear background para eficiencia
    idx = np.random.choice(len(ecg_background), size=min(n_background, len(ecg_background)), replace=False)
    ecg_bg  = ecg_background[idx]
    clin_bg = clinical_background[idx]

    predict_fn = _build_predict_fn(model, ecg_bg, class_idx=class_idx)

    explainer   = shap.KernelExplainer(predict_fn, clin_bg)
    shap_values = explainer.shap_values(clinical_samples, nsamples=100, silent=True)

    return shap_values


def compute_shap_all_classes(
    model:               tf.keras.Model,
    ecg_background:      np.ndarray,
    clinical_background: np.ndarray,
    clinical_samples:    np.ndarray,
    label_names:         List[str],
    n_background:        int = 50,
) -> Dict[str, np.ndarray]:
    """
    Calcula SHAP values para todas las clases.

    Args:
        model:               Modelo Keras.
        ecg_background:      Array (N, T, 12).
        clinical_background: Array (N, n_clinical).
        clinical_samples:    Array (M, n_clinical).
        label_names:         Lista de nombres de clase.
        n_background:        Número de muestras de background.

    Returns:
        Diccionario {clase: shap_values(M, n_clinical)}.
    """
    return {
        name: compute_shap_values(
            model, ecg_background, clinical_background,
            clinical_samples, class_idx=i, n_background=n_background,
        )
        for i, name in enumerate(label_names)
    }


def mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    """
    Calcula la importancia media como |SHAP| promediado sobre muestras.

    Args:
        shap_values: Array (M, n_features).

    Returns:
        importances: Array (n_features,) — importancia media por feature.
    """
    return np.mean(np.abs(shap_values), axis=0)
