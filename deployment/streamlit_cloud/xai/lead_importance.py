"""
Lead Importance Score mediante ablación de derivaciones.

Para cada derivación (lead), pone a cero esa derivación en la señal
y mide la caída de AUC o de la probabilidad predicha. Las derivaciones
cuya ablación más perjudica al modelo son las más importantes.

Este enfoque es agnóstico al modelo (model-agnostic) y directamente
interpretable clínicamente: un cardiólogo sabe qué ve cada lead.

Derivaciones del ECG estándar de 12 leads (orden PTB-XL):
    0=I, 1=II, 2=III, 3=aVR, 4=aVL, 5=aVF,
    6=V1, 7=V2, 8=V3, 9=V4, 10=V5, 11=V6
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


LEAD_NAMES: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"]


def _predict_proba(
    model:          tf.keras.Model,
    ecg:            np.ndarray,
    clinical:       np.ndarray,
) -> np.ndarray:
    """
    Obtiene las probabilidades de salida para un ECG.

    Args:
        model:    Modelo Keras (dos entradas).
        ecg:      Array (T, 12) sin batch dim.
        clinical: Array (n_features,) sin batch dim.

    Returns:
        probas: Array (n_classes,) con probabilidades en [0, 1].
    """
    ecg_t = tf.convert_to_tensor(ecg[np.newaxis], dtype=tf.float32)
    clin_t = tf.convert_to_tensor(clinical[np.newaxis], dtype=tf.float32)
    return model([ecg_t, clin_t], training=False).numpy()[0]


def compute_lead_importance_single(
    model:           tf.keras.Model,
    ecg_sample:      np.ndarray,
    clinical_sample: np.ndarray,
    class_idx:       Optional[int] = None,
) -> np.ndarray:
    """
    Calcula la importancia de cada lead por ablación para una muestra.

    Para cada lead k:
        ablated_ecg = copia con lead k puesta a cero
        drop_k = proba_original - proba_ablada

    Una caída positiva indica que el lead contribuye positivamente.

    Args:
        model:           Modelo Keras.
        ecg_sample:      Array (T, 12).
        clinical_sample: Array (n_features,).
        class_idx:       Si se indica, devuelve importancia solo para esa clase.
                         Si None, devuelve la media sobre todas las clases.

    Returns:
        importances: Array (12,) con la importancia de cada lead.
    """
    baseline_proba = _predict_proba(model, ecg_sample, clinical_sample)

    importances = np.zeros(12, dtype=np.float32)
    for lead_idx in range(12):
        ablated = ecg_sample.copy()
        ablated[:, lead_idx] = 0.0
        ablated_proba = _predict_proba(model, ablated, clinical_sample)

        if class_idx is not None:
            drop = baseline_proba[class_idx] - ablated_proba[class_idx]
        else:
            drop = np.mean(baseline_proba - ablated_proba)

        importances[lead_idx] = drop

    return importances


def compute_lead_importance_batch(
    model:          tf.keras.Model,
    ecg_batch:      np.ndarray,
    clinical_batch: np.ndarray,
    class_idx:      Optional[int] = None,
) -> np.ndarray:
    """
    Calcula lead importance promedio sobre un conjunto de muestras.

    Args:
        model:          Modelo Keras.
        ecg_batch:      Array (N, T, 12).
        clinical_batch: Array (N, n_features).
        class_idx:      Índice de la clase objetivo, o None para media.

    Returns:
        mean_importances: Array (12,) promediado sobre las N muestras.
    """
    all_importances = []
    for i in range(len(ecg_batch)):
        imp = compute_lead_importance_single(
            model, ecg_batch[i], clinical_batch[i], class_idx=class_idx
        )
        all_importances.append(imp)
    return np.mean(all_importances, axis=0)


def compute_lead_importance_per_class(
    model:          tf.keras.Model,
    ecg_batch:      np.ndarray,
    clinical_batch: np.ndarray,
    label_names:    List[str],
) -> Dict[str, np.ndarray]:
    """
    Calcula lead importance por clase, promediado sobre un batch.

    Args:
        model:          Modelo Keras.
        ecg_batch:      Array (N, T, 12).
        clinical_batch: Array (N, n_features).
        label_names:    Lista de nombres de clase.

    Returns:
        Diccionario {clase: array(12,)} con importancias por lead y clase.
    """
    return {
        name: compute_lead_importance_batch(
            model, ecg_batch, clinical_batch, class_idx=i
        )
        for i, name in enumerate(label_names)
    }
