"""
Funciones auxiliares para el cálculo de métricas de evaluación.

Complementa evaluation/evaluate.py con funciones específicas
para métricas que scikit-learn no ofrece directamente en
configuración multilabel, como la especificidad por clase.
"""

import numpy as np
from sklearn.metrics import (
    multilabel_confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
)
from typing import Dict, List, Tuple


def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Tuple[Dict[str, float], float]:
    """
    Calcula la especificidad por clase y macro para clasificación multilabel.

    La especificidad (True Negative Rate) mide la proporción de
    verdaderos negativos sobre el total de negativos reales:
        Especificidad = TN / (TN + FP)

    Se obtiene directamente de la matriz de confusión multilabel
    que devuelve sklearn, donde para cada clase:
        conf_matrix[i] = [[TN, FP], [FN, TP]]

    Args:
        y_true:      Array binario de shape (N, C) con etiquetas reales.
        y_pred:      Array binario de shape (N, C) con predicciones.
        label_names: Lista de C nombres de clase.

    Returns:
        Tupla (specificity_per_class, specificity_macro) donde:
        - specificity_per_class: dict {nombre_clase: float}
        - specificity_macro:     float con la media macro
    """
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    specificity_per_class: Dict[str, float] = {}

    for i, name in enumerate(label_names):
        tn = conf_matrices[i][0, 0]
        fp = conf_matrices[i][0, 1]
        denominator = tn + fp
        specificity_per_class[name] = float(tn / denominator) if denominator > 0 else 0.0

    specificity_macro = float(np.mean(list(specificity_per_class.values())))
    return specificity_per_class, specificity_macro


def compute_macro_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    """
    Calcula un resumen de métricas macro para logging rápido.

    Args:
        y_true:      Array binario (N, C).
        y_pred:      Array binario (N, C).
        label_names: Lista de C nombres de clase.

    Returns:
        Diccionario con f1_macro, recall_macro, precision_macro,
        specificity_macro.
    """
    _, specificity_macro = compute_specificity(y_true, y_pred, label_names)

    return {
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": specificity_macro,
    }
