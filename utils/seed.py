"""
Módulo para el fijado global de semillas aleatorias.

Garantiza reproducibilidad completa en Python, NumPy y TensorFlow.
Debe llamarse al inicio del pipeline, antes de cualquier operación aleatoria.
"""

import os
import random
import numpy as np
import tensorflow as tf

# Semilla global del proyecto
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """
    Fija las semillas aleatorias en Python, NumPy y TensorFlow.

    Establece la reproducibilidad en:
    - random (módulo estándar de Python)
    - PYTHONHASHSEED (variable de entorno para hash determinista)
    - numpy.random
    - tf.random

    Args:
        seed: Valor de la semilla. Por defecto 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Seed] Semillas fijadas con seed={seed}")
