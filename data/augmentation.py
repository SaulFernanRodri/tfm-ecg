"""
Módulo de augmentación de señales ECG para entrenamiento.

Implementa cuatro tipos de aumentación de datos sobre señales 12-lead,
aplicados en el pipeline tf.data.Dataset durante entrenamiento.

Aumentaciones estándar (ampliamente validadas):
  1. Amplitude scaling — escalado aleatorio de toda la señal
  2. Gaussian noise    — ruido aditivo gaussiano (simula artefactos)
  3. Time shift        — desplazamiento circular aleatorio

Aumentación innovadora:
  4. Lead masking      — enmascara aleatoriamente una derivación
     Justificación clínica: en la práctica clínica es habitual que
     algún electrodo tenga mala colocación o artefacto de movimiento.
     Entrenando con lead masking, el modelo aprende a usar la
     redundancia inherente del sistema de 12 derivaciones (muchas son
     combinaciones lineales del triángulo de Einthoven) y se vuelve
     robusto a derivaciones parcialmente informativas.
     Beneficio XAI: al comparar predicciones con/sin cada derivación
     enmascarada, se obtiene directamente un mapa de importancia por
     lead sin necesidad de algoritmos de atribución adicionales.

Diseño de la pipeline:
  - Todas las operaciones son funciones TensorFlow puras, ejecutables
    dentro de tf.data.Dataset.map() con paralelismo AUTOTUNE.
  - Cada aumentación tiene probabilidad p configurable para que el
    modelo también vea ejemplos sin aumentar (mezcla de distribuciones).
  - Los hiperparámetros están elegidos conservadoramente para no
    distorsionar morfologías diagnósticas.

Referencias:
  Kiyasseh et al. (2021): CLOCS: Contrastive Learning of Cardiac Signals.
    ICML 2021. (usa time shift + amplitude scale para ECG)
  Um et al. (2017): Data augmentation of wearable sensor data for
    Parkinson's disease monitoring. PETRA 2017.
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------
_SCALE_MIN:        float = 0.8    # Factor mínimo de escalado de amplitud
_SCALE_MAX:        float = 1.2    # Factor máximo de escalado de amplitud
_NOISE_STD_MAX:    float = 0.05   # Std máxima del ruido gaussiano (en unidades normalizadas)
_MAX_SHIFT_FRAC:   float = 0.05   # Fracción máxima de desplazamiento temporal (±5%)
_LEAD_MASK_PROB:   float = 0.15   # Probabilidad de enmascarar una derivación
_AUG_PROB:         float = 0.6    # Probabilidad de aplicar cada aumentación individual


# ---------------------------------------------------------------------------
# Funciones de aumentación individuales (TensorFlow)
# ---------------------------------------------------------------------------

def _amplitude_scaling(ecg: tf.Tensor) -> tf.Tensor:
    """
    Escala aleatoriamente la amplitud de toda la señal ECG.

    Simula variaciones de impedancia de los electrodos y diferencias
    de ganancia entre dispositivos ECG. El factor se aplica de forma
    uniforme a todas las derivaciones para preservar las relaciones
    relativas de amplitud diagnósticas.

    Args:
        ecg: Tensor (T, 12).

    Returns:
        Tensor (T, 12) escalado.
    """
    scale = tf.random.uniform([], _SCALE_MIN, _SCALE_MAX)
    return ecg * scale


def _gaussian_noise(ecg: tf.Tensor) -> tf.Tensor:
    """
    Añade ruido gaussiano aditivo a la señal ECG.

    Simula artefactos de músculo esquelético (EMG) y ruido eléctrico
    de red. La std máxima de 0.05 en unidades normalizadas corresponde
    a un SNR ≈ 26 dB, manteniendo la morfología diagnóstica legible.

    Args:
        ecg: Tensor (T, 12).

    Returns:
        Tensor (T, 12) con ruido añadido.
    """
    noise_std = tf.random.uniform([], 0.0, _NOISE_STD_MAX)
    noise = tf.random.normal(tf.shape(ecg), stddev=noise_std)
    return ecg + noise


def _time_shift(ecg: tf.Tensor) -> tf.Tensor:
    """
    Desplaza la señal circularmente en el eje temporal.

    El desplazamiento circular (wrap-around) evita introducir bordes
    artificiales. Simula variaciones en el tiempo de inicio de captura
    del ECG respecto al ciclo cardíaco.

    El límite del ±5% a 500 Hz equivale a ±250 muestras = ±500 ms,
    suficiente para variar la fase del ciclo sin perder latidos completos.

    Args:
        ecg: Tensor (T, 12).

    Returns:
        Tensor (T, 12) desplazado circularmente.
    """
    t = tf.shape(ecg)[0]
    max_shift = tf.cast(tf.cast(t, tf.float32) * _MAX_SHIFT_FRAC, tf.int32)
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    return tf.roll(ecg, shift=shift, axis=0)


def _lead_masking(ecg: tf.Tensor) -> tf.Tensor:
    """
    Enmascara (pone a cero) una derivación aleatoria de la señal.

    INNOVACIÓN respecto al benchmark de Strodthoff et al. (2021):
    Este tipo de aumentación no se usa en ninguno de los modelos del
    benchmark PTB-XL. Tiene tres ventajas:

    1. Regularización: el modelo no puede depender de una sola
       derivación, forzándolo a explotar la redundancia del sistema
       12-lead (las derivaciones de extremidades son combinaciones
       lineales entre sí según la ley de Kirchhoff para ECG).

    2. Robustez clínica: en la práctica hospitalaria es habitual
       tener 1-2 derivaciones con artefacto severo o electrodo
       despegado. Un modelo robusto a lead masking seguirá
       funcionando en estas condiciones.

    3. XAI sinérgico: el mismo mecanismo de masking que se usa
       en entrenamiento se puede aplicar en inferencia para medir
       la caída de AUC al eliminar cada derivación, produciendo
       un "Lead Importance Score" complementario al Grad-CAM.

    Args:
        ecg: Tensor (T, 12).

    Returns:
        Tensor (T, 12) con una derivación puesta a cero.
    """
    # Solo aplicar con probabilidad _LEAD_MASK_PROB
    if_mask = tf.random.uniform([]) < _LEAD_MASK_PROB
    lead_idx = tf.random.uniform([], 0, 12, dtype=tf.int32)

    def mask_one_lead():
        # Construir máscara de derivaciones: 1.0 en todas menos lead_idx
        mask = tf.ones([12], dtype=ecg.dtype)
        indices = tf.expand_dims(tf.expand_dims(lead_idx, 0), 1)
        updates = tf.zeros([1, 1], dtype=ecg.dtype)
        # Crear un tensor de 1s con un 0 en la posición lead_idx
        mask = tf.tensor_scatter_nd_update(mask, [[lead_idx]], [0.0])
        # Broadcast: (12,) → (1, 12) → multiplicar con (T, 12)
        return ecg * tf.expand_dims(mask, 0)

    return tf.cond(if_mask, mask_one_lead, lambda: ecg)


# ---------------------------------------------------------------------------
# Función principal de aumentación (combinada)
# ---------------------------------------------------------------------------

def augment_ecg(ecg: tf.Tensor, training: bool = True) -> tf.Tensor:
    """
    Aplica la pipeline completa de aumentación a una señal ECG.

    Cada aumentación (excepto lead masking, con su propia prob.) se
    aplica con probabilidad _AUG_PROB para garantizar que una fracción
    de los ejemplos de entrenamiento lleguen sin aumentar, evitando
    el distribution shift severo.

    Las aumentaciones se aplican en orden:
        1. Amplitude scaling  (prob=_AUG_PROB)
        2. Gaussian noise     (prob=_AUG_PROB)
        3. Time shift         (prob=_AUG_PROB)
        4. Lead masking       (prob=_LEAD_MASK_PROB, siempre evaluado)

    Args:
        ecg:      Tensor (T, 12) ya normalizado.
        training: Si False, devuelve ecg sin modificar.

    Returns:
        Tensor (T, 12) aumentado (o el original si training=False).
    """
    if not training:
        return ecg

    # Aumentación 1: escalado de amplitud
    ecg = tf.cond(
        tf.random.uniform([]) < _AUG_PROB,
        lambda: _amplitude_scaling(ecg),
        lambda: ecg,
    )

    # Aumentación 2: ruido gaussiano
    ecg = tf.cond(
        tf.random.uniform([]) < _AUG_PROB,
        lambda: _gaussian_noise(ecg),
        lambda: ecg,
    )

    # Aumentación 3: desplazamiento temporal
    ecg = tf.cond(
        tf.random.uniform([]) < _AUG_PROB,
        lambda: _time_shift(ecg),
        lambda: ecg,
    )

    # Aumentación 4: lead masking (tiene su propia probabilidad interna)
    ecg = _lead_masking(ecg)

    return ecg


# ---------------------------------------------------------------------------
# Funciones de map() para tf.data.Dataset
# ---------------------------------------------------------------------------

def make_augment_map_fn(with_weights: bool = False):
    """
    Crea una función compatible con tf.data.Dataset.map().

    Args:
        with_weights: True si el dataset incluye sample_weights
                      como tercer elemento de cada tupla.

    Returns:
        Función callable para Dataset.map().
    """
    if with_weights:
        def map_fn(inputs, labels, weights):
            ecg = inputs["ecg_input"]
            ecg_aug = augment_ecg(ecg, training=True)
            new_inputs = {**inputs, "ecg_input": ecg_aug}
            return new_inputs, labels, weights
    else:
        def map_fn(inputs, labels):
            ecg = inputs["ecg_input"]
            ecg_aug = augment_ecg(ecg, training=True)
            new_inputs = {**inputs, "ecg_input": ecg_aug}
            return new_inputs, labels

    return map_fn
