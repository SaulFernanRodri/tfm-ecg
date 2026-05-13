"""
Grad-CAM 1D para el modelo ECG multimodal.

Calcula mapas de calor temporales que indican qué ventana
de la señal ECG activó más cada diagnóstico. Permite al
cardiólogo verificar que el modelo se fija en los segmentos
clínicamente relevantes (ej. onda Q para MI, QRS para CD).

Referencia:
    Selvaraju et al. (2017): Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization. ICCV.
    https://arxiv.org/abs/1611.05418
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# Nombre de la última capa convolucional de la rama ECG.
# Shape de salida: (batch, 62, 512) a 100 Hz.
LAST_CONV_LAYER = "ecg_block5_relu2"


def _get_grad_model(model: tf.keras.Model, conv_layer_name: str) -> tf.keras.Model:
    """
    Construye un modelo auxiliar que devuelve simultáneamente:
    - Las activaciones de la capa convolucional indicada
    - La salida final (logits / probabilidades)

    Args:
        model:           Modelo Keras entrenado.
        conv_layer_name: Nombre de la capa convolucional objetivo.

    Returns:
        Modelo auxiliar con dos salidas: (conv_output, model_output).
    """
    conv_layer = model.get_layer(conv_layer_name)
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )


def compute_gradcam(
    model:          tf.keras.Model,
    ecg_sample:     np.ndarray,
    clinical_sample: np.ndarray,
    class_idx:      int,
    conv_layer_name: str = LAST_CONV_LAYER,
) -> np.ndarray:
    """
    Calcula el mapa Grad-CAM 1D para una muestra y una clase.

    Pasos:
    1. Forward pass guardando activaciones del último bloque conv
    2. Cálculo del gradiente de la clase objetivo respecto a esas activaciones
    3. Global Average Pooling de los gradientes → pesos por canal α_k
    4. Combinación lineal ponderada de los mapas de activación
    5. ReLU para quedarse solo con influencia positiva
    6. Interpolación bilineal 1D para reescalar al tamaño de la señal

    Args:
        model:           Modelo Keras (dos entradas: ECG + clínica).
        ecg_sample:      Array ECG (T, 12) — sin dimensión de batch.
        clinical_sample: Array clínico (n_features,) — sin dimensión de batch.
        class_idx:       Índice de la clase objetivo (0=CD, 1=HYP, 2=MI, 3=NORM, 4=STTC).
        conv_layer_name: Nombre de la capa convolucional objetivo.

    Returns:
        cam: Array 1D de shape (T,) con valores en [0, 1] normalizados,
             donde T es la longitud temporal de la señal de entrada.
    """
    grad_model = _get_grad_model(model, conv_layer_name)

    ecg_tensor      = tf.convert_to_tensor(ecg_sample[np.newaxis], dtype=tf.float32)
    clinical_tensor = tf.convert_to_tensor(clinical_sample[np.newaxis], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([ecg_tensor, clinical_tensor])
        loss = predictions[:, class_idx]

    # Gradientes: (1, T_conv, C)
    grads = tape.gradient(loss, conv_outputs)

    # Pesos por canal: α_k = GlobalAveragePool(gradients) → (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (C,)

    # Mapa de activación ponderado: (T_conv, C) · (C,) → (T_conv,)
    conv_outputs = conv_outputs[0]  # (T_conv, C)
    cam = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (T_conv,)

    # ReLU: solo influencia positiva
    cam = tf.nn.relu(cam).numpy()  # (T_conv,)

    # Interpolación al tamaño original de la señal
    T_signal = ecg_sample.shape[0]
    cam_resized = np.interp(
        np.linspace(0, len(cam) - 1, T_signal),
        np.arange(len(cam)),
        cam,
    )

    # Normalización [0, 1]
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()

    return cam_resized.astype(np.float32)


def compute_gradcam_all_classes(
    model:           tf.keras.Model,
    ecg_sample:      np.ndarray,
    clinical_sample: np.ndarray,
    label_names:     List[str],
    conv_layer_name: str = LAST_CONV_LAYER,
) -> Dict[str, np.ndarray]:
    """
    Calcula Grad-CAM para todas las clases de una muestra.

    Args:
        model:           Modelo Keras.
        ecg_sample:      Array ECG (T, 12).
        clinical_sample: Array clínico (n_features,).
        label_names:     Lista de nombres de clase (ej. ["CD","HYP","MI","NORM","STTC"]).
        conv_layer_name: Nombre de la capa convolucional objetivo.

    Returns:
        Diccionario {clase: cam_array(T,)}.
    """
    return {
        name: compute_gradcam(
            model, ecg_sample, clinical_sample,
            class_idx=i, conv_layer_name=conv_layer_name,
        )
        for i, name in enumerate(label_names)
    }


def batch_gradcam(
    model:           tf.keras.Model,
    ecg_batch:       np.ndarray,
    clinical_batch:  np.ndarray,
    class_idx:       int,
    conv_layer_name: str = LAST_CONV_LAYER,
) -> np.ndarray:
    """
    Calcula Grad-CAM para un batch de muestras y una clase.
    Útil para calcular el CAM promedio de una clase sobre múltiples ECGs.

    Args:
        model:           Modelo Keras.
        ecg_batch:       Array (N, T, 12).
        clinical_batch:  Array (N, n_features).
        class_idx:       Índice de la clase objetivo.
        conv_layer_name: Nombre de la capa convolucional objetivo.

    Returns:
        cams: Array (N, T) con un mapa por muestra.
    """
    cams = []
    for i in range(len(ecg_batch)):
        cam = compute_gradcam(
            model, ecg_batch[i], clinical_batch[i],
            class_idx=class_idx, conv_layer_name=conv_layer_name,
        )
        cams.append(cam)
    return np.stack(cams, axis=0)
