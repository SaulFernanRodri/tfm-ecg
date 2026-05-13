"""
Módulo de la rama tabular: MLP para metadatos clínicos.

Implementa una red neuronal densa de dos capas para procesar
las 4 variables clínicas del paciente: age, sex, height, weight.

La arquitectura deliberadamente compacta (32→64 unidades) es
adecuada para el bajo número de características de entrada y
evita el sobreajuste típico de redes más profundas en datos
tabulares de baja dimensionalidad
(Narotamo et al., 2024; Gorishniy et al., 2021).

Referencias:
    Narotamo et al. (2024): A machine learning approach to predicting
    breast cancer risk based on medical history and routine blood tests.
    Scientific Reports. https://doi.org/10.1038/s41598-023-50478-4

    Gorishniy et al. (2021): Revisiting Deep Learning Models for
    Tabular Data. NeurIPS. https://arxiv.org/abs/2106.11959
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def build_tabular_branch(
    input_tensor: Optional[tf.Tensor] = None,
    input_shape:  Tuple[int, ...] = (4,),
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Construye la rama tabular MLP para variables clínicas.

    Arquitectura:
        Input(4,)                                     [age, sex, h, w]
        → Dense(32)  → BatchNorm → ReLU → Dropout(0.3)
        → Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
        → Output: (64,)

    Decisiones de diseño:
    - Se usa use_bias=False antes de BatchNorm porque la BN ya
      introduce un parámetro de desplazamiento (beta), haciendo
      el bias de la Dense redundante (Ioffe & Szegedy, 2015).
    - La activación ReLU se coloca después de BatchNorm para que
      la normalización actúe sobre distribuciones no acotadas,
      siguiendo la recomendación de la BN original.
    - El Dropout tras la activación previene co-adaptación de
      neuronas sin interferir con la normalización.

    Args:
        input_tensor: Tensor Input preexistente. Si None se crea uno nuevo.
        input_shape:  Shape del vector clínico. Por defecto (4,).

    Returns:
        Tupla (clinical_input_layer, output_tensor_64d).
    """
    if input_tensor is None:
        clinical_input = tf.keras.Input(
            shape=input_shape, name="clinical_input"
        )
    else:
        clinical_input = input_tensor

    # Capa 1: expansión inicial de representación
    x = layers.Dense(32, use_bias=False, name="tab_dense1")(clinical_input)
    x = layers.BatchNormalization(name="tab_bn1")(x)
    x = layers.Activation("relu", name="tab_relu1")(x)
    x = layers.Dropout(0.3, name="tab_dropout1")(x)

    # Capa 2: vector de representación de 64 dimensiones
    x = layers.Dense(64, use_bias=False, name="tab_dense2")(x)
    x = layers.BatchNormalization(name="tab_bn2")(x)
    x = layers.Activation("relu", name="tab_relu2")(x)
    x = layers.Dropout(0.3, name="tab_dropout2")(x)

    return clinical_input, x
