"""
Módulo de la rama ECG: bloques residuales 1D y rama completa.

Implementa una ResNet1D de 3 bloques para el procesamiento de
señales ECG de 12 derivaciones a 100 Hz.

La arquitectura residual fue propuesta originalmente por He et al. (2016)
para visión artificial y adaptada al dominio ECG por Strodthoff et al.
(2021), quienes demostraron que los kernels de gran tamaño (9–15) son
clave para capturar patrones de larga duración en señales cardíacas.

Referencias:
    He et al. (2016): Deep Residual Learning for Image Recognition.
    CVPR. https://arxiv.org/abs/1512.03385

    Strodthoff et al. (2021): Deep Learning for ECG Analysis:
    Benchmarks and Insights from PTB-XL.
    IEEE J. Biomed. Health Inform. 25(5):1519-1528.
    https://doi.org/10.1109/JBHI.2020.3022989
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers


def build_se_block(
    x:          tf.Tensor,
    reduction:  int = 8,
    block_name: str = "se",
) -> tf.Tensor:
    """
    Squeeze-and-Excitation block (Hu et al., 2018) para señales 1D.

    Recalibra los feature maps aprendiendo pesos por canal:
    - Squeeze: GlobalAveragePooling colapsa la dimensión temporal → (C,)
    - Excitation: FC → ReLU → FC → Sigmoid → (C,) de pesos en [0, 1]
    - Scale: multiplica canal a canal los pesos por el tensor original

    Esto permite al modelo enfocarse en los canales (filtros) más
    relevantes para cada patrón ECG, mejorando la discriminación
    entre clases sin añadir parámetros de convolución extra.

    Args:
        x:          Tensor (batch, T, C).
        reduction:  Factor de reducción para la capa FC interna. Por defecto 8.
        block_name: Prefijo de nombres de capa.

    Returns:
        Tensor (batch, T, C) con canales recalibrados.

    Referencias:
        Hu et al. (2018): Squeeze-and-Excitation Networks. CVPR.
        https://arxiv.org/abs/1709.01507
    """
    c = x.shape[-1]
    # Squeeze
    s = layers.GlobalAveragePooling1D(name=f"{block_name}_squeeze")(x)
    # Excitation
    e = layers.Dense(max(c // reduction, 1), activation="relu",
                     use_bias=False, name=f"{block_name}_fc1")(s)
    e = layers.Dense(c, activation="sigmoid",
                     use_bias=False, name=f"{block_name}_fc2")(e)
    # Reshape para broadcast en dimensión temporal: (batch, 1, C)
    e = layers.Reshape((1, c), name=f"{block_name}_reshape")(e)
    # Scale
    return layers.Multiply(name=f"{block_name}_scale")([x, e])


def build_resnet_block(
    x:            tf.Tensor,
    filters:      int,
    kernel_size:  int,
    use_pooling:  bool = False,
    use_se:       bool = True,
    se_reduction: int  = 8,
    block_name:   str  = "resnet_block",
) -> tf.Tensor:
    """
    Construye un bloque residual 1D pre-activation style.

    Estructura interna:
        Input
        ├─ Conv1D(filters, kernel_size, padding='same') → BN → ReLU
        │  Conv1D(filters, kernel_size, padding='same') → BN
        └─ Skip connection:
               si in_filters == filters → identidad
               si in_filters != filters → Conv1D(filters, 1) → BN
        Add([main_path, skip]) → ReLU
        [MaxPooling1D(2)] ← solo si use_pooling=True

    El padding='same' en todas las convoluciones mantiene la longitud
    de la secuencia constante dentro del bloque (el downsampling se
    realiza exclusivamente con MaxPooling1D).

    El skip connection con proyección 1×1 es la solución estándar
    de He et al. (2016) para alinear dimensiones cuando el número
    de filtros cambia entre bloques.

    Args:
        x:            Tensor de entrada de shape (batch, T, C).
        filters:      Número de filtros de las convoluciones principales.
        kernel_size:  Tamaño del kernel de las convoluciones principales.
        use_pooling:  Si True, aplica MaxPooling1D(2) al final del bloque.
        use_se:       Si True, aplica SE attention tras la fusión residual.
        se_reduction: Factor de reducción del SE block. Por defecto 8.
        block_name:   Prefijo para los nombres de capa del bloque.

    Returns:
        Tensor de salida del bloque residual.
    """
    input_filters = x.shape[-1]

    # ── Rama principal ──────────────────────────────────────────────────────
    h = layers.Conv1D(
        filters, kernel_size, padding="same", use_bias=False,
        name=f"{block_name}_conv1"
    )(x)
    h = layers.BatchNormalization(name=f"{block_name}_bn1")(h)
    h = layers.Activation("relu", name=f"{block_name}_relu1")(h)

    h = layers.Conv1D(
        filters, kernel_size, padding="same", use_bias=False,
        name=f"{block_name}_conv2"
    )(h)
    h = layers.BatchNormalization(name=f"{block_name}_bn2")(h)

    # ── Skip connection (proyección si cambian los filtros) ─────────────────
    if input_filters != filters:
        shortcut = layers.Conv1D(
            filters, 1, padding="same", use_bias=False,
            name=f"{block_name}_skip_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            name=f"{block_name}_skip_bn"
        )(shortcut)
    else:
        shortcut = x

    # ── SE attention (antes de la fusión residual) ──────────────────────────
    if use_se:
        h = build_se_block(h, reduction=se_reduction, block_name=f"{block_name}_se")

    # ── Fusión residual ─────────────────────────────────────────────────────
    out = layers.Add(name=f"{block_name}_add")([h, shortcut])
    out = layers.Activation("relu", name=f"{block_name}_relu2")(out)

    # ── Downsampling opcional ───────────────────────────────────────────────
    if use_pooling:
        out = layers.MaxPooling1D(2, name=f"{block_name}_pool")(out)

    return out


def build_ecg_branch(
    input_tensor: Optional[tf.Tensor] = None,
    input_shape:  Tuple[int, int] = (1000, 12),
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Construye la rama ECG completa para señales a 100 Hz.

    5 bloques ResNet1D con SE attention.

    Resumen de dimensiones:
        (batch, 1000, 12)  → Input (100 Hz, 10 s)
        (batch, 1000, 64)  → Bloque 1 (k=15, sin pool, sin SE)
        (batch,  500, 128) → Bloque 2 (k=11, MaxPool×2, SE)
        (batch,  250, 256) → Bloque 3 (k=9,  MaxPool×2, SE)
        (batch,  125, 384) → Bloque 4 (k=7,  MaxPool×2, SE)
        (batch,   62, 512) → Bloque 5 (k=5,  MaxPool×2, SE)
        (batch,       512) → GlobalAveragePooling1D
        (batch,       512) → Dropout(0.3)

    Compatibilidad XAI:
        Grad-CAM 1D se aplica sobre la salida del último bloque
        convolucional (shape: batch, 62, 512).

    Args:
        input_tensor: Tensor Input preexistente. Si None se crea uno nuevo.
        input_shape:  Shape del ECG sin dimensión de batch.
                      Por defecto (1000, 12) para 100 Hz.

    Returns:
        Tupla (ecg_input_layer, output_tensor_512d).
    """
    if input_tensor is None:
        ecg_input = tf.keras.Input(shape=input_shape, name="ecg_input")
    else:
        ecg_input = input_tensor

    # ── Bloque 1 (T=1000): receptive field para ondas completas ─────────────
    x = build_resnet_block(
        ecg_input, filters=64, kernel_size=15,
        use_pooling=False, use_se=False, block_name="ecg_block1"
    )

    # ── Bloque 2 (T=1000→500) ────────────────────────────────────────────────
    x = build_resnet_block(
        x, filters=128, kernel_size=11,
        use_pooling=True, use_se=True, block_name="ecg_block2"
    )

    # ── Bloque 3 (T=500→250) ─────────────────────────────────────────────────
    x = build_resnet_block(
        x, filters=256, kernel_size=9,
        use_pooling=True, use_se=True, block_name="ecg_block3"
    )

    # ── Bloque 4 (T=250→125) ─────────────────────────────────────────────────
    x = build_resnet_block(
        x, filters=384, kernel_size=7,
        use_pooling=True, use_se=True, block_name="ecg_block4"
    )

    # ── Bloque 5 (T=125→62) ──────────────────────────────────────────────────
    x = build_resnet_block(
        x, filters=512, kernel_size=5,
        use_pooling=True, use_se=True, block_name="ecg_block5"
    )

    # ── Agregación temporal y regularización ─────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="ecg_gap")(x)
    x = layers.Dropout(0.3, name="ecg_dropout")(x)

    return ecg_input, x
