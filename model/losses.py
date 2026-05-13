"""
Funciones de pérdida personalizadas para clasificación multilabel ECG.

Incluye:
- FocalLoss:        ya definida en fusion.py; re-exportada aquí por comodidad.
- AsymmetricLoss:   Ridnik et al. (2021). Trata positivos y negativos de forma
                    asimétrica para mejorar precisión sin sacrificar sensibilidad.

Referencia ASL:
    Ridnik, T., Ben-Baruch, E., Noy, A., & Zelnik-Manor, L. (2021).
    Asymmetric Loss For Multi-Label Classification.
    ICCV 2021. https://arxiv.org/abs/2009.14119
"""

import keras
import tensorflow as tf


# ===========================================================================
# ASYMMETRIC LOSS
# ===========================================================================

@keras.saving.register_keras_serializable(package="tfm_ecg")
class AsymmetricLoss(tf.keras.losses.Loss):
    """
    Asymmetric Loss (ASL) para clasificación multilabel.

    Aplica exponentes de enfoque distintos para positivos y negativos:

      L_pos = -(1 - p)^gamma_pos  * log(p)
      L_neg = -(p_m)^gamma_neg    * log(1 - p_m)
      p_m   = max(p - clip, 0)    [probability shifting]

    El *probability shifting* desplaza a cero las predicciones negativas
    de baja confianza (p < clip), eliminando el ruido de ejemplos fáciles
    negativos antes de aplicar el enfoque con gamma_neg.

    Configuraciones recomendadas:
        Config A (paper): gamma_neg=4, gamma_pos=0, clip=0.05
        Config B (suave): gamma_neg=3, gamma_pos=0, clip=0.05
        Config C (mixta): gamma_neg=4, gamma_pos=1, clip=0.05
        Config D (clip+): gamma_neg=4, gamma_pos=0, clip=0.10

    Ventaja frente a Focal Loss simétrica:
        Focal Loss (γ=2) penaliza igual positivos y negativos difíciles,
        lo que con datasets desbalanceados lleva a umbrales muy bajos
        y alta tasa de falsos positivos. ASL separa el tratamiento:
        gamma_pos=0 preserva el gradiente completo en positivos (alta
        sensibilidad), gamma_neg alto reduce los falsos positivos (mayor
        precisión).

    Args:
        gamma_neg: Exponente focal para negativos. Por defecto 4.
        gamma_pos: Exponente focal para positivos. Por defecto 0.
        clip:      Margen de probability shifting. Por defecto 0.05.
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        clip:      float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # ── Rama positiva ────────────────────────────────────────────────
        # L_pos = -(1 - p)^gamma_pos * log(p)
        xs_pos = y_pred
        los_pos = tf.math.log(xs_pos)
        if self.gamma_pos > 0:
            los_pos = tf.pow(1.0 - xs_pos, self.gamma_pos) * los_pos

        # ── Rama negativa con probability shifting ───────────────────────
        # p_m = max(p - clip, 0)  →  elimina negativos de baja confianza
        # L_neg = -(p_m)^gamma_neg * log(1 - p_m)
        xs_neg = tf.maximum(y_pred - self.clip, 0.0)
        los_neg = tf.math.log(1.0 - xs_neg)
        if self.gamma_neg > 0:
            los_neg = tf.pow(xs_neg, self.gamma_neg) * los_neg

        # ── Combinación asimétrica ───────────────────────────────────────
        loss = -(y_true * los_pos + (1.0 - y_true) * los_neg)
        return tf.reduce_mean(loss)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "gamma_neg": self.gamma_neg,
            "gamma_pos": self.gamma_pos,
            "clip":      self.clip,
        })
        return config
