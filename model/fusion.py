"""
Módulo de fusión multimodal: modelo completo.

Combina la rama ECG (ResNet1D → 256 dims) y la rama tabular
(MLP → 64 dims) mediante concatenación tardía (late fusion) y
un clasificador conjunto para salida multilabel de 23 clases.

La fusión por concatenación es el enfoque más habitual y efectivo
para modalidades de naturaleza heterogénea (señal temporal vs.
variables tabulares escalares), permitiendo al clasificador aprender
las interacciones entre ambas representaciones
(Narotamo et al., 2024; Ramachandram & Taylor, 2017).

Referencias:
    Narotamo et al. (2024): A machine learning approach to predicting
    breast cancer risk. Scientific Reports.
    https://doi.org/10.1038/s41598-023-50478-4

    Ribeiro et al. (2020): Automatic diagnosis of the 12-lead ECG
    using a deep neural network. Nature Communications.
    https://doi.org/10.1038/s41467-020-15432-4

    Strodthoff et al. (2021): Deep Learning for ECG Analysis.
    IEEE J. Biomed. Health Inform. https://doi.org/10.1109/JBHI.2020.3022989
"""

import keras
import tensorflow as tf
from tensorflow.keras import layers, Model

from model.resnet1d import build_ecg_branch
from model.mlp import build_tabular_branch

# 5 superclases diagnósticas del PTB-XL: CD, HYP, MI, NORM, STTC
NUM_CLASSES: int = 5


# ===========================================================================
# FOCAL LOSS
# ===========================================================================

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal Loss binaria para clasificación multilabel.

    Penaliza más los ejemplos difíciles (mal clasificados) reduciendo
    la contribución de los ejemplos fáciles al gradiente. Esto mejora
    el recall en clases minoritarias como MI o HYP.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        gamma: Factor de enfoque. γ=0 equivale a BCE. Por defecto 2.
        alpha: Peso para la clase positiva. Por defecto 0.25.

    Returns:
        Función de pérdida compatible con model.compile(loss=...).
    """
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        p_t          = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        alpha_t      = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(alpha_t * focal_weight * bce)

    loss_fn.__name__ = "focal_loss"
    return loss_fn


@keras.saving.register_keras_serializable(package="tfm_ecg")
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss binaria como clase Keras serializable.

    Equivalente a focal_loss() pero como clase para permitir la
    serialización/deserialización correcta del modelo con Keras.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        gamma: Factor de enfoque. γ=0 equivale a BCE. Por defecto 2.
        alpha: Peso para la clase positiva. Por defecto 0.25.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        p_t          = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        alpha_t      = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        bce = -(
            y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(alpha_t * focal_weight * bce)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


def build_full_model(
    ecg_input_shape:      tuple = (1000, 12),
    clinical_input_shape: tuple = (4,),
    num_classes:          int   = NUM_CLASSES,
    dropout_fusion:       float = 0.4,
) -> Model:
    """
    Construye el modelo multimodal completo para diagnóstico ECG.

    Arquitectura de fusión:
        Rama ECG     → (512,)  ─┬
                                 ├─ Concatenate → (576,)
        Rama Tabular → ( 64,)  ─┘
        → Dense(256) → BatchNorm → ReLU → Dropout(0.4)
        → Dense(128) → BatchNorm → ReLU → Dropout(0.2)
        → Dense(5, sigmoid)     [clasificación multilabel]

    La dimensión de 576 = 512 + 64 combina la representación
    temporal de la señal ECG con el contexto clínico del paciente.

    Args:
        ecg_input_shape:      Shape del ECG (T, leads). Por defecto (1000, 12).
        clinical_input_shape: Shape de variables clínicas. Por defecto (4,).
        num_classes:          Número de clases de salida. Por defecto 5.
        dropout_fusion:       Dropout en la capa de fusión. Por defecto 0.4.

    Returns:
        Modelo Keras (API funcional) con dos entradas nombradas y una
        salida multilabel con activación sigmoid.
    """
    # ── Ramas individuales ──────────────────────────────────────────────────
    ecg_input,      ecg_features      = build_ecg_branch(
        input_shape=ecg_input_shape
    )
    clinical_input, clinical_features = build_tabular_branch(
        input_shape=clinical_input_shape
    )

    # ── Concatenación tardía ────────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion_concat")(
        [ecg_features, clinical_features]
    )  # shape: (batch, 576)  = 512 + 64

    # ── Clasificador conjunto ───────────────────────────────────────────────
    x = layers.Dense(256, use_bias=False, name="fusion_dense1")(fused)
    x = layers.BatchNormalization(name="fusion_bn1")(x)
    x = layers.Activation("relu", name="fusion_relu1")(x)
    x = layers.Dropout(dropout_fusion, name="fusion_dropout1")(x)

    x = layers.Dense(128, use_bias=False, name="fusion_dense2")(x)
    x = layers.BatchNormalization(name="fusion_bn2")(x)
    x = layers.Activation("relu", name="fusion_relu2")(x)
    x = layers.Dropout(dropout_fusion * 0.5, name="fusion_dropout2")(x)

    # Capa de salida: sigmoid para clasificación multilabel independiente
    output = layers.Dense(
        num_classes,
        activation="sigmoid",
        name="output_multilabel",
    )(x)

    # ── Modelo funcional ────────────────────────────────────────────────────
    model = Model(
        inputs=[ecg_input, clinical_input],
        outputs=output,
        name="ECG_Multimodal_ResNet1D",
    )
    return model


def compile_model(
    model:           Model,
    learning_rate:   float = 1e-3,
    loss=None,
    use_focal_loss:  bool  = True,
    focal_gamma:     float = 2.0,
    focal_alpha:     float = 0.25,
) -> Model:
    """
    Compila el modelo con la configuración estándar de entrenamiento.

    Configuración:
    - Loss: si se pasa `loss`, se usa directamente (permite ASL u otras).
      Si loss=None y use_focal_loss=True, usa FocalLoss(γ, α).
      Si loss=None y use_focal_loss=False, usa binary_crossentropy.
    - Optimizer: Adam(lr=1e-3).
    - Métricas: AUC(multi_label=True) + BinaryAccuracy.

    Args:
        model:          Modelo Keras sin compilar.
        learning_rate:  Tasa de aprendizaje inicial. Por defecto 1e-3.
        loss:           Instancia de loss o string. Si se provee, ignora
                        use_focal_loss/focal_gamma/focal_alpha.
        use_focal_loss: Si True y loss=None, usa Focal Loss. Por defecto True.
        focal_gamma:    Exponente de enfoque. Por defecto 2.0.
        focal_alpha:    Peso de la clase positiva. Por defecto 0.25.

    Returns:
        Modelo compilado, listo para model.fit().
    """
    if loss is None:
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha) if use_focal_loss else "binary_crossentropy"
    else:
        loss_fn = loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.AUC(
                num_thresholds=200,
                curve="ROC",
                multi_label=True,
                name="auc",
            ),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ],
    )
    return model
