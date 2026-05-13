"""
Calibración post-hoc de probabilidades mediante Temperature Scaling.

El modelo entrenado produce probabilidades a menudo mal calibradas:
una predicción de p=0.80 no significa necesariamente que el evento
ocurra el 80% de las veces. Temperature Scaling corrige esto sin
reentrenar la red.

Idea central:
    p_calibrada = sigmoid(logit(p) / T)

    T > 1 → probabilidades más bajas, más conservadoras (reduce FP)
    T < 1 → probabilidades más altas, más agresivas  (reduce FN)
    T = 1 → sin cambio

T se aprende minimizando Binary Cross-Entropy sobre el conjunto de
validación. Al ser un solo parámetro escalar, no hay riesgo de
sobreajuste.

Referencia:
    Guo et al. (2017). On Calibration of Modern Neural Networks.
    ICML 2017. https://arxiv.org/abs/1706.04599
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


class TemperatureScaling:
    """
    Calibración post-hoc de un modelo multilabel con un escalar T.

    Uso típico:
        cal = TemperatureScaling()
        cal.fit(y_true_val, y_pred_val)      # aprende T en validación
        y_cal = cal.calibrate(y_pred_test)   # aplica en test
        cal.save("saved_model/v4/temperature.json")

    Attributes:
        temperature: Valor aprendido de T (float, > 0). None antes de fit().
    """

    def __init__(self) -> None:
        self.temperature: float | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        lr: float = 0.01,
        max_steps: int = 1000,
        tol: float = 1e-6,
        verbose: bool = True,
    ) -> "TemperatureScaling":
        """
        Aprende T minimizando BCE sobre (y_true, y_pred_proba).

        Convierte las probabilidades a logits, optimiza T con Adam,
        y registra la pérdida cada 100 pasos si verbose=True.

        Args:
            y_true:       Array (N, C) de etiquetas binarias.
            y_pred_proba: Array (N, C) de probabilidades del modelo.
            lr:           Tasa de aprendizaje de Adam. Por defecto 0.01.
            max_steps:    Máximo de pasos de optimización. Por defecto 1000.
            tol:          Tolerancia de convergencia (cambio en loss). Por defecto 1e-6.
            verbose:      Imprimir progreso. Por defecto True.

        Returns:
            self (para encadenamiento).
        """
        y_true  = tf.constant(y_true,       dtype=tf.float32)
        logits  = self._to_logits(y_pred_proba)

        # T como variable entrenable, inicializado en 1.0
        log_T   = tf.Variable(0.0, trainable=True, dtype=tf.float32)  # T = exp(log_T) > 0
        opt     = tf.keras.optimizers.Adam(learning_rate=lr)
        bce     = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        prev_loss = float("inf")
        for step in range(max_steps):
            with tf.GradientTape() as tape:
                T    = tf.exp(log_T)
                loss = bce(y_true, logits / T)

            grads = tape.gradient(loss, [log_T])
            opt.apply_gradients(zip(grads, [log_T]))

            loss_val = float(loss.numpy())
            if verbose and (step % 100 == 0 or step == max_steps - 1):
                print(f"  [TemperatureScaling] step={step:4d}  T={float(tf.exp(log_T)):.4f}  BCE={loss_val:.6f}")

            if abs(prev_loss - loss_val) < tol:
                if verbose:
                    print(f"  [TemperatureScaling] Convergencia en step={step}  T={float(tf.exp(log_T)):.4f}")
                break
            prev_loss = loss_val

        self.temperature = float(tf.exp(log_T).numpy())
        return self

    def calibrate(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Aplica Temperature Scaling a un array de probabilidades.

        Args:
            y_pred_proba: Array (N, C) de probabilidades del modelo.

        Returns:
            Array (N, C) de probabilidades calibradas.
        """
        if self.temperature is None:
            raise RuntimeError("Llama a fit() antes de calibrate().")

        logits    = self._to_logits(y_pred_proba)
        cal_proba = tf.sigmoid(logits / self.temperature).numpy()
        return cal_proba.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """
        Guarda el valor de T en un fichero JSON.

        Args:
            path: Ruta del fichero, p.ej. 'saved_model/v4/temperature.json'.
        """
        if self.temperature is None:
            raise RuntimeError("Llama a fit() antes de save().")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)
        print(f"[TemperatureScaling] T={self.temperature:.4f} guardado → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaling":
        """
        Carga T desde un fichero JSON guardado con save().

        Args:
            path: Ruta del fichero JSON.

        Returns:
            Instancia con self.temperature ya establecido.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        obj.temperature = float(data["temperature"])
        print(f"[TemperatureScaling] T={obj.temperature:.4f} cargado desde {path}")
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_logits(y_pred_proba: np.ndarray) -> tf.Tensor:
        """
        Convierte probabilidades σ(·) a logits: logit(p) = log(p / (1-p)).
        Aplica clipping para evitar log(0).
        """
        p = np.clip(np.asarray(y_pred_proba, dtype=np.float32), 1e-7, 1.0 - 1e-7)
        return tf.constant(np.log(p / (1.0 - p)), dtype=tf.float32)
