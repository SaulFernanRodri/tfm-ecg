"""
Módulo de entrenamiento del modelo multimodal ECG.

Responsabilidades:
- Calcular pesos por muestra para manejar el desbalanceo multilabel
- Construir y compilar el modelo
- Configurar callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Ejecutar el ciclo de entrenamiento con model.fit()
- Guardar el modelo en formato SavedModel para deployment

El desbalanceo de clases en el PTB-XL es severo: la clase NORM
representa >50% de los registros mientras que algunas subclases
diagnósticas tienen < 1%. Los sample_weight permiten que el
gradiente dé mayor importancia a las muestras de clases raras.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla para servidores sin display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from model.fusion import build_full_model, compile_model
from utils.seed import set_global_seed

if TYPE_CHECKING:
    from utils.mlflow_logger import MLflowLogger

# ---------------------------------------------------------------------------
# Directorios
# ---------------------------------------------------------------------------
SAVED_MODEL_DIR = Path("Desarrollo/tfm-ecg/saved_model")
RESULTS_DIR     = Path("Desarrollo/tfm-ecg/results")
PLOTS_DIR       = RESULTS_DIR / "plots"

# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------
EPOCHS        : int   = 50
BATCH_SIZE    : int   = 32
LEARNING_RATE : float = 1e-3
SEED          : int   = 42
NUM_CLASSES   : int   = 5


# ===========================================================================
# GRÁFICAS DE ENTRENAMIENTO
# ===========================================================================

def plot_training_curves(hist_dict: dict, plots_dir: Path) -> None:
    """
    Genera y guarda las curvas de pérdida y AUC-ROC por época.

    Produce dos ficheros PNG en plots_dir:
    - curva_loss.png: pérdida (binary_crossentropy) de entrenamiento
      y validación por época.
    - curva_auc.png:  AUC-ROC macro de entrenamiento y validación
      por época.

    Args:
        hist_dict:  Diccionario del historial de Keras (history.history).
        plots_dir:  Directorio donde guardar las figuras.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    epocas = range(1, len(hist_dict["loss"]) + 1)

    # ── Curva de pérdida ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epocas, hist_dict["loss"],     label="Entrenamiento", linewidth=2)
    ax.plot(epocas, hist_dict["val_loss"], label="Validación",    linewidth=2, linestyle="--")
    ax.set_xlabel("Época", fontsize=13)
    ax.set_ylabel("Pérdida (Binary Crossentropy)", fontsize=13)
    ax.set_title("Curva de pérdida durante el entrenamiento", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "curva_loss.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Curva de pérdida guardada → {out}")

    # ── Curva AUC ────────────────────────────────────────────────────────────
    if "auc" not in hist_dict:
        print("[Plots] WARN: clave 'auc' no encontrada en historial; omitiendo curva AUC.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epocas, hist_dict["auc"],     label="Entrenamiento", linewidth=2)
    ax.plot(epocas, hist_dict["val_auc"], label="Validación",    linewidth=2, linestyle="--")
    ax.set_xlabel("Época", fontsize=13)
    ax.set_ylabel("AUC-ROC", fontsize=13)
    ax.set_title("Curva AUC-ROC durante el entrenamiento", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "curva_auc.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Plots] Curva AUC guardada → {out}")


# ===========================================================================
# PESOS DE MUESTRA
# ===========================================================================

def compute_multilabel_class_weights(
    labels:      np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[int, float]:
    """
    Calcula el peso de la clase positiva para cada una de las 23 clases.

    Para cada clase binaria (i) se invoca compute_class_weight de
    sklearn con la estrategia 'balanced', que devuelve:
        w_positive = n_total / (2 * n_positive)

    Esto asigna mayor peso a la clase con menos muestras y menor peso
    a la clase mayoritaria, equilibrando el gradiente durante el entrena-
    miento sin alterar el dataset ni el umbral de decisión.

    Args:
        labels:      Array binario (N, 23) de etiquetas de entrenamiento.
        num_classes: Número de clases. Por defecto 23.

    Returns:
        Diccionario {índice_clase: peso_clase_positiva}.
    """
    class_weights: Dict[int, float] = {}
    for i in range(num_classes):
        col     = labels[:, i]
        classes = np.unique(col)
        if len(classes) < 2:
            # Clase ausente o siempre presente → peso neutro
            class_weights[i] = 1.0
            continue
        weights     = compute_class_weight("balanced", classes=classes, y=col)
        pos_idx     = int(np.where(classes == 1)[0][0])
        class_weights[i] = float(weights[pos_idx])
    return class_weights


def compute_sample_weights(
    labels:        np.ndarray,
    class_weights: Dict[int, float],
) -> np.ndarray:
    """
    Convierte los pesos de clase en pesos por muestra.

    Para cada muestra, el peso se calcula como la media de los pesos
    de sus clases positivas. Si una muestra no tiene ninguna clase
    positiva, se asigna peso 1.0.

    Este enfoque es necesario porque Keras soporta sample_weight de
    forma nativa en model.fit(), pero no class_weight para problemas
    multilabel (la API class_weight solo funciona con una sola salida
    de clasificación multiclase).

    Args:
        labels:        Array binario (N, 23).
        class_weights: Dict {clase_idx: peso} de compute_multilabel_class_weights.

    Returns:
        Array float32 de shape (N,) con el peso de cada muestra.
    """
    n = labels.shape[0]
    sample_w = np.ones(n, dtype=np.float32)
    for i in range(n):
        pos_classes = np.where(labels[i] == 1)[0]
        if len(pos_classes) > 0:
            sample_w[i] = float(np.mean([class_weights[c] for c in pos_classes]))
    return sample_w


# ===========================================================================
# CALLBACKS
# ===========================================================================

def get_callbacks(
    checkpoint_dir: Path,
    results_dir:    Path,
) -> list:
    """
    Configura los callbacks de entrenamiento.

    Callbacks:
    - EarlyStopping: monitorea val_auc (maximizar). Para si no mejora
      en 10 épocas consecutivas y restaura los mejores pesos.
    - ReduceLROnPlateau: monitorea val_loss. Reduce lr × 0.5 si no
      mejora en 5 épocas. LR mínimo 1e-6.
    - ModelCheckpoint: guarda el mejor modelo (val_auc máximo) en
      formato SavedModel en checkpoint_dir/best_model.
    - CSVLogger: registra métricas de entrenamiento por época en CSV.

    Args:
        checkpoint_dir: Directorio para guardar el checkpoint.
        results_dir:    Directorio para guardar el CSV del historial.

    Returns:
        Lista de callbacks configurados.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=10,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "best_model.keras"),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(results_dir / "training_history.csv"),
        append=False,
    )

    return [early_stop, reduce_lr, checkpoint, csv_logger]


# ===========================================================================
# ENTRENAMIENTO PRINCIPAL
# ===========================================================================

def train_model(
    train_ds:       tf.data.Dataset,
    val_ds:         tf.data.Dataset,
    seed:           int = SEED,
    mlflow_logger:  "Optional[MLflowLogger]" = None,
    loss_fn=None,
    output_dir:     Optional[Path] = None,
) -> Tuple[tf.keras.Model, dict]:
    """
    Ejecuta el ciclo completo de entrenamiento del modelo multimodal.

    Pasos:
    1. Fijar semillas globales
    2. Crear directorios de salida
    3. Construir y compilar el modelo
    4. Configurar callbacks
    5. Ejecutar model.fit()
    6. Guardar modelo en formato SavedModel
    7. Persistir el historial como JSON

    Nota sobre sample_weights:
    Los pesos de muestra se calculan fuera de esta función (en main.py)
    y se incorporan en train_ds mediante pipeline.create_dataset().
    Esto permite que el dataset ya lleve los pesos embebidos y los
    exponga correctamente a model.fit() sin configuración adicional.

    Args:
        train_ds:   tf.data.Dataset de entrenamiento (con sample_weights).
        val_ds:     tf.data.Dataset de validación (sin sample_weights).
        seed:       Semilla aleatoria. Por defecto 42.
        loss_fn:    Función/instancia de loss. Si None, usa FocalLoss por defecto.
        output_dir: Directorio raíz de salida. Si None, usa saved_model/ y results/
                    por defecto. Para v4 pasar Path('Desarrollo/tfm_ecg/saved_model/v4').

    Returns:
        Tupla (model, history_dict).
    """
    # 1. Semillas
    set_global_seed(seed)

    # 2. Directorios
    if output_dir is not None:
        saved_dir  = output_dir
        results_dir = Path(str(output_dir).replace("saved_model", "results"))
    else:
        saved_dir   = SAVED_MODEL_DIR
        results_dir = RESULTS_DIR
    plots_dir = results_dir / "plots"
    saved_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Modelo
    model = build_full_model()
    model = compile_model(model, learning_rate=LEARNING_RATE, loss=loss_fn)
    model.summary()

    # 4. Callbacks
    callbacks = get_callbacks(saved_dir, results_dir)
    if mlflow_logger is not None:
        callbacks.append(mlflow_logger.keras_callback())

    # 5. Entrenamiento
    print("[Train] Iniciando entrenamiento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Guardar modelo
    saved_path = saved_dir / "ecg_model.keras"
    model.save(str(saved_path))
    print(f"[Train] Modelo guardado → {saved_path}")

    # 7. Historial JSON
    hist_dict = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    hist_path = results_dir / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"[Train] Historial guardado → {hist_path}")

    # 8. Gráficas de entrenamiento
    plot_training_curves(hist_dict, plots_dir)

    return model, hist_dict
