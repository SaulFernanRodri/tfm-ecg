"""
Punto de entrada principal del pipeline TFM ECG.

Orquesta el flujo completo de extremo a extremo:
1. Fijado de semillas para reproducibilidad
2. Carga del dataset PTB-XL (señales + metadatos + etiquetas)
3. Preprocesamiento de señales ECG (normalización z-score)
4. Preprocesamiento de variables clínicas (imputación + escalado)
5. Cálculo de pesos de muestra para manejar el desbalanceo
6. Creación de pipelines tf.data.Dataset
7. Entrenamiento del modelo multimodal ResNet1D + MLP
8. Evaluación completa sobre el conjunto de test
9. Guardado de métricas y artefactos
10. Registro del experimento en MLflow

Uso:
    cd /home/saul/IA/TFM
    python Desarrollo/tfm_ecg/main.py
"""

import sys
from pathlib import Path

# Añadir la raíz del paquete al path de Python
# para permitir imports absolutos desde cualquier subdirectorio
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathlib import Path

from data.loader import load_dataset
from data.pipeline import create_all_datasets
from data.preprocessor import preprocess_ecg_splits, preprocess_clinical
from evaluation.evaluate import evaluate_model
from model.losses import AsymmetricLoss
from training.train import (
    BATCH_SIZE,
    LEARNING_RATE,
    SEED,
    compute_multilabel_class_weights,
    compute_sample_weights,
    train_model,
)
from utils.mlflow_logger import MLflowLogger
from utils.seed import set_global_seed

# ---------------------------------------------------------------------------
# Configuración del experimento
# ---------------------------------------------------------------------------
MODEL_VERSION = "0.5.0"
RUN_NAME      = "resnet5-100hz-global-norm-augment"
DESCRIPTION   = (
    "500 Hz | Norm. global (Strodthoff) | ASL | "
    "Augmentation: amplitude+noise+timeshift+lead-masking (innovación)"
)

# Hiperparámetros de Asymmetric Loss (Config A — paper original)
ASL_GAMMA_NEG: float = 4
ASL_GAMMA_POS: float = 0
ASL_CLIP:      float = 0.05

# Directorio de salida para v5
OUTPUT_DIR = Path("Desarrollo/tfm-ecg/saved_model/v5")


def main() -> None:
    """
    Ejecuta el pipeline completo de entrenamiento y evaluación.

    Todas las rutas son relativas a la raíz del proyecto TFM
    (/home/saul/IA/TFM). El script debe ejecutarse desde dicha raíz.
    El experimento queda registrado automáticamente en MLflow.
    """
    sep = "=" * 62
    print(f"\n{sep}")
    print("  TFM ECG — Diagnóstico Multimodal de Infarto")
    print(f"  Versión: {MODEL_VERSION}  |  Run: {RUN_NAME}")
    print(f"  Semilla: {SEED}  |  Batch: {BATCH_SIZE}")
    print(f"{sep}\n")

    # Inicializar MLflow logger
    logger = MLflowLogger(
        experiment_name="TFM_ECG",
        run_name=RUN_NAME,
        version=MODEL_VERSION,
        tags={"description": DESCRIPTION, "dataset": "PTB-XL v1.0.3"},
    )

    with logger:
        # Parámetros del experimento
        logger.log_params({
            "seed":            SEED,
            "batch_size":      BATCH_SIZE,
            "learning_rate":   LEARNING_RATE,
            "epochs_max":      50,
            "early_stopping_patience": 10,
            "reduce_lr_patience":      5,
            "reduce_lr_factor":        0.5,
            "loss":            "asymmetric_loss",
            "asl_gamma_neg":   ASL_GAMMA_NEG,
            "asl_gamma_pos":   ASL_GAMMA_POS,
            "asl_clip":        ASL_CLIP,
            "resnet_blocks":   5,
            "se_attention":    True,
            "stem_pool":       "N/A",
            "filters":         "64/128/256/384/512",
            "kernels":         "15/11/9/7/5",
            "fusion_dense":    "256/128",
            "dropout":         "0.4/0.2",
            "ecg_output_dim":  512,
            "tabular_features": "age,sex,height,weight",
            "threshold_strategy": "sensitivity-constrained (spec>=0.65)",
            "sample_rate_hz":  100,
            "signal_length":   1000,
            "leads":           12,
            "normalization":   "global_zscore",
            "min_confidence":  0.0,
            "augmentation":    "amplitude_scale+gaussian_noise+time_shift+lead_masking",
        })

        # ── 1. Semillas ──────────────────────────────────────────────────────────
        set_global_seed(SEED)

        # ── 2. Carga del dataset ─────────────────────────────────────────────────
        print("[Main] Cargando dataset PTB-XL...")
        train_data, val_data, test_data, label_names = load_dataset()
        print(f"[Main] {len(label_names)} subclases diagnósticas cargadas.")

        # ── 3. Preprocesamiento ECG ──────────────────────────────────────────────
        print("\n[Main] Preprocesando señales ECG (normalización z-score)...")
        train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
            train_data["ecg"],
            val_data["ecg"],
            test_data["ecg"],
        )

        # ── 4. Preprocesamiento tabular ──────────────────────────────────────────
        print("\n[Main] Preprocesando variables clínicas (imputación + StandardScaler)...")
        (
            train_clin,
            val_clin,
            test_clin,
            scaler,
            train_medians,
        ) = preprocess_clinical(
            train_data["clinical"],
            val_data["clinical"],
            test_data["clinical"],
        )

        # ── 5. Pesos de muestra (manejo del desbalanceo) ─────────────────────────
        print("\n[Main] Calculando pesos de muestra para el desbalanceo de clases...")
        class_weights = compute_multilabel_class_weights(train_data["labels"])
        sample_weights = compute_sample_weights(train_data["labels"], class_weights)
        print(f"[Main] Rango de pesos: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

        # ── 6. Pipelines tf.data.Dataset ─────────────────────────────────────────
        print("\n[Main] Construyendo pipelines tf.data.Dataset...")
        train_ds, val_ds, test_ds = create_all_datasets(
            train_ecg,  train_clin,  train_data["labels"],
            val_ecg,    val_clin,    val_data["labels"],
            test_ecg,   test_clin,   test_data["labels"],
            batch_size=BATCH_SIZE,
            seed=SEED,
            train_sample_weights=sample_weights,
        )

        # ── 7. Entrenamiento ─────────────────────────────────────────────────────
        print("\n[Main] Iniciando entrenamiento del modelo...")
        asl = AsymmetricLoss(gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS, clip=ASL_CLIP)
        model, history = train_model(
            train_ds=train_ds,
            val_ds=val_ds,
            seed=SEED,
            mlflow_logger=logger,
            loss_fn=asl,
            output_dir=OUTPUT_DIR,
        )
        # ── 8. Evaluación ────────────────────────────────────────────────────────
        print("\n[Main] Evaluando sobre el conjunto de test...")
        metrics = evaluate_model(
            test_ds=test_ds,
            y_true=test_data["labels"],
            label_names=label_names,
            model=model,
            train_labels=train_data["labels"],
            val_labels=val_data["labels"],
            val_ds=val_ds,
            val_true=val_data["labels"],
            results_dir=Path("Desarrollo/tfm-ecg/results/v5"),
            saved_model_dir=OUTPUT_DIR,
        )

        # ── 9. Logging final en MLflow ────────────────────────────────────────────
        import json as _json
        baseline_path = Path("Desarrollo/tfm-ecg/results/v5/metrics_baseline.json")
        metrics_baseline = None
        if baseline_path.exists():
            with open(baseline_path) as f:
                metrics_baseline = _json.load(f)

        logger.log_metrics_from_eval(metrics, metrics_baseline)

        if "thresholds" in metrics:
            logger.log_thresholds(metrics["thresholds"])

        results_v5 = "Desarrollo/tfm-ecg/results/v5"
        logger.log_artifacts_dir(f"{results_v5}/plots")
        logger.log_artifact(f"{results_v5}/metrics.json")
        if Path(f"{results_v5}/metrics_baseline.json").exists():
            logger.log_artifact(f"{results_v5}/metrics_baseline.json")
        logger.log_artifact("Desarrollo/tfm-ecg/saved_model/v5/optimal_thresholds.json")

        logger.register_model(model, model_name="ECG_Multimodal")

    print("[Main] Pipeline completado exitosamente.")
    return model, metrics


if __name__ == "__main__":
    main()
