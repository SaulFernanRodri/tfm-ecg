"""
Punto de entrada principal para el entrenamiento y evaluación (v6.2).

Orquesta el flujo completo de extremo a extremo:
1. Fijado de semillas para reproducibilidad
2. Carga del dataset PTB-XL (señales + metadatos + etiquetas)
3. Preprocesamiento de señales ECG y variables clínicas
4. Cálculo de pesos de muestra
5. Creación de pipelines tf.data.Dataset
6. Entrenamiento del modelo multimodal (ResNet1D + MLP) con AsymmetricLossPerClass
7. Evaluación completa sobre el conjunto de test (optimización de umbrales con F0.5-score)
8. Guardado de métricas y artefactos
9. Registro del experimento en MLflow

Motivación de los gammas por clase:
    MI   gamma_neg=4: infarto es urgente → tolerar FP para no perder ningún TP
    HYP  gamma_neg=2: hipertrofia es crónica → penalizar más los FP (precisión)
    CD   gamma_neg=3: trastorno de conducción, moderado
    NORM gamma_neg=3: ECG normal, moderado
    STTC gamma_neg=3: cambios ST/T, moderado

Uso:
    cd /home/saul/IA/TFM
    python Desarrollo/tfm-ecg/main.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json as _json
import numpy as np

from data.loader import load_dataset
from data.pipeline import create_all_datasets
from data.preprocessor import preprocess_ecg_splits, preprocess_clinical
from evaluation.evaluate import evaluate_model
from model.losses import AsymmetricLossPerClass
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
# Configuración del experimento v6.2
# ---------------------------------------------------------------------------
MODEL_VERSION = "0.6.2"
RUN_NAME      = "resnet5-v62-perclass-asl"
DESCRIPTION   = (
    "v6.2 | ASL per-class gamma_neg [CD=3,HYP=2,MI=4,NORM=3,STTC=3] | "
    "F0.5-threshold | 100 Hz | Norm. global | Augmentation"
)

# Gammas por clase (orden: CD, HYP, MI, NORM, STTC)
# HYP=2 reduce falsos positivos en hipertrofia
# MI=4  preserva sensibilidad máxima en infarto
ASL_GAMMA_NEG_PER_CLASS = [3, 2, 4, 3, 3]   # [CD, HYP, MI, NORM, STTC]
ASL_GAMMA_POS           = 0
ASL_CLIP                = 0.05

OUTPUT_DIR   = Path("Desarrollo/tfm-ecg/saved_model/v6.2")
RESULTS_DIR  = Path("Desarrollo/tfm-ecg/results/v6.2")


def main() -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  TFM ECG — v6.2: ASL Per-Class Gamma")
    print(f"  Versión: {MODEL_VERSION}  |  Run: {RUN_NAME}")
    print(f"  Gammas: CD={ASL_GAMMA_NEG_PER_CLASS[0]} HYP={ASL_GAMMA_NEG_PER_CLASS[1]} "
          f"MI={ASL_GAMMA_NEG_PER_CLASS[2]} NORM={ASL_GAMMA_NEG_PER_CLASS[3]} "
          f"STTC={ASL_GAMMA_NEG_PER_CLASS[4]}")
    print(f"{sep}\n")

    logger = MLflowLogger(
        experiment_name="TFM_ECG",
        run_name=RUN_NAME,
        version=MODEL_VERSION,
        tags={"description": DESCRIPTION, "dataset": "PTB-XL v1.0.3"},
    )

    with logger:
        logger.log_params({
            "seed":                    SEED,
            "batch_size":              BATCH_SIZE,
            "learning_rate":           LEARNING_RATE,
            "epochs_max":              50,
            "early_stopping_patience": 10,
            "reduce_lr_patience":      5,
            "reduce_lr_factor":        0.5,
            "loss":                    "asymmetric_loss_per_class",
            "asl_gamma_neg_CD":        ASL_GAMMA_NEG_PER_CLASS[0],
            "asl_gamma_neg_HYP":       ASL_GAMMA_NEG_PER_CLASS[1],
            "asl_gamma_neg_MI":        ASL_GAMMA_NEG_PER_CLASS[2],
            "asl_gamma_neg_NORM":      ASL_GAMMA_NEG_PER_CLASS[3],
            "asl_gamma_neg_STTC":      ASL_GAMMA_NEG_PER_CLASS[4],
            "asl_gamma_pos":           ASL_GAMMA_POS,
            "asl_clip":                ASL_CLIP,
            "threshold_strategy":      "F0.5-score (precision-focused, recall>=0.85/MI>=0.90)",
            "resnet_blocks":           5,
            "se_attention":            True,
            "filters":                 "64/128/256/384/512",
            "kernels":                 "15/11/9/7/5",
            "normalization":           "global_zscore",
            "augmentation":            "amplitude_scale+gaussian_noise+time_shift+lead_masking",
        })

        # ── 1. Semillas ──────────────────────────────────────────────────────
        set_global_seed(SEED)

        # ── 2. Carga del dataset ─────────────────────────────────────────────
        print("[v6.2] Cargando dataset PTB-XL...")
        train_data, val_data, test_data, label_names = load_dataset()

        # ── 3. Preprocesamiento ECG ──────────────────────────────────────────
        print("\n[v6.2] Preprocesando señales ECG...")
        train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
            train_data["ecg"], val_data["ecg"], test_data["ecg"]
        )

        # ── 4. Preprocesamiento tabular ──────────────────────────────────────
        print("\n[v6.2] Preprocesando variables clínicas...")
        train_clin, val_clin, test_clin, scaler, train_medians = preprocess_clinical(
            train_data["clinical"], val_data["clinical"], test_data["clinical"]
        )

        # ── 5. Pesos de muestra ──────────────────────────────────────────────
        print("\n[v6.2] Calculando pesos de muestra...")
        class_weights  = compute_multilabel_class_weights(train_data["labels"])
        sample_weights = compute_sample_weights(train_data["labels"], class_weights)

        # ── 6. Pipelines tf.data ─────────────────────────────────────────────
        print("\n[v6.2] Construyendo pipelines tf.data.Dataset...")
        train_ds, val_ds, test_ds = create_all_datasets(
            train_ecg,  train_clin,  train_data["labels"],
            val_ecg,    val_clin,    val_data["labels"],
            test_ecg,   test_clin,   test_data["labels"],
            batch_size=BATCH_SIZE,
            seed=SEED,
            train_sample_weights=sample_weights,
        )

        # ── 7. Entrenamiento con ASL per-class ───────────────────────────────
        print("\n[v6.2] Iniciando entrenamiento con ASL per-class gamma...")
        asl = AsymmetricLossPerClass(
            gamma_neg_per_class=ASL_GAMMA_NEG_PER_CLASS,
            gamma_pos=ASL_GAMMA_POS,
            clip=ASL_CLIP,
        )
        model, history = train_model(
            train_ds=train_ds,
            val_ds=val_ds,
            seed=SEED,
            mlflow_logger=logger,
            loss_fn=asl,
            output_dir=OUTPUT_DIR,
        )

        # ── 8. Evaluación con F0.5-threshold ────────────────────────────────
        print("\n[v6.2] Evaluando sobre el conjunto de test...")
        metrics = evaluate_model(
            test_ds=test_ds,
            y_true=test_data["labels"],
            label_names=label_names,
            model=model,
            train_labels=train_data["labels"],
            val_labels=val_data["labels"],
            val_ds=val_ds,
            val_true=val_data["labels"],
            results_dir=RESULTS_DIR,
            saved_model_dir=OUTPUT_DIR,
            # Pasar beta para usar F0.5 en la optimización de umbrales
            threshold_beta=0.5,
        )

        # ── 9. MLflow logging ────────────────────────────────────────────────
        baseline_path = RESULTS_DIR / "metrics_baseline.json"
        metrics_baseline = None
        if baseline_path.exists():
            with open(baseline_path) as f:
                metrics_baseline = _json.load(f)

        logger.log_metrics_from_eval(metrics, metrics_baseline)
        if "thresholds" in metrics:
            logger.log_thresholds(metrics["thresholds"])

        logger.log_artifacts_dir(str(RESULTS_DIR / "plots"))
        logger.log_artifact(str(RESULTS_DIR / "metrics.json"))
        if (RESULTS_DIR / "metrics_baseline.json").exists():
            logger.log_artifact(str(RESULTS_DIR / "metrics_baseline.json"))
        logger.log_artifact(str(OUTPUT_DIR / "optimal_thresholds.json"))
        logger.register_model(model, model_name="ECG_Multimodal_v62")

    print("[v6.2] Pipeline completado exitosamente.")
    return model, metrics


if __name__ == "__main__":
    main()
