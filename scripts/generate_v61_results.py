"""
Script para generar los resultados completos de v6.1.

v6.1 = modelo v5 + umbrales F0.5 (precision-focused con restricción de recall).

Lo que hace este script:
  1. Carga el modelo v5 (best_model.keras)
  2. Carga y preprocesa el dataset PTB-XL completo
  3. Genera predicciones sobre test set
  4. Calcula métricas baseline (threshold = 0.5)  → results/v6.1/metrics_baseline.json
  5. Aplica los umbrales v6.1 (ya calculados)      → results/v6.1/metrics.json
  6. Genera TODAS las gráficas en                  → results/v6.1/plots/
  7. Copia el training_history de v5 a v6.1

Diferencia _base vs _opt:
  _base  → gráficas generadas con threshold fijo 0.5 (alta precisión, baja sensibilidad)
  _opt   → gráficas generadas con umbrales optimizados por clase (equilibrio precision/recall)

Uso:
    cd /home/saul/IA/TFM
    python Desarrollo/tfm-ecg/scripts/generate_v61_results.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rutas absolutas desde la raíz del workspace
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tensorflow as tf

from data.loader import load_dataset
from data.pipeline import create_all_datasets
from data.preprocessor import preprocess_clinical, preprocess_ecg_splits
from evaluation.evaluate import (
    compute_all_metrics,
    plot_class_distribution,
    plot_confusion_matrix_multilabel,
    plot_f1_barplot,
    plot_precision_recall_barplot,
    plot_roc_curves,
    plot_sensitivity_barplot,
)
from model.losses import AsymmetricLoss
from training.train import BATCH_SIZE, SEED

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MODEL_PATH   = Path("Desarrollo/tfm-ecg/saved_model/v5/best_model.keras")
THR_PATH     = Path("Desarrollo/tfm-ecg/saved_model/v6.1/optimal_thresholds.json")
RESULTS_DIR  = Path("Desarrollo/tfm-ecg/results/v6.1")
PLOTS_DIR    = RESULTS_DIR / "plots"
V5_HIST_CSV  = Path("Desarrollo/tfm-ecg/results/v5/training_history.csv")
V5_HIST_JSON = Path("Desarrollo/tfm-ecg/results/v5/training_history.json")


def main() -> None:
    print("=" * 62)
    print("  v6.1 — Generación de resultados y gráficas")
    print("=" * 62)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar umbrales v6.1 ──────────────────────────────────────────────
    with open(THR_PATH) as f:
        v61_thresholds: dict = json.load(f)
    print(f"\n[v6.1] Umbrales cargados desde {THR_PATH}:")
    for cls, thr in v61_thresholds.items():
        print(f"  {cls}: {thr:.3f}")

    # ── 2. Cargar modelo ─────────────────────────────────────────────────────
    print(f"\n[v6.1] Cargando modelo desde {MODEL_PATH}...")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"AsymmetricLoss": AsymmetricLoss},
    )
    print("[v6.1] Modelo cargado.")

    # ── 3. Datos ─────────────────────────────────────────────────────────────
    print("\n[v6.1] Cargando dataset PTB-XL...")
    train_data, val_data, test_data, label_names = load_dataset()
    print(f"[v6.1] Clases: {label_names}")

    print("[v6.1] Preprocesando señales ECG...")
    train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
        train_data["ecg"], val_data["ecg"], test_data["ecg"]
    )

    print("[v6.1] Preprocesando variables clínicas...")
    train_clin, val_clin, test_clin, _, _ = preprocess_clinical(
        train_data["clinical"], val_data["clinical"], test_data["clinical"]
    )

    # ── 4. Pipelines tf.data ─────────────────────────────────────────────────
    print("\n[v6.1] Construyendo pipelines tf.data...")
    train_ds, val_ds, test_ds = create_all_datasets(
        train_ecg, train_clin, train_data["labels"],
        val_ecg,   val_clin,   val_data["labels"],
        test_ecg,  test_clin,  test_data["labels"],
        batch_size=BATCH_SIZE,
        seed=SEED,
        train_sample_weights=np.ones(len(train_data["labels"]), dtype=np.float32),
    )

    # ── 5. Predicciones ──────────────────────────────────────────────────────
    print("\n[v6.1] Generando predicciones sobre test set...")
    y_pred_proba = model.predict(test_ds, verbose=1).astype(np.float32)
    y_true = test_data["labels"]

    # ── 6. Métricas baseline (thr = 0.5) ─────────────────────────────────────
    print("\n[v6.1] Métricas baseline (threshold = 0.5)...")
    metrics_base = compute_all_metrics(y_true, y_pred_proba, label_names, threshold=0.5)
    base_path = RESULTS_DIR / "metrics_baseline.json"
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(metrics_base, f, indent=2, ensure_ascii=False)
    print(f"[v6.1] Métricas baseline guardadas → {base_path}")
    print(f"       Prec macro={metrics_base['precision']['macro']:.3f}  "
          f"Sens macro={metrics_base['sensitivity_recall']['macro']:.3f}  "
          f"F1 macro={metrics_base['f1_score']['macro']:.3f}  "
          f"AUC={metrics_base['auc_roc']['macro']:.3f}")

    # ── 7. Métricas con umbrales v6.1 ────────────────────────────────────────
    print("\n[v6.1] Métricas con umbrales v6.1...")
    thr_arr = np.array([v61_thresholds[n] for n in label_names], dtype=np.float32)
    y_pred_opt = (y_pred_proba >= thr_arr).astype(int)

    from sklearn.metrics import (
        classification_report,
        f1_score,
        multilabel_confusion_matrix,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from utils.metrics import compute_specificity

    auc_per_class: dict = {}
    for i, name in enumerate(label_names):
        if y_true[:, i].sum() > 0:
            try:
                auc_per_class[name] = float(roc_auc_score(y_true[:, i], y_pred_proba[:, i]))
            except ValueError:
                auc_per_class[name] = None
        else:
            auc_per_class[name] = None

    valid_aucs = [v for v in auc_per_class.values() if v is not None]
    auc_macro  = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    f1_arr       = f1_score(y_true, y_pred_opt, average=None, zero_division=0)
    f1_macro     = float(f1_score(y_true, y_pred_opt, average="macro", zero_division=0))
    f1_per_class = {n: float(f1_arr[i]) for i, n in enumerate(label_names)}

    prec_arr      = precision_score(y_true, y_pred_opt, average=None, zero_division=0)
    prec_macro    = float(precision_score(y_true, y_pred_opt, average="macro", zero_division=0))
    prec_per_class = {n: float(prec_arr[i]) for i, n in enumerate(label_names)}

    recall_arr   = recall_score(y_true, y_pred_opt, average=None, zero_division=0)
    recall_macro = float(recall_score(y_true, y_pred_opt, average="macro", zero_division=0))
    recall_per_class = {n: float(recall_arr[i]) for i, n in enumerate(label_names)}

    spec_per_class, spec_macro = compute_specificity(y_true, y_pred_opt, label_names)
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred_opt).tolist()
    report = classification_report(
        y_true, y_pred_opt, target_names=label_names, zero_division=0, output_dict=True
    )

    sensitivity_ok = recall_macro >= 0.90
    status = "✓ OBJETIVO ALCANZADO" if sensitivity_ok else f"✗ Por debajo de 0.90"
    print(f"[v6.1] Sensibilidad macro: {recall_macro:.4f}  {status}")
    print(f"[v6.1] Precisión macro:    {prec_macro:.4f}")
    print(f"[v6.1] F1 macro:           {f1_macro:.4f}")
    print(f"[v6.1] AUC macro:          {auc_macro:.4f}")

    metrics_opt = {
        "thresholds": v61_thresholds,
        "auc_roc": {"macro": auc_macro, "per_class": auc_per_class},
        "f1_score": {"macro": f1_macro, "per_class": f1_per_class},
        "precision": {"macro": prec_macro, "per_class": prec_per_class},
        "sensitivity_recall": {
            "macro": recall_macro,
            "per_class": recall_per_class,
            "target_achieved": sensitivity_ok,
            "target_value": 0.90,
        },
        "specificity": {"macro": spec_macro, "per_class": spec_per_class},
        "confusion_matrix_multilabel": conf_matrix,
        "classification_report": report,
    }

    opt_path = RESULTS_DIR / "metrics.json"
    with open(opt_path, "w", encoding="utf-8") as f:
        json.dump(metrics_opt, f, indent=2, ensure_ascii=False)
    print(f"[v6.1] Métricas v6.1 guardadas → {opt_path}")

    # ── 8. Gráficas ──────────────────────────────────────────────────────────
    print("\n[v6.1] Generando gráficas...")

    # ROC (usa probabilidades, no depende del threshold)
    plot_roc_curves(y_true, y_pred_proba, label_names, PLOTS_DIR)

    # Distribución de clases
    plot_class_distribution(
        train_data["labels"], val_data["labels"], y_true, label_names, PLOTS_DIR
    )

    # ── Gráficas baseline (_base) ────────────────────────────────────────────
    y_pred_base = (y_pred_proba >= 0.5).astype(int)
    plot_confusion_matrix_multilabel(y_true, y_pred_base, label_names, PLOTS_DIR)
    plot_f1_barplot(
        metrics_base["f1_score"]["per_class"], label_names, PLOTS_DIR, suffix="_base"
    )
    plot_sensitivity_barplot(
        metrics_base["sensitivity_recall"]["per_class"],
        metrics_base["specificity"]["per_class"],
        label_names, PLOTS_DIR, suffix="_base",
    )

    # ── Gráficas optimizadas (_opt) ──────────────────────────────────────────
    plot_confusion_matrix_multilabel(y_true, y_pred_opt, label_names, PLOTS_DIR)
    plot_f1_barplot(
        f1_per_class, label_names, PLOTS_DIR, suffix="_opt"
    )
    plot_sensitivity_barplot(
        recall_per_class, spec_per_class, label_names, PLOTS_DIR, suffix="_opt"
    )
    plot_precision_recall_barplot(
        prec_per_class, recall_per_class, label_names, PLOTS_DIR
    )

    # ── 9. Copiar training history de v5 ────────────────────────────────────
    for src in [V5_HIST_CSV, V5_HIST_JSON]:
        if src.exists():
            dst = RESULTS_DIR / src.name
            shutil.copy2(src, dst)
            print(f"[v6.1] Copiado {src.name} → {dst}")

    # ── Resumen final ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RESUMEN v6.1 (umbrales F0.5 precision-focused)")
    print("=" * 62)
    print(f"  AUC macro:   {auc_macro:.4f}")
    print(f"  F1  macro:   {f1_macro:.4f}   (baseline: {metrics_base['f1_score']['macro']:.4f})")
    print(f"  Prec macro:  {prec_macro:.4f}   (baseline: {metrics_base['precision']['macro']:.4f})")
    print(f"  Sens macro:  {recall_macro:.4f}   (baseline: {metrics_base['sensitivity_recall']['macro']:.4f})")
    print(f"  Spec macro:  {spec_macro:.4f}   (baseline: {metrics_base['specificity']['macro']:.4f})")
    print(f"\n  Umbrales por clase:")
    for n in label_names:
        print(f"    {n:6s}: thr={v61_thresholds[n]:.3f}  "
              f"prec={prec_per_class[n]:.3f}  "
              f"sens={recall_per_class[n]:.3f}  "
              f"f1={f1_per_class[n]:.3f}")
    print(f"\n  Gráficas guardadas en:  {PLOTS_DIR}")
    print(f"  Métricas guardadas en:  {RESULTS_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()
