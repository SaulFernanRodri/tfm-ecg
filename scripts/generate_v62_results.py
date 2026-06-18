"""
Script v6.2 — Optimización Restringida de umbrales (RNF-04).

Contexto:
    v6.1 detectó una regresión clínica crítica: al maximizar F0.5-score
    la sensibilidad macro cayó a 0.855, violando el requisito no funcional
    RNF-04 (Sensibilidad >= 0.90 para triaje clínico).

Estrategia v6.2 (Constrained Precision Maximization):
    Para CADA clase por separado:
    1. Grid search de umbrales de 0.95 a 0.05 con paso 0.01 (91 candidatos).
    2. Se selecciona el umbral MÁS ALTO que garantice recall >= 0.90.
       (threshold alto → mayor precisión; recall decreciente con threshold)
    3. Si ningún umbral alcanza 0.90 de recall (caso infactible),
       se usa el threshold con el recall más cercano a 0.90 por debajo.

Qué produce:
    - saved_model/v6.2/optimal_thresholds.json  : nuevos umbrales
    - saved_model/v6.2/v62_meta.json            : metadatos del experimento
    - results/v6.2/metrics_baseline.json        : métricas con thr=0.5
    - results/v6.2/metrics.json                 : métricas con umbrales v6.2
    - results/v6.2/plots/                       : todas las gráficas

Uso:
    cd /home/saul/IA/TFM
    python3 Desarrollo/tfm-ecg/scripts/generate_v62_results.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.loader import load_dataset
from data.pipeline import create_all_datasets
from data.preprocessor import preprocess_clinical, preprocess_ecg_splits
from evaluation.evaluate import (
    compute_all_metrics,
    find_optimal_thresholds_recall_constrained,
    plot_class_distribution,
    plot_confusion_matrix_multilabel,
    plot_f1_barplot,
    plot_precision_recall_barplot,
    plot_roc_curves,
    plot_sensitivity_barplot,
)
from model.losses import AsymmetricLoss
from training.train import BATCH_SIZE, SEED
from utils.metrics import compute_specificity

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MODEL_PATH   = Path("Desarrollo/tfm-ecg/saved_model/v5/best_model.keras")
OUTPUT_DIR   = Path("Desarrollo/tfm-ecg/saved_model/v6.2")
RESULTS_DIR  = Path("Desarrollo/tfm-ecg/results/v6.2")
PLOTS_DIR    = RESULTS_DIR / "plots"
V5_HIST_CSV  = Path("Desarrollo/tfm-ecg/results/v5/training_history.csv")
V5_HIST_JSON = Path("Desarrollo/tfm-ecg/results/v5/training_history.json")

MIN_RECALL_GLOBAL = 0.90   # RNF-04


def _compute_metrics_from_binary(
    y_true:      np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_bin:  np.ndarray,
    label_names: list,
) -> dict:
    """Calcula el conjunto completo de métricas dado un array ya binarizado."""
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

    f1_arr       = f1_score(y_true, y_pred_bin, average=None, zero_division=0)
    f1_macro     = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))
    f1_per_class = {n: float(f1_arr[i]) for i, n in enumerate(label_names)}

    prec_arr      = precision_score(y_true, y_pred_bin, average=None, zero_division=0)
    prec_macro    = float(precision_score(y_true, y_pred_bin, average="macro", zero_division=0))
    prec_per_class = {n: float(prec_arr[i]) for i, n in enumerate(label_names)}

    recall_arr    = recall_score(y_true, y_pred_bin, average=None, zero_division=0)
    recall_macro  = float(recall_score(y_true, y_pred_bin, average="macro", zero_division=0))
    recall_per_class = {n: float(recall_arr[i]) for i, n in enumerate(label_names)}

    spec_per_class, spec_macro = compute_specificity(y_true, y_pred_bin, label_names)
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred_bin).tolist()
    report = classification_report(
        y_true, y_pred_bin, target_names=label_names, zero_division=0, output_dict=True
    )

    return {
        "auc_roc": {"macro": auc_macro, "per_class": auc_per_class},
        "f1_score": {"macro": f1_macro, "per_class": f1_per_class},
        "precision": {"macro": prec_macro, "per_class": prec_per_class},
        "sensitivity_recall": {
            "macro": recall_macro,
            "per_class": recall_per_class,
            "target_achieved": recall_macro >= MIN_RECALL_GLOBAL,
            "target_value": MIN_RECALL_GLOBAL,
        },
        "specificity": {"macro": spec_macro, "per_class": spec_per_class},
        "confusion_matrix_multilabel": conf_matrix,
        "classification_report": report,
    }


def main() -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  v6.2 — Optimización Restringida de Umbrales (RNF-04)")
    print(f"  Objetivo: recall >= {MIN_RECALL_GLOBAL:.2f} garantizado por clase")
    print(f"  Modelo base: {MODEL_PATH}")
    print(f"{sep}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar modelo ─────────────────────────────────────────────────────
    print(f"[v6.2] Cargando modelo desde {MODEL_PATH}...")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"AsymmetricLoss": AsymmetricLoss},
    )
    print("[v6.2] Modelo cargado.\n")

    # ── 2. Datos ─────────────────────────────────────────────────────────────
    print("[v6.2] Cargando dataset PTB-XL...")
    train_data, val_data, test_data, label_names = load_dataset()
    print(f"[v6.2] Clases: {label_names}")
    print(f"[v6.2] Train={len(train_data['labels'])}  "
          f"Val={len(val_data['labels'])}  "
          f"Test={len(test_data['labels'])}\n")

    print("[v6.2] Preprocesando señales ECG...")
    train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
        train_data["ecg"], val_data["ecg"], test_data["ecg"]
    )

    print("[v6.2] Preprocesando variables clínicas...")
    train_clin, val_clin, test_clin, _, _ = preprocess_clinical(
        train_data["clinical"], val_data["clinical"], test_data["clinical"]
    )

    # ── 3. Pipelines tf.data ─────────────────────────────────────────────────
    print("\n[v6.2] Construyendo pipelines tf.data...")
    train_ds, val_ds, test_ds = create_all_datasets(
        train_ecg, train_clin, train_data["labels"],
        val_ecg,   val_clin,   val_data["labels"],
        test_ecg,  test_clin,  test_data["labels"],
        batch_size=BATCH_SIZE,
        seed=SEED,
        train_sample_weights=np.ones(len(train_data["labels"]), dtype=np.float32),
    )

    # ── 4. Predicciones ──────────────────────────────────────────────────────
    print("\n[v6.2] Generando predicciones sobre VALIDACIÓN (para optimizar umbrales)...")
    y_val_proba = model.predict(val_ds, verbose=1).astype(np.float32)
    y_val_true  = val_data["labels"]

    print("\n[v6.2] Generando predicciones sobre TEST (para evaluación final)...")
    y_test_proba = model.predict(test_ds, verbose=1).astype(np.float32)
    y_test_true  = test_data["labels"]

    # ── 5. Optimización Restringida (RNF-04) en VALIDACIÓN ───────────────────
    v62_thresholds = find_optimal_thresholds_recall_constrained(
        y_true       = y_val_true,
        y_pred_proba = y_val_proba,
        label_names  = label_names,
        min_recall   = MIN_RECALL_GLOBAL,
        step         = 0.01,
    )

    # ── 6. Guardar umbrales y metadatos ──────────────────────────────────────
    thr_path = OUTPUT_DIR / "optimal_thresholds.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(v62_thresholds, f, indent=2)
    print(f"\n[v6.2] Umbrales guardados → {thr_path}")

    # Comparativa con versiones anteriores
    prev_versions = {
        "v5":  Path("Desarrollo/tfm-ecg/saved_model/v5/optimal_thresholds.json"),
        "v6.1": Path("Desarrollo/tfm-ecg/saved_model/v6.1/optimal_thresholds.json"),
    }
    print(f"\n  {'Clase':<6}  " + "  ".join(f"{v:>6}" for v in [*prev_versions, "v6.2"]))
    print(f"  {'-'*40}")
    for name in label_names:
        vals = []
        for vname, vpath in prev_versions.items():
            try:
                vals.append(f"{json.load(open(vpath)).get(name, float('nan')):.3f}")
            except FileNotFoundError:
                vals.append("  N/A")
        vals.append(f"{v62_thresholds[name]:.3f}")
        print(f"  {name:<6}  " + "  ".join(f"{v:>6}" for v in vals))

    meta = {
        "version": "v6.2",
        "strategy": "Constrained Precision Maximization (RNF-04)",
        "description": (
            "Para cada clase: umbral más alto con recall >= 0.90 en validación. "
            "Si es infactible: umbral de máximo recall alcanzable."
        ),
        "min_recall_required": MIN_RECALL_GLOBAL,
        "grid_step": 0.01,
        "base_model": str(MODEL_PATH),
        "thresholds": v62_thresholds,
    }
    meta_path = OUTPUT_DIR / "v62_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[v6.2] Metadatos guardados → {meta_path}")

    # ── 7. Métricas baseline (thr = 0.5) sobre TEST ──────────────────────────
    print("\n[v6.2] Calculando métricas baseline (thr=0.5)...")
    y_pred_base = (y_test_proba >= 0.5).astype(int)
    metrics_base = _compute_metrics_from_binary(
        y_test_true, y_test_proba, y_pred_base, label_names
    )
    metrics_base["threshold"] = 0.5
    base_path = RESULTS_DIR / "metrics_baseline.json"
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(metrics_base, f, indent=2, ensure_ascii=False)
    print(f"[v6.2] Baseline: prec={metrics_base['precision']['macro']:.3f}  "
          f"recall={metrics_base['sensitivity_recall']['macro']:.3f}  "
          f"f1={metrics_base['f1_score']['macro']:.3f}")

    # ── 8. Métricas con umbrales v6.2 sobre TEST ─────────────────────────────
    print("\n[v6.2] Calculando métricas con umbrales v6.2...")
    thr_arr = np.array([v62_thresholds[n] for n in label_names], dtype=np.float32)
    y_pred_opt = (y_test_proba >= thr_arr).astype(int)
    metrics_opt = _compute_metrics_from_binary(
        y_test_true, y_test_proba, y_pred_opt, label_names
    )
    metrics_opt["thresholds"] = v62_thresholds

    rnf04_ok = metrics_opt["sensitivity_recall"]["target_achieved"]
    status   = "✓ RNF-04 CUMPLIDO" if rnf04_ok else "✗ RNF-04 INCUMPLIDO"
    print(f"[v6.2] {status}")
    print(f"[v6.2] Sens macro: {metrics_opt['sensitivity_recall']['macro']:.4f}  "
          f"(objetivo >= {MIN_RECALL_GLOBAL})")
    print(f"[v6.2] Prec macro: {metrics_opt['precision']['macro']:.4f}")
    print(f"[v6.2] F1   macro: {metrics_opt['f1_score']['macro']:.4f}")
    print(f"[v6.2] AUC  macro: {metrics_opt['auc_roc']['macro']:.4f}")

    opt_path = RESULTS_DIR / "metrics.json"
    with open(opt_path, "w", encoding="utf-8") as f:
        json.dump(metrics_opt, f, indent=2, ensure_ascii=False)
    print(f"[v6.2] Métricas guardadas → {opt_path}")

    # ── 9. Gráficas ──────────────────────────────────────────────────────────
    print("\n[v6.2] Generando gráficas...")

    # ROC (invariante al threshold)
    plot_roc_curves(y_test_true, y_test_proba, label_names, PLOTS_DIR)

    # Distribución de clases
    plot_class_distribution(
        train_data["labels"], val_data["labels"], y_test_true, label_names, PLOTS_DIR
    )

    # Baseline (_base)
    plot_f1_barplot(
        metrics_base["f1_score"]["per_class"], label_names, PLOTS_DIR, suffix="_base"
    )
    plot_sensitivity_barplot(
        metrics_base["sensitivity_recall"]["per_class"],
        metrics_base["specificity"]["per_class"],
        label_names, PLOTS_DIR, suffix="_base",
    )

    # Optimizado (_opt)
    plot_confusion_matrix_multilabel(y_test_true, y_pred_opt, label_names, PLOTS_DIR)
    plot_f1_barplot(
        metrics_opt["f1_score"]["per_class"], label_names, PLOTS_DIR, suffix="_opt"
    )
    plot_sensitivity_barplot(
        metrics_opt["sensitivity_recall"]["per_class"],
        metrics_opt["specificity"]["per_class"],
        label_names, PLOTS_DIR, suffix="_opt",
    )
    plot_precision_recall_barplot(
        metrics_opt["precision"]["per_class"],
        metrics_opt["sensitivity_recall"]["per_class"],
        label_names, PLOTS_DIR,
    )

    # ── 10. Copiar training history de v5 ────────────────────────────────────
    for src in [V5_HIST_CSV, V5_HIST_JSON]:
        if src.exists():
            shutil.copy2(src, RESULTS_DIR / src.name)
            print(f"[v6.2] Copiado {src.name}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESUMEN COMPARATIVO (test set)")
    print(sep)
    print(f"  {'Métrica':<18}  {'baseline (0.5)':>14}  {'v6.1':>8}  {'v6.2':>8}")
    print(f"  {'-'*54}")

    v61_m: dict = {}
    try:
        v61_m = json.load(open("Desarrollo/tfm-ecg/results/v6.1/metrics.json"))
    except FileNotFoundError:
        pass

    rows = [
        ("AUC macro",  "auc_roc",             "macro"),
        ("F1 macro",   "f1_score",             "macro"),
        ("Prec macro", "precision",            "macro"),
        ("Sens macro", "sensitivity_recall",   "macro"),
        ("Spec macro", "specificity",          "macro"),
    ]
    for label, key, subkey in rows:
        base_val = metrics_base[key][subkey]
        opt_val  = metrics_opt[key][subkey]
        v61_val  = v61_m.get(key, {}).get(subkey, float("nan")) if v61_m else float("nan")
        rnf_flag = " ← RNF-04" if key == "sensitivity_recall" else ""
        print(f"  {label:<18}  {base_val:>14.4f}  {v61_val:>8.4f}  {opt_val:>8.4f}{rnf_flag}")

    print(f"\n  Umbrales v6.2 por clase:")
    print(f"  {'Clase':<6}  {'thr':>5}  {'recall':>7}  {'prec':>7}  {'f1':>7}")
    print(f"  {'-'*38}")
    for n in label_names:
        r = metrics_opt["sensitivity_recall"]["per_class"][n]
        p = metrics_opt["precision"]["per_class"][n]
        f = metrics_opt["f1_score"]["per_class"][n]
        ok = "✓" if r >= MIN_RECALL_GLOBAL else "✗"
        print(f"  {n:<6}  {v62_thresholds[n]:>5.3f}  {r:>7.3f}  {p:>7.3f}  {f:>7.3f}  {ok}")

    print(f"\n  Gráficas → {PLOTS_DIR}")
    print(f"  Métricas → {RESULTS_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
