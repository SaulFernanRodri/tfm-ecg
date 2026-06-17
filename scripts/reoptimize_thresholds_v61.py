"""
Script v6.1 — Refinamiento de Inferencia y Lógica de Negocio.

Reoptimiza los umbrales de decisión del modelo v5 usando F0.5-score
(precision-focused) en lugar de la estrategia original que maximizaba
sensibilidad sujeta a especificidad ≥ 0.65.

Estrategia v6.1:
    - Maximizar F-beta (beta=0.5) por clase sobre el conjunto de validación
    - Restricción de recall mínimo por clase (clínicamente motivado):
        MI   ≥ 0.90  (infarto: prioridad máxima de sensibilidad)
        CD   ≥ 0.85
        HYP  ≥ 0.80  (menos urgente que MI)
        NORM ≥ 0.85
        STTC ≥ 0.85
    - Guarda los nuevos umbrales en saved_model/v6.1/optimal_thresholds.json

Uso:
    cd /home/saul/IA/TFM
    python Desarrollo/tfm-ecg/scripts/reoptimize_thresholds_v61.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import tensorflow as tf

from data.loader import load_dataset
from data.preprocessor import preprocess_ecg_splits, preprocess_clinical
from data.pipeline import create_all_datasets
from evaluation.evaluate import find_optimal_thresholds_fbeta
from model.losses import AsymmetricLoss
from training.train import BATCH_SIZE, SEED

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MODEL_PATH      = Path("Desarrollo/tfm-ecg/saved_model/v5/best_model.keras")
OUTPUT_DIR      = Path("Desarrollo/tfm-ecg/saved_model/v6.1")
RESULTS_DIR     = Path("Desarrollo/tfm-ecg/results/v6.1")

FBETA           = 0.5   # precision-focused (precision pesa el doble que recall)
MIN_RECALL      = {
    "CD":   0.85,
    "HYP":  0.80,
    "MI":   0.90,
    "NORM": 0.85,
    "STTC": 0.85,
}


def main() -> None:
    print("=" * 60)
    print("  v6.1 — Refinamiento de umbrales con F0.5-score")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar modelo v5 ──────────────────────────────────────────────────
    print(f"\n[v6.1] Cargando modelo desde {MODEL_PATH}...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AsymmetricLoss": AsymmetricLoss},
    )
    print("[v6.1] Modelo cargado.")

    # ── 2. Cargar y preprocesar datos ────────────────────────────────────────
    print("\n[v6.1] Cargando dataset PTB-XL...")
    train_data, val_data, test_data, label_names = load_dataset()

    print("[v6.1] Preprocesando señales ECG...")
    train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
        train_data["ecg"], val_data["ecg"], test_data["ecg"]
    )

    print("[v6.1] Preprocesando variables clínicas...")
    train_clin, val_clin, test_clin, _, _ = preprocess_clinical(
        train_data["clinical"], val_data["clinical"], test_data["clinical"]
    )

    # ── 3. Dataset de validación ─────────────────────────────────────────────
    print("\n[v6.1] Construyendo pipeline de validación...")
    _, val_ds, _ = create_all_datasets(
        train_ecg,  train_clin,  train_data["labels"],
        val_ecg,    val_clin,    val_data["labels"],
        test_ecg,   test_clin,   test_data["labels"],
        batch_size=BATCH_SIZE,
        seed=SEED,
        train_sample_weights=np.ones(len(train_data["labels"]), dtype=np.float32),
    )

    # ── 4. Predicciones sobre validación ────────────────────────────────────
    print("\n[v6.1] Generando predicciones en validación...")
    val_preds = model.predict(val_ds, verbose=1)

    # ── 5. Optimización F0.5 ─────────────────────────────────────────────────
    new_thresholds = find_optimal_thresholds_fbeta(
        y_true=val_data["labels"],
        y_pred_proba=val_preds,
        label_names=label_names,
        beta=FBETA,
        min_recall=MIN_RECALL,
    )

    # ── 6. Comparar con umbrales v5 ──────────────────────────────────────────
    v5_thr_path = Path("Desarrollo/tfm-ecg/saved_model/v5/optimal_thresholds.json")
    if v5_thr_path.exists():
        with open(v5_thr_path) as f:
            v5_thresholds = json.load(f)
        print("\n[v6.1] Comparativa v5 → v6.1:")
        print(f"  {'Clase':<6}  {'v5':>6}  {'v6.1':>6}  {'Δ':>7}")
        print(f"  {'-'*30}")
        for name in label_names:
            old = v5_thresholds.get(name, 0.5)
            new = new_thresholds[name]
            delta = new - old
            sign = "▲" if delta > 0 else "▼"
            print(f"  {name:<6}  {old:.3f}  {new:.3f}  {sign}{abs(delta):.3f}")

    # ── 7. Guardar ───────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "optimal_thresholds.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_thresholds, f, indent=2)
    print(f"\n[v6.1] Umbrales guardados → {out_path}")

    # Guardar también registro de la estrategia
    meta = {
        "version": "v6.1",
        "strategy": f"F{FBETA}-score maximization",
        "beta": FBETA,
        "min_recall_per_class": MIN_RECALL,
        "base_model": str(MODEL_PATH),
        "thresholds": new_thresholds,
    }
    meta_path = OUTPUT_DIR / "v61_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[v6.1] Metadatos guardados → {meta_path}")
    print("\n[v6.1] ✓ Listo. Actualiza app.py con THRESHOLDS_PATH → saved_model/v6.1/")


if __name__ == "__main__":
    main()
