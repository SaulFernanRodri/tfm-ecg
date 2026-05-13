"""
Punto de entrada del pipeline XAI (Explainable AI).

Ejecuta los tres análisis de explicabilidad sobre el modelo v5:
1. Grad-CAM 1D        — qué ventana temporal activa cada clase
2. Lead Importance    — qué derivación es más importante por clase
3. SHAP clínico       — contribución de variables clínicas (edad, sexo, altura, peso)

Los resultados se guardan en:
    Desarrollo/tfm-ecg/results/xai/
        gradcam/         ← PNG por clase (+ plot de una muestra representativa)
        lead_importance/ ← PNG por clase + heatmap global
        shap/            ← barras + beeswarm por clase

Uso:
    cd /home/saul/IA/TFM
    python Desarrollo/tfm-ecg/xai_main.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import tensorflow as tf

from data.loader import load_dataset
from data.preprocessor import preprocess_ecg_splits, preprocess_clinical
from model.losses import AsymmetricLoss
from xai.gradcam import compute_gradcam_all_classes, batch_gradcam
from xai.lead_importance import compute_lead_importance_per_class
from xai.shap_clinical import compute_shap_all_classes, mean_abs_shap
from xai.visualize import (
    plot_gradcam_all_classes,
    plot_lead_importance_all_classes,
    plot_lead_importance_heatmap,
    plot_shap_all_classes,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
SAVED_MODEL_PATH = Path("Desarrollo/tfm-ecg/saved_model/v5/best_model.keras")
OUTPUT_DIR       = Path("Desarrollo/tfm-ecg/results/xai")

# Número de muestras de test a usar para los análisis (cuantas más, más lento)
N_GRADCAM_SAMPLES   = 1    # Grad-CAM: 1 muestra representativa (visualización)
N_LEAD_SAMPLES      = 200  # Lead importance: promedio sobre N muestras
N_SHAP_SAMPLES      = 100  # SHAP: M muestras a explicar
N_SHAP_BACKGROUND   = 50   # SHAP: background del training set

# Índice de muestra representativa para Grad-CAM individual
REPRESENTATIVE_IDX  = 0


def load_model() -> tf.keras.Model:
    """Carga el modelo v5 con la loss personalizada registrada."""
    asl = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    model = tf.keras.models.load_model(
        SAVED_MODEL_PATH,
        custom_objects={"AsymmetricLoss": AsymmetricLoss},
    )
    print(f"[XAI] Modelo cargado desde {SAVED_MODEL_PATH}")
    return model


def main() -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  TFM ECG — Pipeline XAI")
    print(f"{sep}\n")

    # ── 1. Carga y preprocesamiento de datos ─────────────────────────────────
    print("[XAI] Cargando dataset PTB-XL...")
    train_data, val_data, test_data, label_names = load_dataset()
    print(f"[XAI] Clases: {label_names}")

    print("[XAI] Preprocesando ECG (normalización global)...")
    train_ecg, val_ecg, test_ecg = preprocess_ecg_splits(
        train_data["ecg"], val_data["ecg"], test_data["ecg"]
    )

    print("[XAI] Preprocesando variables clínicas...")
    train_clin, val_clin, test_clin, _, _ = preprocess_clinical(
        train_data["clinical"], val_data["clinical"], test_data["clinical"]
    )

    # ── 2. Cargar modelo ─────────────────────────────────────────────────────
    model = load_model()

    # ── 3. Submuestras para el análisis ──────────────────────────────────────
    rng = np.random.default_rng(seed=42)

    # Muestra representativa (para Grad-CAM visual)
    rep_ecg   = test_ecg[REPRESENTATIVE_IDX]
    rep_clin  = test_clin[REPRESENTATIVE_IDX]

    # Batch para lead importance
    lead_idx = rng.choice(len(test_ecg), size=min(N_LEAD_SAMPLES, len(test_ecg)), replace=False)
    lead_ecg  = test_ecg[lead_idx]
    lead_clin = test_clin[lead_idx]

    # Batch para SHAP
    shap_idx  = rng.choice(len(test_ecg), size=min(N_SHAP_SAMPLES, len(test_ecg)), replace=False)
    shap_ecg_test  = test_ecg[shap_idx]
    shap_clin_test = test_clin[shap_idx]

    # Background SHAP (del training)
    bg_idx   = rng.choice(len(train_ecg), size=min(N_SHAP_BACKGROUND, len(train_ecg)), replace=False)
    bg_ecg   = train_ecg[bg_idx]
    bg_clin  = train_clin[bg_idx]

    # ── 4. Grad-CAM ──────────────────────────────────────────────────────────
    print("\n[XAI] Calculando Grad-CAM 1D...")
    gradcam_dir = OUTPUT_DIR / "gradcam"
    cams = compute_gradcam_all_classes(
        model, rep_ecg, rep_clin, label_names
    )
    plot_gradcam_all_classes(rep_ecg, cams, output_dir=gradcam_dir)
    print(f"[XAI] Grad-CAM guardado en {gradcam_dir}/")

    # ── 5. Lead Importance ───────────────────────────────────────────────────
    print("\n[XAI] Calculando Lead Importance (ablación)...")
    lead_dir = OUTPUT_DIR / "lead_importance"
    importances_per_class = compute_lead_importance_per_class(
        model, lead_ecg, lead_clin, label_names
    )
    plot_lead_importance_all_classes(importances_per_class, output_dir=lead_dir)
    plot_lead_importance_heatmap(
        importances_per_class,
        save_path=lead_dir / "lead_importance_heatmap.png",
    )
    # Guardar valores numéricos
    lead_results = {
        cls: imp.tolist() for cls, imp in importances_per_class.items()
    }
    with open(lead_dir / "lead_importance.json", "w") as f:
        json.dump(lead_results, f, indent=2)
    print(f"[XAI] Lead Importance guardado en {lead_dir}/")

    # ── 6. SHAP clínico ──────────────────────────────────────────────────────
    print("\n[XAI] Calculando SHAP values para variables clínicas...")
    print("      (puede tardar varios minutos con KernelExplainer)")
    shap_dir = OUTPUT_DIR / "shap"
    shap_per_class = compute_shap_all_classes(
        model,
        ecg_background=bg_ecg,
        clinical_background=bg_clin,
        clinical_samples=shap_clin_test,
        label_names=label_names,
        n_background=N_SHAP_BACKGROUND,
    )
    plot_shap_all_classes(
        shap_per_class,
        clinical_data=shap_clin_test,
        feature_names=["age", "sex", "height", "weight"],
        output_dir=shap_dir,
    )
    # Guardar importancias medias
    shap_results = {
        cls: mean_abs_shap(sv).tolist()
        for cls, sv in shap_per_class.items()
    }
    with open(shap_dir / "shap_mean_importance.json", "w") as f:
        json.dump(shap_results, f, indent=2)
    print(f"[XAI] SHAP guardado en {shap_dir}/")

    print(f"\n{sep}")
    print(f"  Pipeline XAI completado.")
    print(f"  Resultados en: {OUTPUT_DIR}/")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
