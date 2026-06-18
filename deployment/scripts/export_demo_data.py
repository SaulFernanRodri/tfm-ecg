"""
Exporta muestras de demo del test set para el despliegue en HuggingFace Spaces.

Genera 3 ficheros .npy en hf/demo_data/:
    ecg_samples.npy    — (50, 1000, 12) señales preprocesadas
    clin_samples.npy   — (50, 4)        variables clínicas preprocesadas
    true_labels.npy    — (50, 5)        etiquetas reales

Uso (desde la raíz del TFM):
    cd /home/saul/IA/TFM
    python Desarrollo/tfm-ecg/scripts/export_demo_data.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from data.loader import load_dataset
from data.preprocessor import (
    load_preprocessing_artifacts,
    _apply_global_stats,
    apply_scaler,
    impute_missing_values,
    ECG_STATS_PATH,
)
import joblib

N_SAMPLES  = 50
OUTPUT_DIR = _ROOT / "deployment" / "app" / "demo_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[Export] Cargando dataset PTB-XL (solo test)...")
_, _, test_data, label_names = load_dataset()
print(f"[Export] Clases: {label_names}")

# Cargar artefactos ya ajustados en entrenamiento — NO re-ajustar
scaler, train_medians = load_preprocessing_artifacts()
ecg_stats = joblib.load(ECG_STATS_PATH)
print(f"[Export] Artefactos cargados — ECG mean={ecg_stats['mean']:.4f}, std={ecg_stats['std']:.4f}")

print("[Export] Normalizando ECG (global z-score con stats de train)...")
test_ecg = _apply_global_stats(test_data["ecg"], ecg_stats["mean"], ecg_stats["std"])

print("[Export] Escalando variables clínicas con scaler de train...")
test_clin_imp = impute_missing_values(test_data["clinical"], train_medians)
test_clin     = apply_scaler(test_clin_imp, scaler)

# Guardar primeras N_SAMPLES del test set
ecg_out  = test_ecg[:N_SAMPLES].astype(np.float32)
clin_out = test_clin[:N_SAMPLES].astype(np.float32)
lbl_out  = test_data["labels"][:N_SAMPLES].astype(np.float32)

np.save(OUTPUT_DIR / "ecg_samples.npy",  ecg_out)
np.save(OUTPUT_DIR / "clin_samples.npy", clin_out)
np.save(OUTPUT_DIR / "true_labels.npy",  lbl_out)

print(f"[Export] Guardado en {OUTPUT_DIR}/")
print(f"  ecg_samples.npy  {ecg_out.shape}")
print(f"  clin_samples.npy {clin_out.shape}")
print(f"  true_labels.npy  {lbl_out.shape}")
