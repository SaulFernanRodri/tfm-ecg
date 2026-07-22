"""
Sube todos los ficheros del Space a HuggingFace.

Requisitos previos:
  1. huggingface-cli login   (pide el token una sola vez)
  2. Haber ejecutado:  python Desarrollo/tfm-ecg/scripts/export_demo_data.py

Uso:
  cd /home/saul/IA/TFM
  python Desarrollo/tfm-ecg/scripts/upload_to_hf.py
"""

from pathlib import Path
from huggingface_hub import HfApi

# ── Configuración ────────────────────────────────────────────────────────────
HF_USER   = "SaulFernanRodri"
SPACE_NAME = "ecg-diagnosis-ai"
REPO_ID   = f"{HF_USER}/{SPACE_NAME}"

TFM_ROOT  = Path(__file__).resolve().parent.parent.parent.parent.parent
HF_DIR    = TFM_ROOT / "Desarrollo" / "tfm-ecg" / "deployment" / "app"
ECG_DIR   = TFM_ROOT / "Desarrollo" / "tfm-ecg"

api = HfApi()

# Crear el Space si no existe
try:
    api.repo_info(repo_id=REPO_ID, repo_type="space")
    print(f"[HF] Space ya existe: {REPO_ID}")
except Exception:
    print(f"[HF] Creando Space: {REPO_ID} ...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
    )
    print("[HF] Space creado.")

def upload(local: Path, remote: str):
    print(f"  ↑ {remote}")
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=remote,
        repo_id=REPO_ID,
        repo_type="space",
    )

print(f"\n[HF] Subiendo ficheros a: {REPO_ID}\n")

# ── Ficheros raíz del Space ──────────────────────────────────────────────────
upload(HF_DIR / "app.py",          "app.py")
upload(HF_DIR / "pages" / "1_inicio.py",   "pages/1_inicio.py")
upload(HF_DIR / "pages" / "2_test_simulador.py","pages/2_test_simulador.py")
upload(HF_DIR / "pages" / "3_simulador_csv.py","pages/3_simulador_csv.py")
upload(HF_DIR / "requirements.txt","requirements.txt")
upload(HF_DIR / "README.md",       "README.md")
upload(HF_DIR / "Dockerfile",      "Dockerfile")
upload(HF_DIR / ".streamlit" / "config.toml", ".streamlit/config.toml")

# ── Assets (imágenes estáticas) ───────────────────────────────────────────────
ASSETS_DIR = HF_DIR / "assets"
if ASSETS_DIR.exists():
    for asset in sorted(ASSETS_DIR.iterdir()):
        if asset.is_file():
            upload(asset, f"assets/{asset.name}")
else:
    print("  [INFO] No existe assets/ — se omite.")

# ── Módulo app_utils/ (UI compartida) ─────────────────────────────────────────
upload(HF_DIR / "app_utils" / "__init__.py", "app_utils/__init__.py")
upload(HF_DIR / "app_utils" / "ui.py",       "app_utils/ui.py")

# Limpieza: eliminar el paquete legacy 'utils/' (renombrado a 'app_utils/' para
# evitar colisión con un futuro utils/ de la raíz del proyecto).
for legacy in ("utils/__init__.py", "utils/ui.py"):
    try:
        api.delete_file(path_in_repo=legacy, repo_id=REPO_ID, repo_type="space")
        print(f"  ✗ eliminado legacy: {legacy}")
    except Exception:
        pass

# ── Módulo model/ ────────────────────────────────────────────────────────────
upload(ECG_DIR / "model" / "__init__.py", "model/__init__.py")
upload(ECG_DIR / "model" / "losses.py",  "model/losses.py")

# ── Módulo xai/ ──────────────────────────────────────────────────────────────
upload(ECG_DIR / "xai" / "__init__.py",       "xai/__init__.py")
upload(ECG_DIR / "xai" / "gradcam.py",        "xai/gradcam.py")
upload(ECG_DIR / "xai" / "lead_importance.py","xai/lead_importance.py")
upload(ECG_DIR / "xai" / "clinical_ablation.py","xai/clinical_ablation.py")

# ── Artefactos del modelo (saved_model/) ─────────────────────────────────────
SAVED = TFM_ROOT / "Desarrollo" / "tfm-ecg" / "saved_model"
upload(SAVED / "ecg_global_stats.joblib",  "saved_model/ecg_global_stats.joblib")
upload(SAVED / "scaler.joblib",            "saved_model/scaler.joblib")
upload(SAVED / "train_medians.joblib",     "saved_model/train_medians.joblib")
upload(SAVED / "v6.2" / "optimal_thresholds.json", "saved_model/v6.2/optimal_thresholds.json")


# Umbrales v6.1 (F0.5-score) si existen
v61_thr = SAVED / "v6.1" / "optimal_thresholds.json"
if v61_thr.exists():
    print("  ↑ saved_model/v6.1/optimal_thresholds.json")
    api.upload_file(
        path_or_fileobj=str(v61_thr),
        path_in_repo="saved_model/v6.1/optimal_thresholds.json",
        repo_id=REPO_ID,
        repo_type="space",
    )

print("  ↑ saved_model/v5/best_model.keras  (fichero grande, puede tardar...)")
api.upload_file(
    path_or_fileobj=str(SAVED / "v5" / "best_model.keras"),
    path_in_repo="saved_model/v5/best_model.keras",
    repo_id=REPO_ID,
    repo_type="space",
)

# ── Datos de demo ────────────────────────────────────────────────────────────
DEMO = HF_DIR / "demo_data"
if not DEMO.exists():
    print("\n[ERROR] No existe demo_data/. Ejecuta primero:")
    print("  python Desarrollo/tfm-ecg/scripts/export_demo_data.py")
    raise SystemExit(1)

upload(DEMO / "ecg_samples.npy",  "demo_data/ecg_samples.npy")
upload(DEMO / "clin_samples.npy", "demo_data/clin_samples.npy")
upload(DEMO / "true_labels.npy",  "demo_data/true_labels.npy")

print(f"\n✅ Space disponible en: https://huggingface.co/spaces/{REPO_ID}")
