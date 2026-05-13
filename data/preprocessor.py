"""
Módulo de preprocesamiento del dataset PTB-XL.

Responsabilidades:
- Normalización z-score GLOBAL de señales ECG (media y std calculados
  sobre todos los valores del conjunto de entrenamiento, como en
  Strodthoff et al. 2021). Los estadísticos se guardan en disco.
- Imputación de valores nulos con la mediana del conjunto de entrenamiento
- Escalado de variables clínicas continuas con StandardScaler
- Guardado y carga de artefactos (scaler, medianas, ecg_stats) con joblib

Diferencia respecto a la normalización por-derivación/por-registro:
  La normalización global preserva diferencias relativas de amplitud
  entre derivaciones y entre pacientes, que son clínicamente relevantes
  (p.ej. QRS de bajo voltaje en CD, ondas altas en HYP).

Referencias:
    Strodthoff et al. (2021): Deep Learning for ECG Analysis:
    Benchmarks and Insights from PTB-XL. IEEE JBHI 25(5):1519-1528.
    https://doi.org/10.1109/JBHI.2020.3022989
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Rutas de los artefactos guardados
# ---------------------------------------------------------------------------
_BASE = Path("Desarrollo/tfm-ecg/saved_model")
SCALER_PATH    = _BASE / "scaler.joblib"
MEDIANS_PATH   = _BASE / "train_medians.joblib"
ECG_STATS_PATH = _BASE / "ecg_global_stats.joblib"  # {mean, std} de la señal

# ---------------------------------------------------------------------------
# Índices de las columnas en el vector clínico (4,)
# Orden: [age, sex, height, weight]
# NOTA: heart_rate no existe en ptbxl_database.csv y fue eliminada.
# ---------------------------------------------------------------------------
CONTINUOUS_IDX = [0, 2, 3]   # age, height, weight
BINARY_IDX     = [1]          # sex — no se escala ni imputa


# ===========================================================================
# PREPROCESAMIENTO DE SEÑAL ECG — NORMALIZACIÓN GLOBAL
# ===========================================================================

def _apply_global_stats(
    signals: np.ndarray,
    mean:    float,
    std:     float,
) -> np.ndarray:
    """
    Aplica z-score global (un único mean/std para toda la señal).

    Args:
        signals: Array (..., T, 12) de señales ECG brutas.
        mean:    Media global calculada sobre el conjunto de train.
        std:     Desviación estándar global calculada sobre train.

    Returns:
        Array float32 normalizado de la misma shape.
    """
    return ((signals.astype(np.float32) - mean) / (std + 1e-8)).astype(np.float32)


def preprocess_ecg_splits(
    train_ecg: np.ndarray,
    val_ecg:   np.ndarray,
    test_ecg:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalización z-score GLOBAL de los tres splits de señales ECG.

    A diferencia de la normalización por-derivación/por-registro,
    aquí se calcula UN ÚNICO mean y std sobre todos los valores del
    conjunto de entrenamiento (aplanado: N_train × T × 12 valores).
    Val y test se normalizan con los mismos estadísticos.

    Estrategia adoptada de Strodthoff et al. (2021), que es el
    benchmark de referencia para el dataset PTB-XL. Preserva:
      - Diferencias de amplitud entre derivaciones (relevante para HYP)
      - Diferencias de amplitud entre pacientes (relevante para CD, MI)

    Los estadísticos se guardan en disco para reproducir el
    preprocesamiento en inferencia.

    Args:
        train_ecg: Array (N_train, T, 12) señales brutas.
        val_ecg:   Array (N_val,   T, 12) señales brutas.
        test_ecg:  Array (N_test,  T, 12) señales brutas.

    Returns:
        Tupla (train_norm, val_norm, test_norm).
    """
    # Calcular estadísticos sobre TODOS los valores de train
    # SIN flatten() para evitar copiar ~4GB adicionales en memoria
    # np.mean/std de array (N, T, 12) es equivalente a mean/std del array aplanado
    train_f32 = train_ecg if train_ecg.dtype == np.float32 else train_ecg.astype(np.float32)
    global_mean = float(np.mean(train_f32))
    global_std  = float(np.std(train_f32))
    print(f"[Preprocessor] ECG global — mean={global_mean:.4f}, std={global_std:.4f}")

    # Guardar artefactos para inferencia
    ECG_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mean": global_mean, "std": global_std}, ECG_STATS_PATH)
    print(f"[Preprocessor] ECG stats guardados en {ECG_STATS_PATH}")

    print("[Preprocessor] Normalizando ECG — train...")
    train_norm = _apply_global_stats(train_ecg, global_mean, global_std)
    print("[Preprocessor] Normalizando ECG — val...")
    val_norm   = _apply_global_stats(val_ecg, global_mean, global_std)
    print("[Preprocessor] Normalizando ECG — test...")
    test_norm  = _apply_global_stats(test_ecg, global_mean, global_std)
    return train_norm, val_norm, test_norm


# ===========================================================================
# PREPROCESAMIENTO DE VARIABLES CLÍNICAS
# ===========================================================================

def compute_train_medians(clinical_train: np.ndarray) -> np.ndarray:
    """
    Calcula las medianas de las variables clínicas continuas en train.

    Las medianas se calculan ignorando NaN (np.nanmedian), de modo que
    registros con datos faltantes no distorsionen el estadístico.
    Las posiciones de variables binarias (sex) se dejan en 0.0.

    Args:
        clinical_train: Array (N_train, 4) con posibles NaN en
                        age, height, weight.

    Returns:
        Array float32 de shape (4,) con medianas; 0.0 en posición sex.
    """
    medians = np.zeros(4, dtype=np.float32)
    for idx in CONTINUOUS_IDX:
        medians[idx] = float(np.nanmedian(clinical_train[:, idx]))
    return medians


def impute_missing_values(
    clinical:      np.ndarray,
    train_medians: np.ndarray,
) -> np.ndarray:
    """
    Imputa valores NaN usando las medianas del conjunto de entrenamiento.

    Solo se imputan las columnas continuas (CONTINUOUS_IDX).
    La columna sex (binaria) no se modifica.

    Args:
        clinical:      Array (N, 5) con posibles NaN.
        train_medians: Array (5,) de medianas calculadas en train.

    Returns:
        Array float32 (N, 5) sin valores NaN.
    """
    imputed = clinical.copy()
    for idx in CONTINUOUS_IDX:
        nan_mask = np.isnan(imputed[:, idx])
        if nan_mask.any():
            imputed[nan_mask, idx] = train_medians[idx]
    return imputed


def fit_scaler(clinical_train_imputed: np.ndarray) -> StandardScaler:
    """
    Ajusta un StandardScaler sobre las columnas continuas de train.

    El scaler se ajusta ÚNICAMENTE con datos de entrenamiento ya
    imputados, garantizando que la media y varianza de normalización
    no incluyan información de val/test (data leakage).

    La columna sex (índice 1) se excluye del ajuste.

    Args:
        clinical_train_imputed: Array (N_train, 5) sin NaN.

    Returns:
        StandardScaler ajustado sobre las columnas continuas.
    """
    scaler = StandardScaler()
    scaler.fit(clinical_train_imputed[:, CONTINUOUS_IDX])
    return scaler


def apply_scaler(
    clinical: np.ndarray,
    scaler:   StandardScaler,
) -> np.ndarray:
    """
    Aplica el StandardScaler a las columnas continuas de un split.

    La columna sex (binaria) se copia sin modificar.

    Args:
        clinical: Array (N, 5) ya imputado, sin NaN.
        scaler:   StandardScaler previamente ajustado sobre train.

    Returns:
        Array float32 (N, 5) con columnas continuas escaladas.
    """
    scaled = clinical.copy()
    scaled[:, CONTINUOUS_IDX] = scaler.transform(
        clinical[:, CONTINUOUS_IDX]
    )
    return scaled


def preprocess_clinical(
    train_clinical: np.ndarray,
    val_clinical:   np.ndarray,
    test_clinical:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]:
    """
    Pipeline completo de preprocesamiento para variables clínicas.

    Pasos en orden:
    1. Calcular medianas de train (ignora NaN)
    2. Imputar NaN en train, val y test con medianas de train
    3. Ajustar StandardScaler sobre train imputado
    4. Aplicar scaler a train, val y test
    5. Persistir scaler y medianas en disco (joblib)

    La persistencia de artefactos garantiza que en inferencia se
    pueda reproducir exactamente el mismo preprocesamiento.

    Args:
        train_clinical: Array (N_train, 5) con posibles NaN.
        val_clinical:   Array (N_val,   5) con posibles NaN.
        test_clinical:  Array (N_test,  5) con posibles NaN.

    Returns:
        Tupla (train_scaled, val_scaled, test_scaled, scaler, medians).
    """
    # 1. Medianas del train
    train_medians = compute_train_medians(train_clinical)
    print(f"[Preprocessor] Medianas de train: age={train_medians[0]:.1f}, "
          f"height={train_medians[2]:.1f}, weight={train_medians[3]:.1f}")

    # 2. Imputación
    train_imp = impute_missing_values(train_clinical, train_medians)
    val_imp   = impute_missing_values(val_clinical,   train_medians)
    test_imp  = impute_missing_values(test_clinical,  train_medians)

    # 3. Ajustar scaler sobre train
    scaler = fit_scaler(train_imp)

    # 4. Aplicar scaler
    train_scaled = apply_scaler(train_imp, scaler)
    val_scaled   = apply_scaler(val_imp,   scaler)
    test_scaled  = apply_scaler(test_imp,  scaler)

    # 5. Persistir artefactos
    _BASE.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(train_medians, MEDIANS_PATH)
    print(f"[Preprocessor] Scaler guardado  → {SCALER_PATH}")
    print(f"[Preprocessor] Medianas guardadas → {MEDIANS_PATH}")

    return train_scaled, val_scaled, test_scaled, scaler, train_medians


# ===========================================================================
# CARGA DE ARTEFACTOS (para inferencia)
# ===========================================================================

def load_preprocessing_artifacts() -> Tuple[StandardScaler, np.ndarray]:
    """
    Carga el scaler y las medianas guardadas para uso en inferencia.

    Returns:
        Tupla (scaler, train_medians).

    Raises:
        FileNotFoundError: Si los archivos no existen; indica que el
                           pipeline de entrenamiento no se ha ejecutado.
    """
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler no encontrado en {SCALER_PATH}. "
            "Ejecuta primero el pipeline completo (main.py)."
        )
    scaler        = joblib.load(SCALER_PATH)
    train_medians = joblib.load(MEDIANS_PATH)
    return scaler, train_medians


def preprocess_single_record(
    ecg_signal:        np.ndarray,
    clinical_features: np.ndarray,
    scaler:            StandardScaler,
    train_medians:     np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa un único registro nuevo para inferencia individual.

    Aplica exactamente el mismo pipeline que en entrenamiento:
    normalización z-score de ECG e imputación + escalado de clínicos.

    Args:
        ecg_signal:        Array (1000, 12) con la señal ECG bruta.
        clinical_features: Array (5,) con [age, sex, height, weight, hr].
                           Puede contener NaN en variables continuas.
        scaler:            StandardScaler cargado desde disco.
        train_medians:     Medianas de train cargadas desde disco.

    Returns:
        Tupla (ecg_norm, clinical_scaled):
        - ecg_norm:       shape (1, 1000, 12) — listo para model.predict()
        - clinical_scaled: shape (1, 4)        — listo para model.predict()
    """
    # Normalizar ECG y añadir dimensión batch
    ecg_norm = normalize_ecg(ecg_signal)[np.newaxis, ...]  # (1, 1000, 12)

    # Imputar y escalar variables clínicas
    clinical_2d     = clinical_features[np.newaxis, ...]       # (1, 4)
    clinical_imp    = impute_missing_values(clinical_2d, train_medians)
    clinical_scaled = apply_scaler(clinical_imp, scaler)        # (1, 4)

    return ecg_norm, clinical_scaled
