"""
Módulo de carga del dataset PTB-XL.

Responsabilidades:
- Leer ptbxl_database.csv y scp_statements.csv
- Agregar etiquetas SCP en las 5 superclases diagnósticas oficiales
- Cargar señales ECG a 100 Hz con wfdb
- Dividir en train/val/test según la columna strat_fold oficial

Estrategia de etiquetado:
    Se usan las 5 superclases diagnósticas del PTB-XL (Wagner et al., 2020):
      CD   — Conduction Disturbance (trastornos de conducción)
      HYP  — Hypertrophy (hipertrofia)
      MI   — Myocardial Infarction (infarto de miocardio)
      NORM — Normal ECG
      STTC — ST/T Change (cambios ST/T)

    Un registro se etiqueta con la superclase X si al menos uno de sus
    códigos SCP pertenece a X con likelihood >= MIN_CONFIDENCE (100).
    La clasificación es multilabel: un ECG puede tener varias superclases
    activas simultáneamente (p.ej. MI + CD).

Referencias:
    Wagner et al. (2020): PTB-XL, a large publicly available
    electrocardiography dataset. Scientific Data.
    https://doi.org/10.1038/s41597-020-0495-6
    Strodthoff et al. (2021): Deep Learning for ECG Analysis: Benchmarks
    and Insights from PTB-XL. IEEE J. Biomed. Health Inform., 25(5).
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import wfdb

# ---------------------------------------------------------------------------
# Rutas del dataset (relativas a la raíz del proyecto TFM)
# ---------------------------------------------------------------------------
DATA_ROOT = Path("physionet.org/files/ptb-xl/1.0.3")
CSV_PATH  = DATA_ROOT / "ptbxl_database.csv"
SCP_PATH  = DATA_ROOT / "scp_statements.csv"

# ---------------------------------------------------------------------------
# Variables clínicas seleccionadas (en orden)
# NOTA: la columna heart_rate no existe en ptbxl_database.csv.
# Se usan 4 variables: [age, sex, height, weight]
# ---------------------------------------------------------------------------
CLINICAL_COLS = ["age", "sex", "height", "weight"]

# ---------------------------------------------------------------------------
# Split oficial PTB-XL
# ---------------------------------------------------------------------------
TRAIN_FOLDS = list(range(1, 9))   # folds 1–8
VAL_FOLD    = 9
TEST_FOLD   = 10

# ---------------------------------------------------------------------------
# Etiquetas objetivo: 5 superclases diagnósticas (orden alfabético fijo)
# ---------------------------------------------------------------------------
SUPERCLASSES: List[str] = ["CD", "HYP", "MI", "NORM", "STTC"]

# Umbral mínimo de likelihood para considerar una etiqueta como válida.
# Strodthoff et al. (2021) usan 0.0 (sin umbral): se incluyen todos
# los códigos SCP asignados con cualquier likelihood, maximizando
# el número de etiquetas disponibles para entrenamiento.
MIN_CONFIDENCE: float = 0.0


# ---------------------------------------------------------------------------
# Funciones de carga y etiquetado
# ---------------------------------------------------------------------------

def _build_code_to_superclass(scp_diag: pd.DataFrame) -> Dict[str, str]:
    """
    Construye un mapa {código_SCP: superclase} para las 44 etiquetas
    diagnósticas (diagnostic==1).

    Args:
        scp_diag: DataFrame filtrado (diagnostic==1) de scp_statements.csv.

    Returns:
        Diccionario {código_SCP: diagnostic_class}.
    """
    return scp_diag["diagnostic_class"].to_dict()


def load_scp_statements() -> pd.DataFrame:
    """
    Carga scp_statements.csv y devuelve solo las filas con diagnostic==1.

    Returns:
        DataFrame indexado por código SCP con las 44 etiquetas diagnósticas.
    """
    scp = pd.read_csv(SCP_PATH, index_col=0)
    return scp[scp["diagnostic"] == 1]


def parse_scp_codes(raw: str) -> Dict[str, float]:
    """
    Convierte el campo scp_codes (string tipo dict) en un diccionario.

    Usa ast.literal_eval para parsearlo de forma segura sin eval().

    Args:
        raw: String con el diccionario de códigos SCP.

    Returns:
        Diccionario {código_scp: likelihood_float}.
        Devuelve {} si el campo no es parseable.
    """
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}


def build_superclass_vector(
    scp_codes:        Dict[str, float],
    code_to_super:    Dict[str, str],
    confidence_threshold: float = MIN_CONFIDENCE,
) -> np.ndarray:
    """
    Construye el vector binario multilabel de 5 superclases para un registro.

    Un registro activa la superclase X si al menos uno de sus códigos SCP
    pertenece a X con likelihood >= confidence_threshold.

    Args:
        scp_codes:            Dict {código_SCP: likelihood} del registro.
        code_to_super:        Mapa {código_SCP: superclase}.
        confidence_threshold: Umbral mínimo de likelihood (>=).

    Returns:
        Array binario float32 de shape (5,) con el orden de SUPERCLASSES.
    """
    active = set()
    for code, conf in scp_codes.items():
        if conf >= confidence_threshold and code in code_to_super:
            active.add(code_to_super[code])

    return np.array(
        [1.0 if sc in active else 0.0 for sc in SUPERCLASSES],
        dtype=np.float32,
    )


def load_ecg_signal(filename_lr: str) -> np.ndarray:
    """
    Carga una señal ECG de 100 Hz con wfdb y la devuelve como array.

    La columna 'filename_lr' del CSV contiene rutas relativas del
    tipo 'records100/00000/00001_lr'. Se compone con DATA_ROOT para
    obtener la ruta absoluta sin extensión que requiere wfdb.rdrecord.

    A 100 Hz, 10 segundos de ECG producen 1000 muestras.

    Args:
        filename_lr: Ruta relativa al registro (sin extensión .hea/.dat),
                     tal como aparece en la columna filename_lr del CSV.

    Returns:
        Array float32 de shape (1000, 12).

    Raises:
        FileNotFoundError: Si el archivo .dat o .hea no existe.
    """
    full_path = DATA_ROOT / filename_lr
    record = wfdb.rdrecord(str(full_path))
    signal = record.p_signal  # shape: (1000, 12) para 100 Hz
    if signal.shape != (1000, 12):
        raise ValueError(
            f"Shape inesperado {signal.shape} en {filename_lr}. "
            "Comprueba que estás usando records100/ (100 Hz)."
        )
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline principal de carga
# ---------------------------------------------------------------------------

def _load_split(
    split_df:      pd.DataFrame,
    code_to_super: Dict[str, str],
    split_name:    str = "",
) -> Dict:
    """
    Carga señales ECG, datos clínicos y etiquetas de superclase para un split.

    Los registros que no se puedan cargar (archivo corrupto o ausente)
    se omiten con un aviso sin interrumpir el proceso.

    Args:
        split_df:      DataFrame con las filas del split actual.
        code_to_super: Mapa {código_SCP: superclase} para 44 etiquetas diag.
        split_name:    Nombre del split para mensajes de log.

    Returns:
        Diccionario con claves:
            'ecg':      np.ndarray (N, 1000, 12)
            'clinical': np.ndarray (N, 4)  — puede tener NaN
            'labels':   np.ndarray (N, 5)  — vector de 5 superclases
            'ids':      list de ecg_id int
    """
    ecg_list      = []
    clinical_list = []
    label_list_   = []
    ids_list      = []
    skipped       = 0

    for ecg_id, row in split_df.iterrows():
        try:
            signal = load_ecg_signal(row["filename_lr"])
        except Exception as exc:
            print(f"[Loader] WARN [{split_name}] ecg_id={ecg_id}: {exc}")
            skipped += 1
            continue

        ecg_list.append(signal)
        clinical_list.append(row[CLINICAL_COLS].values.astype(np.float32))
        label_list_.append(
            build_superclass_vector(row["scp_codes"], code_to_super)
        )
        ids_list.append(ecg_id)

    if skipped:
        print(f"[Loader] [{split_name}] Registros omitidos: {skipped}")

    return {
        "ecg":      np.stack(ecg_list),
        "clinical": np.stack(clinical_list),
        "labels":   np.stack(label_list_),
        "ids":      ids_list,
    }


def load_dataset() -> Tuple[Dict, Dict, Dict, List[str]]:
    """
    Carga el dataset PTB-XL completo y lo divide en train/val/test.

    Flujo:
    1. Lee ptbxl_database.csv (21 799 registros)
    2. Parsea el campo scp_codes
    3. Construye el mapa código→superclase desde scp_statements.csv
    4. Divide por strat_fold: train=1-8, val=9, test=10
    5. Carga señales ECG y construye vectores de 5 superclases por split

    Returns:
        Tupla (train_data, val_data, test_data, label_list):
            - train/val/test: dict con 'ecg' (N,1000,12),
              'clinical' (N,4), 'labels' (N,5), 'ids' (list)
            - label_list: ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    """
    # 1. Cargar metadatos
    df = pd.read_csv(CSV_PATH, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(parse_scp_codes)

    # 2. Construir mapa código SCP → superclase
    scp_diag      = load_scp_statements()
    code_to_super = _build_code_to_superclass(scp_diag)
    print(f"[Loader] Variable objetivo: {len(SUPERCLASSES)} superclases → {SUPERCLASSES}")

    # 3. Dividir por strat_fold
    train_df = df[df["strat_fold"].isin(TRAIN_FOLDS)].copy()
    val_df   = df[df["strat_fold"] == VAL_FOLD].copy()
    test_df  = df[df["strat_fold"] == TEST_FOLD].copy()
    print(f"[Loader] Registros → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 4. Cargar señales ECG para cada split
    train_data = _load_split(train_df, code_to_super, "train")
    val_data   = _load_split(val_df,   code_to_super, "val")
    test_data  = _load_split(test_df,  code_to_super, "test")

    return train_data, val_data, test_data, SUPERCLASSES


def get_label_names() -> List[str]:
    """
    Devuelve la lista de las 5 superclases diagnósticas.

    Returns:
        ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    """
    return SUPERCLASSES
