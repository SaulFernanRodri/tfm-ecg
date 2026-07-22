#!/usr/bin/env python3
"""
validate_loader_csv.py
======================
Valida la carga y el etiquetado CSV de data/loader.py sin señales reales
ni el dataset PTB-XL completo.

Estrategia
----------
* Se crean dos CSV mínimos en /tmp/tfm_loader_test/ (10 registros).
* Se parchea `loader_mod.load_ecg_signal` para devolver un array
  zeros(1000,12) en lugar de llamar a wfdb.  El resto del código
  de loader.py se ejecuta sin modificaciones.
* Todos los paths del módulo se redirigen a los CSV de prueba.

Ejecutar desde la raíz del repo:
    cd /home/saul/IA/TFM/Desarrollo/tfm-ecg
    python3 validate_loader_csv.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Añadir el repo al path ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import data.loader as loader_mod

# ── Rutas a los CSV de prueba (creados en /tmp/ por separado) ────────────────
TEST_DIR  = Path("/tmp/tfm_loader_test")
SCP_PATH  = TEST_DIR / "scp_statements.csv"
DB_PATH   = TEST_DIR / "ptbxl_database.csv"

for p in (SCP_PATH, DB_PATH):
    if not p.exists():
        sys.exit(f"ERROR: fichero de prueba no encontrado: {p}\n"
                 "       Ejecuta primero el bloque shell que crea los CSV.")

# ── Patch: redirigir rutas del módulo a los CSV de prueba ────────────────────
loader_mod.CSV_PATH = DB_PATH
loader_mod.SCP_PATH = SCP_PATH

# ── Patch: load_ecg_signal → zeros(1000,12) sin wfdb ────────────────────────
def _mock_ecg_signal(filename_lr: str) -> np.ndarray:
    """Sustituye wfdb.rdrecord devolviendo un array de ceros (1000,12)."""
    return np.zeros((1000, 12), dtype=np.float32)

loader_mod.load_ecg_signal = _mock_ecg_signal

# ── Helpers de reporte ────────────────────────────────────────────────────────
_failures: list[str] = []

def check(condition: bool, msg_ok: str, msg_fail: str = "") -> bool:
    if condition:
        print(f"  ✓  {msg_ok}")
    else:
        detail = msg_fail or msg_ok
        print(f"  ✗  {detail}")
        _failures.append(detail)
    return condition


# ════════════════════════════════════════════════════════════════════════════════
# Sección 1 — parse_scp_codes
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ Sección 1 — parse_scp_codes ══════════════════════════════════")

r = loader_mod.parse_scp_codes("{'NORM': 100.0}")
check(r == {"NORM": 100.0},
      f"Dict monoetiqueta → {r}",
      f"Esperaba {{'NORM':100.0}}, obtuvo {r}")

r = loader_mod.parse_scp_codes("{'IMI': 100.0, 'LBBB': 100.0}")
check(r == {"IMI": 100.0, "LBBB": 100.0},
      f"Dict multilabel → {r}",
      f"Esperaba {{'IMI':100.0,'LBBB':100.0}}, obtuvo {r}")

r = loader_mod.parse_scp_codes("malformed_not_a_dict")
check(r == {},
      "Malformado → devuelve {} sin excepción",
      f"Esperaba {{}}, obtuvo {r}")

r = loader_mod.parse_scp_codes("")
check(r == {},
      "String vacío → devuelve {} sin excepción",
      f"Esperaba {{}}, obtuvo {r}")


# ════════════════════════════════════════════════════════════════════════════════
# Sección 2 — build_superclass_vector (unit tests directos)
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ Sección 2 — build_superclass_vector ══════════════════════════")

# Construir code_to_super desde el CSV de prueba
_scp_df    = pd.read_csv(SCP_PATH, index_col=0)
_scp_diag  = _scp_df[_scp_df["diagnostic"] == 1]
code_to_super = loader_mod._build_code_to_superclass(_scp_diag)
print(f"  code_to_super construido: {code_to_super}")

# SUPERCLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
CASES = [
    ({"NORM":  100.0},                  [0, 0, 0, 1, 0], "NORM       → [0,0,0,1,0]"),
    ({"LVH":   100.0},                  [0, 1, 0, 0, 0], "HYP (LVH)  → [0,1,0,0,0]"),
    ({"IMI":   100.0},                  [0, 0, 1, 0, 0], "MI  (IMI)  → [0,0,1,0,0]"),
    ({"LBBB":  100.0},                  [1, 0, 0, 0, 0], "CD  (LBBB) → [1,0,0,0,0]"),
    ({"NST_":  100.0},                  [0, 0, 0, 0, 1], "STTC(NST_) → [0,0,0,0,1]"),
    ({"IMI":  100.0, "LBBB": 100.0},   [1, 0, 1, 0, 0], "MI+CD multilabel → [1,0,1,0,0]"),
    ({},                                [0, 0, 0, 0, 0], "Vacío (malformado) → [0,0,0,0,0]"),
    ({"NORM": 100.0, "SR": 100.0},      [0, 0, 0, 1, 0], "NORM+SR(no-diag) → SR ignorado → [0,0,0,1,0]"),
]

for codes, expected, label in CASES:
    vec      = loader_mod.build_superclass_vector(codes, code_to_super)
    expected_arr = np.array(expected, dtype=np.float32)
    ok = np.array_equal(vec, expected_arr)
    check(ok, label, f"{label}  → GOT {vec.tolist()}, EXPECTED {expected}")


# ════════════════════════════════════════════════════════════════════════════════
# Sección 3 — split por strat_fold (lógica CSV, sin cargar señales)
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ Sección 3 — split por strat_fold ═════════════════════════════")

_df = pd.read_csv(DB_PATH, index_col="ecg_id")
_df["scp_codes"] = _df["scp_codes"].apply(loader_mod.parse_scp_codes)

train_df = _df[_df["strat_fold"].isin(loader_mod.TRAIN_FOLDS)]
val_df   = _df[_df["strat_fold"] == loader_mod.VAL_FOLD]
test_df  = _df[_df["strat_fold"] == loader_mod.TEST_FOLD]

check(len(train_df) == 8,
      f"Train: {len(train_df)} registros (esperado 8)",
      f"Train: {len(train_df)} registros — ESPERADO 8")

check(len(val_df) == 1,
      f"Val:   {len(val_df)} registro  (esperado 1)",
      f"Val: {len(val_df)} registros — ESPERADO 1")

check(len(test_df) == 1,
      f"Test:  {len(test_df)} registro  (esperado 1)",
      f"Test: {len(test_df)} registros — ESPERADO 1")

check(set(train_df.index) == {1, 2, 3, 4, 5, 6, 7, 8},
      f"Train ecg_ids = {sorted(train_df.index.tolist())}",
      f"Train ecg_ids incorrectos: {sorted(train_df.index.tolist())}")

check(val_df.index.tolist() == [9],
      f"Val ecg_id   = {val_df.index.tolist()}",
      f"Val ecg_id incorrectos: {val_df.index.tolist()}")

check(test_df.index.tolist() == [10],
      f"Test ecg_id  = {test_df.index.tolist()}",
      f"Test ecg_id incorrectos: {test_df.index.tolist()}")


# ════════════════════════════════════════════════════════════════════════════════
# Sección 4 — load_dataset() end-to-end (con load_ecg_signal mockeada)
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ Sección 4 — load_dataset() end-to-end ════════════════════════")
print("  (load_ecg_signal parcheada → zeros(1000,12), sin wfdb real)")
print()

train, val, test, label_list = loader_mod.load_dataset()

# label_list correcta
check(label_list == loader_mod.SUPERCLASSES,
      f"label_list = {label_list}",
      f"label_list incorrecta: {label_list}")

# shapes y dtypes por split
for split_name, split in [("train", train), ("val", val), ("test", test)]:
    N = len(split["ids"])

    check(split["ecg"].shape      == (N, 1000, 12),
          f"{split_name}['ecg'].shape      = {split['ecg'].shape}",
          f"{split_name}['ecg'].shape = {split['ecg'].shape}, ESPERADO ({N},1000,12)")

    check(split["clinical"].shape == (N, 4),
          f"{split_name}['clinical'].shape = {split['clinical'].shape}",
          f"{split_name}['clinical'].shape = {split['clinical'].shape}, ESPERADO ({N},4)")

    check(split["labels"].shape   == (N, 5),
          f"{split_name}['labels'].shape   = {split['labels'].shape}",
          f"{split_name}['labels'].shape = {split['labels'].shape}, ESPERADO ({N},5)")

    check(split["ecg"].dtype      == np.float32,
          f"{split_name}['ecg'].dtype      = {split['ecg'].dtype}",
          f"{split_name}['ecg'].dtype = {split['ecg'].dtype}, ESPERADO float32")

    check(split["labels"].dtype   == np.float32,
          f"{split_name}['labels'].dtype   = {split['labels'].dtype}",
          f"{split_name}['labels'].dtype = {split['labels'].dtype}, ESPERADO float32")

    check(isinstance(split["ids"], list),
          f"{split_name}['ids'] es list, len={len(split['ids'])}, vals={split['ids']}",
          f"{split_name}['ids'] NO es list: {type(split['ids'])}")


# ════════════════════════════════════════════════════════════════════════════════
# Sección 5 — etiquetas individuales dentro del split train
# ════════════════════════════════════════════════════════════════════════════════
print("\n══ Sección 5 — etiquetas en el split train ══════════════════════")

ids_arr = np.array(train["ids"])

EXPECTED_LABELS = {
    # ecg_id → (expected_label, descripción)
    1: ([0, 0, 0, 1, 0], "NORM"),
    2: ([0, 1, 0, 0, 0], "HYP (LVH)"),
    3: ([0, 0, 1, 0, 0], "MI  (IMI)"),
    4: ([1, 0, 0, 0, 0], "CD  (LBBB)"),
    5: ([0, 0, 0, 0, 1], "STTC (NST_)"),
    6: ([1, 0, 1, 0, 0], "MI + CD multilabel"),
    7: ([0, 0, 0, 0, 0], "malformado → vector cero"),
    8: ([0, 0, 0, 1, 0], "NaN clínicos → label NORM OK"),
}

for ecg_id, (exp, desc) in EXPECTED_LABELS.items():
    idx = np.where(ids_arr == ecg_id)[0]
    if len(idx) == 0:
        check(False, "", f"ecg_id={ecg_id} ({desc}): NO encontrado en train")
        continue
    got = train["labels"][idx[0]].tolist()
    expected_f = [float(x) for x in exp]
    check(got == expected_f,
          f"ecg_id={ecg_id} ({desc}): label={got}",
          f"ecg_id={ecg_id} ({desc}): GOT {got}, ESPERADO {expected_f}")

# Verificar que el registro con NaN clínicos tiene NaN en clinical array
idx8 = np.where(ids_arr == 8)[0]
if len(idx8) > 0:
    clin8 = train["clinical"][idx8[0]]
    has_nan = bool(np.any(np.isnan(clin8)))
    check(has_nan,
          f"ecg_id=8 clinical contiene NaN como se espera: {clin8}",
          f"ecg_id=8 clinical debería tener NaN pero obtuvo: {clin8}")

    # age y height deberían ser NaN; sex=0.0
    check(np.isnan(clin8[0]),   # age
          f"  ecg_id=8 age=NaN    ✓ ({clin8[0]})",
          f"  ecg_id=8 age debería ser NaN, obtuvo {clin8[0]}")
    check(float(clin8[1]) == 0.0,  # sex
          f"  ecg_id=8 sex=0.0    ✓ ({clin8[1]})",
          f"  ecg_id=8 sex debería ser 0.0, obtuvo {clin8[1]}")
    check(np.isnan(clin8[2]),   # height
          f"  ecg_id=8 height=NaN ✓ ({clin8[2]})",
          f"  ecg_id=8 height debería ser NaN, obtuvo {clin8[2]}")
    check(np.isnan(clin8[3]),   # weight
          f"  ecg_id=8 weight=NaN ✓ ({clin8[3]})",
          f"  ecg_id=8 weight debería ser NaN, obtuvo {clin8[3]}")


# ════════════════════════════════════════════════════════════════════════════════
# Resumen
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
if not _failures:
    print("  TODOS LOS CHECKS PASARON (0 fallos)")
else:
    print(f"  {len(_failures)} CHECK(S) FALLARON:")
    for f in _failures:
        print(f"    · {f}")
print("═" * 60)

sys.exit(0 if not _failures else 1)
